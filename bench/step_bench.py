"""Step time and memory of the four optimizers, on the machine it runs on.

What is measured: the float64 optimizers of lpopt.optimizers running in
native NumPy arithmetic (float64, float32, float16). Not the emulation of
lpopt.lowprec, which is 10 to 20 times slower and models no real hardware.

For each optimizer, dtype and parameter count:
* step time: 20 warm-up steps, then STEPS steps timed one by one with
  perf_counter_ns. Reported as median and interquartile range, which a
  single slow step (an OS interrupt, a page fault) does not move.
* optimizer state: bytes held in the optimizer's state after the run.
  Expected: 0 for SGD, 1 x the parameters for Momentum (the buffer), 2 x for
  Adam and AdamW (m and v).
* transient memory: peak of the allocations made during one step, from
  tracemalloc (NumPy reports its buffers to it). These are the temporary
  arrays (g * g, sqrt(v), ...) that a device also has to fit in RAM.

The parameters are a single array. The gradient is a fixed random array
with a standard deviation of 1e-2.

Adam and AdamW in native float16 produce infinite parameters: eps = 1e-8
is 0 in float16 and some denominators are 0 (hypothesis H1 of the
low-precision study). They are still timed, since the arithmetic costs the
same, and the column params_finite says that the configuration is not
usable as it is.

    python -m bench.step_bench --tag pc

Writes results/bench_<tag>.csv and results/bench_<tag>_machine.json.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import platform
import subprocess
import time
import tracemalloc
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from lpopt.optimizers import SGD, Adam, AdamW, Momentum

RESULTS = Path(__file__).resolve().parent.parent / "results"
SIZES = (1_000, 100_000, 1_000_000)
DTYPES = {"float64": np.float64, "float32": np.float32, "float16": np.float16}
OPTIMIZERS = {
    "SGD": lambda p: SGD(p, lr=1e-3),
    "Momentum": lambda p: Momentum(p, lr=1e-3, momentum=0.9),
    "Adam": lambda p: Adam(p, lr=1e-3),
    "AdamW": lambda p: AdamW(p, lr=1e-3, weight_decay=1e-2),
}
STATE_ARRAYS_PER_PARAM = {"SGD": 0, "Momentum": 1, "Adam": 2, "AdamW": 2}
WARMUP = 20
STEPS = 200
GRAD_STD = 1e-2


def expected_state_bytes(optimizer: str, n_params: int, itemsize: int) -> int:
    return STATE_ARRAYS_PER_PARAM[optimizer] * n_params * itemsize


def state_bytes(opt) -> int:
    return sum(a.nbytes for state in opt.state for a in state.values())


def median_iqr(samples) -> tuple[float, float, float]:
    """Median, first and third quartile (linear interpolation, NumPy default)."""
    q1, med, q3 = np.percentile(np.asarray(samples, dtype=np.float64), [25, 50, 75])
    return float(med), float(q1), float(q3)


def make_problem(n: int, dtype, seed: int = 0):
    rng = np.random.default_rng(seed)
    params = [rng.standard_normal(n).astype(dtype)]
    grads = [(GRAD_STD * rng.standard_normal(n)).astype(dtype)]
    return params, grads


def time_steps(optimizer: str, n: int, dtype, warmup: int = WARMUP, steps: int = STEPS) -> tuple[list[int], bool]:
    """Durations of `steps` timed steps in ns, and whether the parameters stayed finite."""
    params, grads = make_problem(n, dtype)
    opt = OPTIMIZERS[optimizer](params)
    times = []
    # Division by zero is the known float16 Adam failure, reported in the
    # output rather than printed as a warning at every step.
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        for _ in range(warmup):
            opt.step(grads)
        gc.disable()  # a collection in the middle of a timed step is not the optimizer's cost
        try:
            for _ in range(steps):
                start = time.perf_counter_ns()
                opt.step(grads)
                times.append(time.perf_counter_ns() - start)
        finally:
            gc.enable()
    return times, all(bool(np.isfinite(p).all()) for p in params)


def memory_of_one_step(optimizer: str, n: int, dtype) -> tuple[int, int]:
    """(optimizer state bytes, peak bytes allocated during one step)."""
    params, grads = make_problem(n, dtype)
    opt = OPTIMIZERS[optimizer](params)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        opt.step(grads)  # the state exists after the first step
        tracemalloc.start()
        tracemalloc.reset_peak()
        opt.step(grads)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    return state_bytes(opt), peak


def _read(path: str) -> str | None:
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def windows_cpu_name() -> str | None:
    try:
        import winreg

        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
        return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
    except (ImportError, OSError):
        return None


def pi_throttled() -> str | None:
    """`vcgencmd get_throttled` on a Raspberry Pi: 0x0 means never throttled
    since boot. Any other value means the timings may be slowed by heat or by
    a weak power supply."""
    try:
        out = subprocess.run(["vcgencmd", "get_throttled"], capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip().removeprefix("throttled=") or None


def cpu_temperature_celsius() -> float | None:
    raw = _read("/sys/class/thermal/thermal_zone0/temp")  # millidegrees on Linux
    return int(raw) / 1000 if raw and raw.isdigit() else None


def machine_info() -> dict:
    model = _read("/proc/device-tree/model")  # "Raspberry Pi 5 Model B Rev 1.0" on a Pi
    cpu = None
    cpuinfo = _read("/proc/cpuinfo")
    if cpuinfo:
        for line in cpuinfo.splitlines():
            if line.lower().startswith(("model name", "hardware")):
                cpu = line.split(":", 1)[1].strip()
                break
    return {
        "date_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "board": model.replace("\x00", "") if model else None,
        "cpu": cpu or windows_cpu_name() or platform.processor() or None,
        "cpu_count": os.cpu_count(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "cpu_governor": _read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"),
        "cpu_max_khz": _read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq"),
        "load_average": os.getloadavg() if hasattr(os, "getloadavg") else None,
        "steps_timed": STEPS,
        "warmup_steps": WARMUP,
    }


def run(tag: str, sizes=SIZES, steps: int = STEPS, out_dir: Path = RESULTS) -> list[dict]:
    info = machine_info()
    info["temperature_start_c"] = cpu_temperature_celsius()
    rows = []
    for n in sizes:
        for dtype_name, dtype in DTYPES.items():
            for optimizer in OPTIMIZERS:
                times, finite = time_steps(optimizer, n, dtype, steps=steps)
                med, q1, q3 = median_iqr(times)
                state, peak = memory_of_one_step(optimizer, n, dtype)
                itemsize = np.dtype(dtype).itemsize
                rows.append(
                    {
                        "tag": tag,
                        "optimizer": optimizer,
                        "dtype": dtype_name,
                        "n_params": n,
                        "steps": len(times),
                        "params_finite": finite,
                        "step_us_median": f"{med / 1e3:.3f}",
                        "step_us_q1": f"{q1 / 1e3:.3f}",
                        "step_us_q3": f"{q3 / 1e3:.3f}",
                        "ns_per_param_median": f"{med / n:.4f}",
                        "param_bytes": n * itemsize,
                        "state_bytes": state,
                        "state_bytes_expected": expected_state_bytes(optimizer, n, itemsize),
                        "transient_peak_bytes": peak,
                    }
                )
                print(f"{tag} n={n:>9,} {dtype_name} {optimizer:9s} {med / 1e3:10.1f} us "
                      f"(IQR {q1 / 1e3:.1f}-{q3 / 1e3:.1f}), state {state:,} B, transient {peak:,} B"
                      + ("" if finite else "  [parameters not finite]"))
    info["temperature_end_c"] = cpu_temperature_celsius()
    info["throttled_end"] = pi_throttled()

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"bench_{tag}.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / f"bench_{tag}_machine.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--tag", required=True, help="name of the machine in the output files, e.g. pc or pi5")
    run(parser.parse_args().tag)


if __name__ == "__main__":
    main()

"""The benchmark helpers: memory accounting, statistics, output files."""

import csv
import json

import numpy as np
import pytest

from bench import step_bench


def test_expected_state_bytes_by_hand():
    # Adam keeps m and v: 2 arrays x 1000 values x 2 bytes (float16) = 4000
    got = step_bench.expected_state_bytes("Adam", 1000, 2)
    assert got == 4000, f"ECHEC: obtenu {got}, attendu 4000"
    got = step_bench.expected_state_bytes("SGD", 1000, 8)
    assert got == 0, f"ECHEC: obtenu {got}, attendu 0"


@pytest.mark.parametrize("dtype", list(step_bench.DTYPES))
@pytest.mark.parametrize("optimizer", list(step_bench.OPTIMIZERS))
def test_measured_state_matches_the_expected_size(optimizer, dtype):
    n = 500
    np_dtype = step_bench.DTYPES[dtype]
    state, _ = step_bench.memory_of_one_step(optimizer, n, np_dtype)
    expected = step_bench.expected_state_bytes(optimizer, n, np.dtype(np_dtype).itemsize)
    assert state == expected, f"ECHEC {optimizer} {dtype}: obtenu {state} octets, attendu {expected}"


def test_median_and_quartiles_by_hand():
    # NumPy's linear interpolation: Q1 at position 0.75 -> 10 + 0.75 * 10 = 17.5,
    # median 25, Q3 at position 2.25 -> 30 + 0.25 * 10 = 32.5
    got = step_bench.median_iqr([40, 10, 30, 20])
    assert got == (25.0, 17.5, 32.5), f"ECHEC: obtenu {got}, attendu (25.0, 17.5, 32.5)"


def test_adam_allocates_temporaries_during_a_step():
    # m * beta1, (1 - beta1) * g, g * g, sqrt(v)... each is a new array of n
    # values: the peak must be at least one array of the parameters' size.
    n = 10_000
    _, peak = step_bench.memory_of_one_step("Adam", n, np.float32)
    assert peak >= n * 4, f"ECHEC: obtenu un pic de {peak} octets, attendu >= {n * 4}"


def test_time_steps_returns_one_positive_duration_per_step():
    times, finite = step_bench.time_steps("Momentum", 100, np.float32, warmup=2, steps=7)
    assert len(times) == 7 and all(t > 0 for t in times), f"ECHEC: obtenu {times}"
    assert finite, "ECHEC: Momentum float32 a produit des valeurs non finies"


def test_native_float16_adam_is_flagged_as_not_finite():
    # H1 in native NumPy: eps = 1e-8 is 0 in float16, so some steps divide by 0.
    _, finite = step_bench.time_steps("Adam", 1000, np.float16, warmup=2, steps=5)
    assert not finite, "ECHEC: obtenu params_finite=True, attendu False (H1)"


def test_run_writes_both_files(tmp_path):
    rows = step_bench.run("unit", sizes=(50,), steps=3, out_dir=tmp_path)
    expected_rows = len(step_bench.OPTIMIZERS) * len(step_bench.DTYPES)
    assert len(rows) == expected_rows, f"ECHEC: obtenu {len(rows)} lignes, attendu {expected_rows}"
    with open(tmp_path / "bench_unit.csv", encoding="utf-8") as f:
        written = list(csv.DictReader(f))
    assert len(written) == expected_rows, f"ECHEC: obtenu {len(written)} lignes ecrites"
    info = json.loads((tmp_path / "bench_unit_machine.json").read_text(encoding="utf-8"))
    for key in ("numpy", "python", "machine", "cpu", "temperature_start_c", "throttled_end", "steps_timed"):
        assert key in info, f"ECHEC: cle {key!r} absente du fichier machine"

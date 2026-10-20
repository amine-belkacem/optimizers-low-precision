"""The four optimizers on digits, in every format and precision policy.

Same network, data split, batch size and epoch budget as the float64
learning-rate sweep (experiments/lr_sweep.py). Each optimizer uses the
learning rate chosen there in float64. Low precision may move the best rate,
but sweeping it again for 15 configurations would multiply the cost by 16.
This is a limit of the study and the README says so.

Configurations: fp64 (reference), fp32 P1/P3, and P1/P2/P3 for fp16,
bf16, Q15.16 and Q7.8. fp32 P2 is skipped because it is fp32 P1. In total
15 configurations x 4 optimizers x 10 seeds = 600 runs, spread over
the CPU cores. A seed fixes the initial weights, the batch order and the
stochastic rounding, so every run is reproducible on its own.

Writes:
  results/precision_matrix.csv   one row per run
  results/raw/precision.npz      per-epoch validation accuracy and counters
"""

from __future__ import annotations

import csv
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from experiments.digits import N_CLASSES, N_FEATURES, load_split
from experiments.lr_sweep import BATCH_SIZE, EPOCHS, HIDDEN
from lpopt.lowprec import FLOAT64_REFERENCE, STAT_KEYS, Policy, make
from lpopt.mlp import MLP
from lpopt.numerics import BF16, FP16, FP32, Q7_8, Q15_16
from lpopt.train import train_classifier

SEEDS = range(10)
OPTIMIZERS = ("SGD", "Momentum", "Adam", "AdamW")
HYPERPARAMETERS = {"Momentum": {"momentum": 0.9}, "AdamW": {"weight_decay": 1e-2}}
SR_SEED_OFFSET = 1000  # the rounding stream is separate from the data stream
RESULTS = Path(__file__).resolve().parent.parent / "results"


def configurations() -> list[Policy]:
    configs = [FLOAT64_REFERENCE, Policy.make("P1", FP32), Policy.make("P3", FP32)]
    for fmt in (FP16, BF16, Q15_16, Q7_8):
        configs += [Policy.make(name, fmt) for name in ("P1", "P2", "P3")]
    return configs


def chosen_learning_rates() -> dict[str, float]:
    with open(RESULTS / "lr_sweep_digits_best.csv", newline="", encoding="utf-8") as f:
        return {r["optimizer"]: float(r["chosen_lr"]) for r in csv.DictReader(f)}


def one_run(task: tuple) -> dict:
    optimizer, policy, lr, seed = task
    data = load_split()
    rng = np.random.default_rng(seed)
    model = MLP((N_FEATURES, HIDDEN, N_CLASSES))
    params = model.init_params(rng)
    rounding_rng = np.random.default_rng(seed + SR_SEED_OFFSET) if policy.rounding == "stochastic" else None
    opt = make(optimizer, params, policy, rounding_rng, lr=lr, **HYPERPARAMETERS.get(optimizer, {}))
    snapshots = []
    history = train_classifier(model, params, opt, data, EPOCHS, BATCH_SIZE, rng,
                               on_epoch=lambda _: snapshots.append([opt.stats[k] for k in STAT_KEYS]))
    finite = not history.diverged
    return {
        "optimizer": optimizer,
        "format": policy.param_format.name,
        "policy": policy.name,
        "seed": seed,
        "lr": lr,
        "diverged": history.diverged,
        "epochs_completed": len(history.val_accuracy) - 1,
        "final_val_accuracy": history.val_accuracy[-1] if finite else float("nan"),
        "final_val_loss": history.val_loss[-1] if finite else float("nan"),
        "test_accuracy": model.accuracy(params, data.X_test, data.y_test) if finite else float("nan"),
        **{k: opt.stats[k] for k in STAT_KEYS},
        "_val_accuracy": np.array(history.val_accuracy),
        "_stats": np.array(snapshots, dtype=np.int64).reshape(-1, len(STAT_KEYS)),
    }


def main() -> None:
    lrs = chosen_learning_rates()
    tasks = [(o, p, lrs[o], s) for p in configurations() for o in OPTIMIZERS for s in SEEDS]
    workers = max(1, (os.cpu_count() or 2) - 1)
    print(f"{len(tasks)} runs on {workers} processes")
    start = time.perf_counter()
    results = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, result in enumerate(pool.map(one_run, tasks, chunksize=4), 1):
            results.append(result)
            if i % 40 == 0:
                print(f"  {i}/{len(tasks)} ({time.perf_counter() - start:.0f} s)")

    raw = {}
    for r in results:
        key = f"{r['optimizer']}|{r['format']}|{r['policy']}|{r['seed']}"
        raw[f"{key}|val_accuracy"] = r.pop("_val_accuracy")
        raw[f"{key}|stats"] = r.pop("_stats")
    (RESULTS / "raw").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(RESULTS / "raw" / "precision.npz", stat_keys=np.array(STAT_KEYS), **raw)
    with open(RESULTS / "precision_matrix.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print(f"done in {time.perf_counter() - start:.0f} s")


if __name__ == "__main__":
    main()

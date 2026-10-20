"""The four optimizers on Rosenbrock, each at its best learning rate.

Comparing optimizers at one hand-picked learning rate mostly measures how
well that rate was picked. So every optimizer gets the same grid, and the
one with the lowest loss after N_STEPS is kept. Rosenbrock is
deterministic: there is no seed, a single run per point is the full answer.

Writes:
  results/rosenbrock_lr_grid.csv  final loss for every (optimizer, lr)
  results/raw/rosenbrock.npz      loss and trajectory at the best lr
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from lpopt.optimizers import SGD, Adam, AdamW, Momentum
from lpopt.tasks import Rosenbrock
from lpopt.train import run

N_STEPS = 5000
LEARNING_RATES = np.logspace(-5, 0, 11)  # half a decade apart
OPTIMIZERS = {
    "SGD": lambda params, lr: SGD(params, lr=lr),
    "Momentum": lambda params, lr: Momentum(params, lr=lr, momentum=0.9),
    "Adam": lambda params, lr: Adam(params, lr=lr),
    "AdamW": lambda params, lr: AdamW(params, lr=lr, weight_decay=1e-2),
}
RESULTS = Path(__file__).resolve().parent.parent / "results"


def main() -> None:
    task = Rosenbrock()
    rows = []
    best = {}
    for name, make in OPTIMIZERS.items():
        for lr in LEARNING_RATES:
            result = run(task, lambda params, lr=lr: make(params, lr), N_STEPS)
            distance = np.linalg.norm(result.trajectory[-1] - np.array(Rosenbrock.minimum))
            rows.append(
                {
                    "optimizer": name,
                    "lr": f"{lr:.3e}",
                    "diverged": result.diverged,
                    "final_loss": f"{result.final_loss:.6e}",
                    "distance_to_minimum": "" if result.diverged else f"{distance:.6e}",
                }
            )
            if not result.diverged and (name not in best or result.final_loss < best[name][1].final_loss):
                best[name] = (lr, result)
        lr, result = best[name]
        print(f"{name:9s} best lr {lr:.1e}: loss {result.final_loss:.3e}")

    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / "rosenbrock_lr_grid.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (RESULTS / "raw").mkdir(exist_ok=True)
    arrays = {}
    for name, (lr, result) in best.items():
        arrays[f"{name}|losses"] = result.losses
        arrays[f"{name}|trajectory"] = result.trajectory
        arrays[f"{name}|lr"] = np.array(lr)
    np.savez_compressed(RESULTS / "raw" / "rosenbrock.npz", **arrays)


if __name__ == "__main__":
    main()

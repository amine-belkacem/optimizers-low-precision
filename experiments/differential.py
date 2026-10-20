"""Error between lpopt and torch.optim over 1000 steps, 10 seeds.

Writes:
  results/raw/differential.npz     per-step errors, for the figure
  results/differential_summary.csv  final and worst error per optimizer and dtype
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from experiments.torch_reference import DEFAULT_SCENARIOS, compare_with_torch

N_STEPS = 1000
SEEDS = range(10)
DTYPES = {"float64": np.float64, "float32": np.float32}
RESULTS = Path(__file__).resolve().parent.parent / "results"


def main() -> None:
    raw = {}
    rows = []
    for scenario in DEFAULT_SCENARIOS:
        for dtype_name, dtype in DTYPES.items():
            errors = np.stack([compare_with_torch(scenario, N_STEPS, seed=s, dtype=dtype) for s in SEEDS])
            raw[f"{scenario.name}|{dtype_name}"] = errors
            final = errors[:, -1]
            rows.append(
                {
                    "optimizer": scenario.name,
                    "dtype": dtype_name,
                    "settings": scenario.kwargs,
                    "seeds": len(SEEDS),
                    "final_error_mean": f"{final.mean():.3e}",
                    "final_error_std": f"{final.std(ddof=1):.3e}",
                    "worst_error_any_step": f"{errors.max():.3e}",
                }
            )
            print(f"{scenario.name:9s} {dtype_name}: worst {errors.max():.3e}")

    (RESULTS / "raw").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(RESULTS / "raw" / "differential.npz", **raw)
    with open(RESULTS / "differential_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()

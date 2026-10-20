"""Aggregate results/precision_matrix.csv over the 10 seeds.

One row per (optimizer, format, policy): accuracy and loss (mean, std over
the runs that finished), number of failed runs, and each counter of
lpopt.lowprec as a fraction of its own denominator, summed over all steps
and seeds:

  stalled         stalled_params / grad_values
  absorbed        lost_updates / param_updates
  frozen_v        frozen_v / v_values
  frozen_buffer   frozen_buffer / buffer_values
  decay_lost      decay_lost / decay_values
  zero_denom      zero_denominator / denominators
  grad_underflow  grad_underflow / grad_values

Writes results/precision_summary.csv.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).resolve().parent.parent / "results"
RATIOS = {
    "stalled": ("stalled_params", "grad_values"),
    "absorbed": ("lost_updates", "param_updates"),
    "frozen_v": ("frozen_v", "v_values"),
    "frozen_buffer": ("frozen_buffer", "buffer_values"),
    "decay_lost": ("decay_lost", "decay_values"),
    "zero_denom": ("zero_denominator", "denominators"),
    "grad_underflow": ("grad_underflow", "grad_values"),
}


def summarize(rows: list[dict]) -> list[dict]:
    groups = list(dict.fromkeys((r["optimizer"], r["format"], r["policy"]) for r in rows))
    out = []
    for optimizer, fmt, policy in groups:
        rs = [r for r in rows if (r["optimizer"], r["format"], r["policy"]) == (optimizer, fmt, policy)]
        ok = [r for r in rs if r["diverged"] != "True"]
        row = {"optimizer": optimizer, "format": fmt, "policy": policy, "runs": len(rs), "failed": len(rs) - len(ok)}
        for metric in ("final_val_accuracy", "final_val_loss", "test_accuracy"):
            values = np.array([float(r[metric]) for r in ok])
            row[f"{metric}_mean"] = f"{values.mean():.4f}" if values.size else ""
            row[f"{metric}_std"] = f"{values.std(ddof=1):.4f}" if values.size > 1 else ""
        for name, (num, den) in RATIOS.items():
            d = sum(int(r[den]) for r in rs)
            row[name] = f"{sum(int(r[num]) for r in rs) / d:.4f}" if d else ""
        out.append(row)
    return out


def main() -> None:
    with open(RESULTS / "precision_matrix.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    summary = summarize(rows)
    with open(RESULTS / "precision_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)


if __name__ == "__main__":
    main()

"""Where plain SGD stops on a quadratic, in each format.

On f(x) = 1/2 * sum(eta_i * x_i^2) with lr = 1, each coordinate is an
independent 1-D problem whose step is round(eta_i * x_i). So one run with
60 coordinates covers 60 effective learning rates eta at once.

From x0 = 1: once eta * x <= q / 2, with q the spacing of the format near
x, the step rounds to 0 and x never moves again. In fixed point q is the
resolution everywhere. In floating point q is proportional to x in the
normal range (so SGD either stalls at the very first step or keeps going)
and becomes constant, the smallest subnormal, near zero. Hence the
predicted stall point

    x* = q / (2 eta)

with q = resolution (fixed point) or smallest subnormal (floating point).
With round-to-nearest the last move is one spacing, so x ends in
(x* - q, x*]. With stochastic rounding the expected step is exactly
eta * x and there is no such floor.

Writes results/quadratic_stall.csv.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from lpopt.lowprec import FLOAT64_REFERENCE, Policy, make
from lpopt.numerics import BF16, FP16, FP32, Q7_8, Q15_16, FloatFormat
from lpopt.tasks import Quadratic
from lpopt.train import run

N_STEPS = 20_000
ETAS = np.logspace(-4, -0.3, 60)
FORMATS = (FP32, FP16, BF16, Q15_16, Q7_8)
SR_SEEDS = range(3)
RESULTS = Path(__file__).resolve().parent.parent / "results"


def spacing_floor(fmt) -> float:
    return fmt.min_subnormal if isinstance(fmt, FloatFormat) else fmt.resolution


def final_x(policy, rng=None) -> np.ndarray:
    task = Quadratic(tuple(ETAS))
    result = run(task, lambda p: make("SGD", p, policy, rng, lr=1.0), N_STEPS)
    return result.trajectory[-1]


def main() -> None:
    rows = []
    exact = (1.0 - ETAS) ** N_STEPS
    runs = [("fp64", "P1", "", FLOAT64_REFERENCE, None, None)]
    for fmt in FORMATS:
        runs.append((fmt.name, "P1", "", Policy.make("P1", fmt), None, fmt))
        for seed in SR_SEEDS:
            runs.append((fmt.name, "P3", seed, Policy.make("P3", fmt), np.random.default_rng(seed), fmt))
    for name, policy_name, seed, policy, rng, fmt in runs:
        x = final_x(policy, rng)
        stall = spacing_floor(fmt) / (2 * ETAS) if fmt is not None else np.full_like(ETAS, np.nan)
        for eta, xi, si, ei in zip(ETAS, x, stall, exact):
            rows.append({"format": name, "policy": policy_name, "seed": seed, "eta": f"{eta:.6e}",
                         "final_x": f"{xi:.6e}", "predicted_stall": f"{si:.6e}", "exact_x": f"{ei:.6e}"})
        print(f"{name} {policy_name} {seed} done")
    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / "quadratic_stall.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()

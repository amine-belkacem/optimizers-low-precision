"""Figure: distance between lpopt and torch.optim, step by step."""

import matplotlib.pyplot as plt
import numpy as np

from figures.common import IMG, OPTIMIZER_ORDER, load_npz
from lpopt.plotting import BASELINE_GRAY, OPTIMIZER_COLORS, apply_style, save_figure

THRESHOLDS = {"float64": 1e-10, "float32": 1e-5}
UNIT_ROUNDOFF = {"float64": 2.0**-53, "float32": 2.0**-24}


def main() -> None:
    apply_style()
    data = load_npz("differential.npz")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharex=True)
    for ax, dtype in zip(axes, ("float64", "float32")):
        for name in OPTIMIZER_ORDER:
            errors = data[f"{name}|{dtype}"]  # (seeds, steps)
            worst = errors.max(axis=0)
            if np.any(worst <= 0.0):
                raise ValueError(f"{name} {dtype}: exact zero error cannot be drawn on a log axis")
            steps = np.arange(1, worst.size + 1)
            ax.plot(steps, worst, color=OPTIMIZER_COLORS[name.lower()], lw=1.2, label=name)
        ax.axhline(THRESHOLDS[dtype], color=BASELINE_GRAY, ls="--", lw=1)
        ax.text(1.2, THRESHOLDS[dtype] * 1.6, f"test threshold {THRESHOLDS[dtype]:.0e}", color=BASELINE_GRAY, fontsize=8)
        ax.axhline(UNIT_ROUNDOFF[dtype], color=BASELINE_GRAY, ls=":", lw=1)
        exponent = -53 if dtype == "float64" else -24
        ax.text(1.2, UNIT_ROUNDOFF[dtype] * 0.45, f"one rounding ($2^{{{exponent}}}$)", color=BASELINE_GRAY, fontsize=8)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(dtype)
        ax.set_xlabel("step")
    axes[0].set_ylabel("normwise relative error vs torch.optim\n(worst of 10 seeds)")
    axes[0].legend(loc="center right")
    save_figure(fig, IMG / "differential_error.png")


if __name__ == "__main__":
    main()

"""Figure: how far apart the representable numbers are, in each format.

Computed from the format definitions in src/lpopt/numerics.py, not from an
experiment: every point is fmt.ulp(x) at that x. The x grid is dense
(2000 points per decade) only so the sawtooth of the relative spacing is
drawn sharply. Each plotted value is exact.

The x range, 1e-10 to 1e6, is where the weights, gradients and Adam
moments of the networks in this repository live. fp32 and bf16 extend
far beyond it on both sides.
"""

import matplotlib.pyplot as plt
import numpy as np

from figures.common import IMG
from lpopt.numerics import BF16, FP16, FP32, Q7_8, Q15_16
from lpopt.plotting import BASELINE_GRAY, apply_style, color_for, save_figure

FORMATS = (FP32, FP16, BF16, Q15_16, Q7_8)
X_MIN, X_MAX = 1e-10, 1e6
ADAM_EPS = 1e-8
ONE_MINUS_BETA2 = 1e-3  # 1 - 0.999


def grid(fmt, per_decade=2000):
    x = np.logspace(np.log10(X_MIN), np.log10(X_MAX), int(np.log10(X_MAX / X_MIN) * per_decade))
    smallest = fmt.min_subnormal if hasattr(fmt, "min_subnormal") else fmt.resolution
    return x[(x >= smallest) & (x <= fmt.max_value)]


def main() -> None:
    apply_style()
    fig, (ax_abs, ax_rel) = plt.subplots(1, 2, figsize=(12, 4.6), sharex=True)
    for i, fmt in enumerate(FORMATS):
        x = grid(fmt)
        ulp = fmt.ulp(x)
        style = "-" if hasattr(fmt, "min_subnormal") else "--"
        label = f"{fmt.name} ({fmt.bits} bits)"
        ax_abs.plot(x, ulp, style, color=color_for(i), lw=1.4, label=label)
        # Half a spacing: an update smaller than this, relative to the value,
        # is rounded away by round-to-nearest.
        ax_rel.plot(x, 0.5 * ulp / x, style, color=color_for(i), lw=0.8, label=label)

    ax_abs.axvline(ADAM_EPS, color=BASELINE_GRAY, ls=":", lw=1)
    ax_abs.set_yscale("log")
    ax_abs.set_ylabel("gap to the next representable number")
    ax_abs.set_title(
        "Absolute spacing. fp16 stops at 6e-8 (smallest subnormal)\n"
        "and 65504, fixed point at its resolution. Dotted: Adam eps = 1e-8",
        fontsize=9,
    )
    ax_abs.legend(loc="upper left", fontsize=8)

    ax_rel.axhline(ONE_MINUS_BETA2, color=BASELINE_GRAY, ls=":", lw=1)
    ax_rel.set_yscale("log")
    ax_rel.set_ylim(1e-9, 1.0)
    ax_rel.set_ylabel("half spacing / |x|\n(relative updates below this are lost)")
    ax_rel.set_title(
        "Relative half spacing. Dotted: 1 - beta2 = 1e-3, the fraction\n"
        "Adam's v loses per step. bf16 sits above it, fp16 below",
        fontsize=9,
    )
    for ax in (ax_abs, ax_rel):
        ax.set_xscale("log")
        ax.set_xlabel("|x|")
        ax.set_xlim(X_MIN, X_MAX)
    save_figure(fig, IMG / "format_spacing.png")


if __name__ == "__main__":
    main()

"""Shared figure style.

Every figure script calls apply_style() first, so all plots in the README
share the same colors, fonts and grid. Blue and orange come last in the
series order on purpose: they are only used once the five other colors are
taken, which is the opposite of the matplotlib default.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
from cycler import cycler

TEAL = "#1baf7a"
VIOLET = "#6250d6"
MAGENTA = "#e87ba4"
YELLOW = "#eda100"
GREEN = "#008300"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
BASELINE_GRAY = "#8a8986"

SERIES_COLORS: tuple[str, ...] = (TEAL, VIOLET, MAGENTA, YELLOW, GREEN, BLUE, ORANGE)

# One fixed color per optimizer, so Adam is magenta in every figure of the
# repository and the reader never has to re-learn the legend.
OPTIMIZER_COLORS: dict[str, str] = {
    "sgd": TEAL,
    "momentum": VIOLET,
    "adam": MAGENTA,
    "adamw": YELLOW,
}

SAVE_DPI = 150


def color_for(index: int) -> str:
    """Color of the series at position `index` (0-based) in a figure."""
    if not 0 <= index < len(SERIES_COLORS):
        raise ValueError(
            f"series index {index} out of range: the palette has "
            f"{len(SERIES_COLORS)} colors, split the figure instead"
        )
    return SERIES_COLORS[index]


def apply_style() -> None:
    """Set the matplotlib defaults used by every figure script."""
    mpl.rcParams.update(
        {
            "axes.prop_cycle": cycler(color=list(SERIES_COLORS)),
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "font.size": 10,
            "legend.frameon": False,
            "figure.dpi": 100,
            "savefig.bbox": "tight",
        }
    )


def save_figure(fig: mpl.figure.Figure, path: str | Path) -> Path:
    """Save `fig` as PNG, creating the parent folder, then close it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=SAVE_DPI)
    mpl.pyplot.close(fig)
    return path

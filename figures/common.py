"""Paths and helpers shared by the figure scripts."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # scripts only write files, no window

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
RAW = RESULTS / "raw"
IMG = ROOT / "docs" / "img"

OPTIMIZER_ORDER = ("SGD", "Momentum", "Adam", "AdamW")


def load_npz(name: str):
    import numpy as np

    path = RAW / name
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing, run `python -m experiments.run_all` first")
    return np.load(path)


def load_csv(name: str) -> list[dict[str, str]]:
    path = RESULTS / name
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing, run `python -m experiments.run_all` first")
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

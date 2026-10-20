"""Figures: Rosenbrock paths, loss curves and final loss against learning rate."""

import matplotlib.pyplot as plt
import numpy as np

from figures.common import IMG, OPTIMIZER_ORDER, load_csv, load_npz
from lpopt.plotting import BASELINE_GRAY, OPTIMIZER_COLORS, apply_style, save_figure
from lpopt.tasks import Rosenbrock


def color(name: str) -> str:
    return OPTIMIZER_COLORS[name.lower()]


def paths(data) -> None:
    # One panel per optimizer: the paths overlap along the valley, on a
    # single axis they hide each other.
    task = Rosenbrock()
    xs, ys = np.meshgrid(np.linspace(-1.6, 1.6, 400), np.linspace(-0.6, 2.0, 400))
    z = (1 - xs) ** 2 + task.valley_stiffness * (ys - xs**2) ** 2
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.5), sharex=True, sharey=True)
    for ax, name in zip(axes.flat, OPTIMIZER_ORDER):
        ax.contour(xs, ys, np.log10(z + 1e-3), levels=20, colors=BASELINE_GRAY, linewidths=0.5, alpha=0.6)
        traj = data[f"{name}|trajectory"]
        lr = float(data[f"{name}|lr"])
        final_loss = data[f"{name}|losses"][-1]
        ax.plot(traj[:, 0], traj[:, 1], color=color(name), lw=1.3)
        ax.plot(*task.start, "o", color="black", ms=4)
        ax.plot(*Rosenbrock.minimum, "*", color="black", ms=9)
        ax.set_title(f"{name}, lr {lr:.1e}: loss {final_loss:.1e} after 5000 steps", fontsize=9)
        ax.grid(False)
    axes[0, 0].annotate("start", task.start, textcoords="offset points", xytext=(4, 6), fontsize=8)
    axes[0, 0].annotate("minimum", Rosenbrock.minimum, textcoords="offset points", xytext=(6, -12), fontsize=8)
    for ax in axes[1]:
        ax.set_xlabel("x")
    for ax in axes[:, 0]:
        ax.set_ylabel("y")
    save_figure(fig, IMG / "rosenbrock_paths.png")


def losses(data) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for name in OPTIMIZER_ORDER:
        loss = data[f"{name}|losses"]
        if np.any(loss <= 0.0):
            raise ValueError(f"{name}: exact zero loss cannot be drawn on a log axis")
        ax.plot(np.arange(loss.size), loss, color=color(name), lw=1.2, label=f"{name} (final {loss[-1]:.1e})")
    ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title("Rosenbrock loss at the best learning rate")
    ax.legend(loc="lower left")
    save_figure(fig, IMG / "rosenbrock_loss.png")


def lr_grid(rows) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.3))
    # Diverged runs have no loss: they are drawn as crosses on a separate
    # row per optimizer above the data, so two optimizers diverging at the
    # same rate do not hide each other.
    diverged_row = 1e3
    for name in OPTIMIZER_ORDER:
        own = [r for r in rows if r["optimizer"] == name]
        lr = np.array([float(r["lr"]) for r in own])
        diverged = np.array([r["diverged"] == "True" for r in own])
        final = np.array([float(r["final_loss"]) for r in own])
        ax.plot(lr[~diverged], final[~diverged], "o-", color=color(name), lw=1.2, ms=4, label=name)
        if diverged.any():
            ax.plot(lr[diverged], np.full(diverged.sum(), diverged_row), "x", color=color(name), ms=7)
            ax.text(lr[diverged][0] / 1.6, diverged_row, f"{name} diverged", color=color(name),
                    fontsize=8, va="center", ha="right")
            diverged_row *= 30
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("learning rate")
    ax.set_ylabel("loss after 5000 steps")
    ax.set_title("Rosenbrock: sensitivity to the learning rate")
    ax.legend(loc="lower left")
    save_figure(fig, IMG / "rosenbrock_lr_grid.png")


def main() -> None:
    apply_style()
    data = load_npz("rosenbrock.npz")
    paths(data)
    losses(data)
    lr_grid(load_csv("rosenbrock_lr_grid.csv"))


if __name__ == "__main__":
    main()

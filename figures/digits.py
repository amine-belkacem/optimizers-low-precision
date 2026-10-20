"""Figures of the learning-rate sweep on digits (10 seeds per point)."""

import matplotlib.pyplot as plt
import numpy as np

from figures.common import IMG, OPTIMIZER_ORDER, load_csv, load_npz
from lpopt.plotting import BASELINE_GRAY, OPTIMIZER_COLORS, apply_style, save_figure

CHANCE_ACCURACY = 0.1  # 10 balanced classes
MOMENTUM = 0.9  # value used in experiments/lr_sweep.py


def color(name: str) -> str:
    return OPTIMIZER_COLORS[name.lower()]


def rows_of(rows, name):
    own = [r for r in rows if r["optimizer"] == name]
    return own, np.array([float(r["lr"]) for r in own])


def sensitivity(rows, best) -> None:
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
    for name in OPTIMIZER_ORDER:
        own, lr = rows_of(rows, name)
        # Loss: geometric mean over seeds, band = x / divided by 10^std of log10.
        mu = np.array([float(r["log10_val_loss_mean"]) for r in own])
        sd = np.array([float(r["log10_val_loss_std"]) for r in own])
        ax_loss.plot(lr, 10**mu, "o-", color=color(name), ms=3.5, lw=1.2, label=name)
        ax_loss.fill_between(lr, 10 ** (mu - sd), 10 ** (mu + sd), color=color(name), alpha=0.2, lw=0)
        acc = np.array([float(r["val_accuracy_mean"]) for r in own])
        acc_sd = np.array([float(r["val_accuracy_std"]) for r in own])
        ax_acc.plot(lr, acc, "o-", color=color(name), ms=3.5, lw=1.2, label=name)
        ax_acc.fill_between(lr, acc - acc_sd, acc + acc_sd, color=color(name), alpha=0.2, lw=0)
        chosen = float(best[name]["chosen_lr"])
        i = int(np.argmin(np.abs(lr - chosen)))
        ax_acc.plot(lr[i], acc[i], "*", color=color(name), ms=12, mec="black", mew=0.6)
    ax_acc.axhline(CHANCE_ACCURACY, color=BASELINE_GRAY, ls="--", lw=1)
    ax_acc.text(1.2e-4, CHANCE_ACCURACY - 0.045, "chance (10 classes)", color=BASELINE_GRAY, fontsize=8)
    ax_acc.set_ylim(0.0, 1.03)
    ax_loss.set_yscale("log")
    ax_loss.set_ylabel("validation loss after 50 epochs\n(geometric mean, band = x/÷ 1 std)")
    ax_acc.set_ylabel("validation accuracy after 50 epochs\n(mean, band = ± 1 std)")
    ax_acc.text(1.2e-4, 0.66, "★ rate chosen on validation loss\n(Adam and AdamW: same rate)", fontsize=8)
    for ax in (ax_loss, ax_acc):
        ax.set_xscale("log")
        ax.set_xlabel("learning rate")
    ax_loss.legend(loc="lower left")
    fig.suptitle("MLP 64-32-10 on digits, 10 seeds per point", fontsize=11)
    save_figure(fig, IMG / "lr_sensitivity_digits.png")


def effective_rate(rows) -> None:
    """SGD and Momentum against lr / (1 - momentum)."""
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for name, scale in (("SGD", 1.0), ("Momentum", 1.0 / (1.0 - MOMENTUM))):
        own, lr = rows_of(rows, name)
        acc = np.array([float(r["val_accuracy_mean"]) for r in own])
        acc_sd = np.array([float(r["val_accuracy_std"]) for r in own])
        label = "SGD, lr" if name == "SGD" else "Momentum 0.9, lr / (1 - 0.9) = 10 lr"
        ax.errorbar(lr * scale, acc, yerr=acc_sd, fmt="o-", color=color(name), ms=4, lw=1.2,
                    capsize=2, label=label)
    ax.axhline(CHANCE_ACCURACY, color=BASELINE_GRAY, ls="--", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("effective learning rate")
    ax.set_ylabel("validation accuracy after 50 epochs")
    ax.set_title("Momentum 0.9 at lr behaves like SGD at 10 lr, until the top end")
    ax.legend(loc="upper left")
    save_figure(fig, IMG / "momentum_effective_lr.png")


def curves(data, best) -> None:
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
    for name in OPTIMIZER_ORDER:
        lr = best[name]["chosen_lr"]
        for ax, metric in ((ax_loss, "train_loss"), (ax_acc, "val_accuracy")):
            runs = data[f"{name}|{lr}|{metric}"]  # (seeds, epochs + 1)
            epochs = np.arange(runs.shape[1])
            mean, sd = runs.mean(axis=0), runs.std(axis=0, ddof=1)
            ax.plot(epochs, mean, color=color(name), lw=1.3, label=f"{name} (lr {float(lr):.2g})")
            ax.fill_between(epochs, mean - sd, mean + sd, color=color(name), alpha=0.2, lw=0)
    ax_loss.set_yscale("log")
    ax_loss.set_ylabel("training loss (mean ± 1 std, 10 seeds)")
    ax_acc.set_ylabel("validation accuracy (mean ± 1 std, 10 seeds)")
    # Zoom on the end of training. Epoch 0 (chance, about 0.11) and SGD at
    # epoch 1 (about 0.80) sit below the axis, which the plot says.
    ax_acc.set_ylim(0.85, 1.0)
    ax_acc.text(50, 0.855, "below the axis: epoch 0 (chance), SGD at epoch 1", ha="right", fontsize=8,
                color=BASELINE_GRAY)
    for ax in (ax_loss, ax_acc):
        ax.set_xlabel("epoch")
    ax_loss.legend(loc="lower left")
    save_figure(fig, IMG / "training_curves_digits.png")


def main() -> None:
    apply_style()
    rows = load_csv("lr_sweep_digits.csv")
    best = {r["optimizer"]: r for r in load_csv("lr_sweep_digits_best.csv")}
    sensitivity(rows, best)
    effective_rate(rows)
    curves(load_npz("lr_sweep.npz"), best)


if __name__ == "__main__":
    main()

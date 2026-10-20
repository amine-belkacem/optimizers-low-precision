"""Figures of the low-precision study (J4)."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from figures.common import IMG, OPTIMIZER_ORDER, load_csv, load_npz
from lpopt.plotting import BASELINE_GRAY, OPTIMIZER_COLORS, TEAL, VIOLET, apply_style, color_for, save_figure

CONFIG_ORDER = [("fp64", "P1"), ("fp32", "P1"), ("fp32", "P3")] + [
    (f, p) for f in ("fp16", "bf16", "Q15.16", "Q7.8") for p in ("P1", "P2", "P3")
]
POLICY_TEXT = {"P1": "all in format", "P2": "fp32 state", "P3": "stochastic"}
FAILED_COLOR = "#d9d9d6"


def config_label(fmt, policy):
    return "fp64 (reference)" if fmt == "fp64" else f"{fmt} {policy} ({POLICY_TEXT[policy]})"


def summary_index():
    return {(r["optimizer"], r["format"], r["policy"]): r for r in load_csv("precision_summary.csv")}


def accuracy_heatmap(summary) -> None:
    acc = np.full((len(CONFIG_ORDER), 4), np.nan)
    fig, ax = plt.subplots(figsize=(8, 7.5))
    cmap = LinearSegmentedColormap.from_list("teal", ["#ffffff", TEAL])
    for i, (fmt, pol) in enumerate(CONFIG_ORDER):
        for j, opt in enumerate(OPTIMIZER_ORDER):
            r = summary[(opt, fmt, pol)]
            if int(r["failed"]) == 0:
                acc[i, j] = float(r["final_val_accuracy_mean"])
    # Color scale from the worst finished cell to the best, so that a drop of
    # one point is visible. Failed cells are drawn separately, in light gray.
    ax.imshow(np.where(np.isnan(acc), np.nanmin(acc), acc), cmap=cmap, aspect="auto",
              vmin=np.nanmin(acc) - 0.005, vmax=np.nanmax(acc))
    for i, (fmt, pol) in enumerate(CONFIG_ORDER):
        for j, opt in enumerate(OPTIMIZER_ORDER):
            r = summary[(opt, fmt, pol)]
            if np.isnan(acc[i, j]):
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, color=FAILED_COLOR))
                ax.text(j, i, f"failed {r['failed']}/{r['runs']}", ha="center", va="center", fontsize=8,
                        color="#555555")
            else:
                text = f"{100 * acc[i, j]:.1f} ± {100 * float(r['final_val_accuracy_std']):.1f}"
                ax.text(j, i, text, ha="center", va="center", fontsize=8)
    ax.set_xticks(range(4), OPTIMIZER_ORDER)
    ax.set_yticks(range(len(CONFIG_ORDER)), [config_label(*c) for c in CONFIG_ORDER], fontsize=8)
    ax.xaxis.tick_top()
    ax.grid(False)
    for y in (2.5, 5.5, 8.5, 11.5):
        ax.axhline(y, color="white", lw=3)
    ax.set_title("Validation accuracy after 50 epochs (%, mean ± std over 10 seeds)", fontsize=10, pad=24)
    save_figure(fig, IMG / "precision_accuracy.png")


MECHANISMS = [
    ("SGD", "stalled", "SGD\nstalled params"),
    ("Momentum", "stalled", "Momentum\nstalled params"),
    ("Momentum", "frozen_buffer", "Momentum\nfrozen buffer"),
    ("Adam", "stalled", "Adam\nstalled params"),
    ("Adam", "frozen_v", "Adam\nfrozen v (H2)"),
    ("Adam", "zero_denom", "Adam\nzero denominator (H1)"),
    ("AdamW", "decay_lost", "AdamW\ndecay lost"),
    ("SGD", "grad_underflow", "gradient\nunderflow"),
]


def mechanisms_heatmap(summary) -> None:
    configs = CONFIG_ORDER[1:]  # fp64 is zero everywhere by construction
    values = np.full((len(configs), len(MECHANISMS)), np.nan)
    fig, ax = plt.subplots(figsize=(12.5, 7))
    cmap = LinearSegmentedColormap.from_list("violet", ["#ffffff", VIOLET])
    for i, (fmt, pol) in enumerate(configs):
        for j, (opt, key, _) in enumerate(MECHANISMS):
            r = summary[(opt, fmt, pol)]
            if r[key] != "":
                values[i, j] = float(r[key])
    ax.imshow(np.nan_to_num(values), cmap=cmap, aspect="auto", vmin=0, vmax=1)
    for i, (fmt, pol) in enumerate(configs):
        for j, (opt, key, _) in enumerate(MECHANISMS):
            r = summary[(opt, fmt, pol)]
            failed = int(r["failed"]) > 0
            if np.isnan(values[i, j]):
                text = "-"
            else:
                text = f"{100 * values[i, j]:.1f}%"
                if failed:
                    text += "*"
            ax.text(j, i, text, ha="center", va="center", fontsize=8,
                    color="white" if values[i, j] > 0.6 else "black")
    ax.set_xticks(range(len(MECHANISMS)), [m[2] for m in MECHANISMS], fontsize=8)
    ax.set_yticks(range(len(configs)), [config_label(*c) for c in configs], fontsize=8)
    ax.xaxis.tick_top()
    ax.grid(False)
    for y in (1.5, 4.5, 7.5, 10.5):
        ax.axhline(y, color="white", lw=3)
    ax.set_title("What the format does to the optimizer (fraction over all steps and 10 seeds)", fontsize=10, pad=40)
    fig.text(0.01, 0.01, "* run failed during the first epoch: the counters include steps after the parameters "
             "became non-finite, only the presence of zero denominators is meaningful there.",
             fontsize=7, color=BASELINE_GRAY)
    save_figure(fig, IMG / "precision_mechanisms.png")


def momentum_fixed_point(data) -> None:
    """Momentum in Q7.8: the three policies against float64, 10 seeds."""
    stat_keys = list(data["stat_keys"])
    stalled, grads = stat_keys.index("stalled_params"), stat_keys.index("grad_values")
    fig, (ax_acc, ax_st) = plt.subplots(1, 2, figsize=(11, 4))
    series = [("fp64", "P1", "fp64 reference"), ("Q7.8", "P1", "Q7.8 P1 (all in format)"),
              ("Q7.8", "P2", "Q7.8 P2 (fp32 state)"), ("Q7.8", "P3", "Q7.8 P3 (stochastic)")]
    for k, (fmt, pol, label) in enumerate(series):
        color = BASELINE_GRAY if fmt == "fp64" else color_for(k - 1)
        acc = np.stack([data[f"Momentum|{fmt}|{pol}|{s}|val_accuracy"] for s in range(10)])
        mean, sd = acc.mean(0), acc.std(0, ddof=1)
        epochs = np.arange(acc.shape[1])
        ax_acc.plot(epochs, mean, color=color, lw=1.3, label=label)
        ax_acc.fill_between(epochs, mean - sd, mean + sd, color=color, alpha=0.2, lw=0)
        if fmt != "fp64":
            st = np.stack([data[f"Momentum|{fmt}|{pol}|{s}|stats"] for s in range(10)]).sum(0)
            per_epoch = np.diff(np.vstack([np.zeros(st.shape[1]), st]), axis=0)
            frac = per_epoch[:, stalled] / per_epoch[:, grads]
            ax_st.plot(np.arange(1, len(frac) + 1), 100 * frac, color=color, lw=1.3, label=label)
    ax_acc.set_ylim(0.85, 1.0)
    ax_acc.set_xlabel("epoch (epoch 0, at chance level, is below the axis)")
    ax_acc.set_ylabel("validation accuracy (mean ± 1 std, 10 seeds)")
    ax_acc.legend(loc="lower right", fontsize=8)
    ax_st.set_xlabel("epoch")
    ax_st.set_ylabel("parameters that did not move although\ntheir gradient was not zero (%)")
    ax_st.set_ylim(0, 100)
    fig.suptitle("Momentum 0.9 in Q7.8 fixed point", fontsize=11)
    save_figure(fig, IMG / "precision_momentum_q78.png")


def quadratic(rows) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ref = [r for r in rows if r["format"] == "fp64"]
    eta = np.array([float(r["eta"]) for r in ref])
    ax.plot(eta, [float(r["exact_x"]) for r in ref], color=BASELINE_GRAY, lw=2, label="exact (1 - eta)^20000")
    zeros = []
    for i, fmt in enumerate(("fp32", "fp16", "bf16", "Q15.16", "Q7.8")):
        rs = [r for r in rows if r["format"] == fmt and r["policy"] == "P1"]
        x = np.array([float(r["final_x"]) for r in rs])
        stall = np.array([float(r["predicted_stall"]) for r in rs])
        # x = 0 is a legitimate end point (x* < q), but a log axis cannot show
        # it: those points are listed in a note instead of being dropped silently.
        zeros += [f"{fmt} at eta = {e:.3g}" for e in eta[x == 0]]
        ax.plot(eta[x > 0], x[x > 0], "o", color=color_for(i), ms=3.5, label=f"{fmt}, measured")
        shown = stall <= 1.0  # above 1 the prediction is "never leaves x0 = 1"
        ax.plot(eta[shown], stall[shown], "--", color=color_for(i), lw=1)
    ax.plot([], [], "--", color="black", lw=1, label="predicted stop q / (2 eta)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    # The exact curve keeps falling to 1e-298 at the largest rates. The axis
    # stops just below the smallest fp32 value so the formats stay readable.
    ax.set_ylim(1e-48, 1e3)
    ax.set_xlabel("effective learning rate eta")
    ax.set_ylabel("|x| after 20,000 steps of SGD from x0 = 1")
    ax.set_title("SGD on x^2 / 2: where each format stops it (round to nearest)", fontsize=10)
    ax.legend(loc="lower left", fontsize=7, ncol=2)
    notes = ["the exact curve continues below the axis"]
    if zeros:
        formats = sorted({z.split(" at ")[0] for z in zeros})
        etas = sorted({z.split(" = ")[1] for z in zeros})
        notes.append(f"reached exactly 0, not drawn: {', '.join(formats)} at eta = {', '.join(etas)}")
    fig.text(0.01, -0.02, "; ".join(notes), fontsize=7, color=BASELINE_GRAY)
    save_figure(fig, IMG / "quadratic_stall.png")


def main() -> None:
    apply_style()
    summary = summary_index()
    accuracy_heatmap(summary)
    mechanisms_heatmap(summary)
    momentum_fixed_point(load_npz("precision.npz"))
    quadratic(load_csv("quadratic_stall.csv"))


if __name__ == "__main__":
    main()

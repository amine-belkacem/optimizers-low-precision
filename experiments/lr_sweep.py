"""Learning-rate sensitivity of the four optimizers on digits.

MLP 64-32-10 (tanh), mini-batches of 64, 50 epochs, 16 learning rates from
1e-4 to 10 (a third of a decade apart), 10 seeds per point. A seed sets the
initial weights and the order of the mini-batches, the data split is fixed.

The grid goes up to 10 although the specification said 1: a first look
with one seed put SGD's best rate at 1, on the edge of the planned grid,
and a best value on the edge of a grid says nothing about what lies past it.

The learning rate of each optimizer is chosen on the mean validation loss
at the last epoch. The test set is evaluated for every run but only reported
at that chosen rate.

Writes:
  results/raw/lr_sweep.npz          per-epoch curves of every run
  results/lr_sweep_digits.csv       one row per (optimizer, lr): mean and std over seeds
  results/lr_sweep_digits_best.csv  chosen rate and test metrics per optimizer
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np

from experiments.digits import N_CLASSES, N_FEATURES, load_split
from lpopt.mlp import MLP
from lpopt.optimizers import SGD, Adam, AdamW, Momentum
from lpopt.train import train_classifier

HIDDEN = 32
EPOCHS = 50
BATCH_SIZE = 64
SEEDS = range(10)
LEARNING_RATES = np.logspace(-4, 1, 16)
OPTIMIZERS = {
    "SGD": lambda params, lr: SGD(params, lr=lr),
    "Momentum": lambda params, lr: Momentum(params, lr=lr, momentum=0.9),
    "Adam": lambda params, lr: Adam(params, lr=lr),
    "AdamW": lambda params, lr: AdamW(params, lr=lr, weight_decay=1e-2),
}
RESULTS = Path(__file__).resolve().parent.parent / "results"

# A learning rate is "usable" when its mean validation accuracy is within one
# percentage point of the best rate of the same optimizer. Fixed before
# looking at which optimizer it would favor.
USABLE_ACCURACY_GAP = 0.01


def usable_rates(rows: list[dict], optimizer: str) -> list[float]:
    own = [r for r in rows if r["optimizer"] == optimizer and r.get("val_accuracy_mean")]
    best = max(float(r["val_accuracy_mean"]) for r in own)
    return [float(r["lr"]) for r in own if float(r["val_accuracy_mean"]) >= best - USABLE_ACCURACY_GAP]


def one_run(data, make, lr: float, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    model = MLP((N_FEATURES, HIDDEN, N_CLASSES))
    params = model.init_params(rng)
    history = train_classifier(model, params, make(params, lr), data, EPOCHS, BATCH_SIZE, rng)
    if history.diverged:
        return {"diverged": True}
    return {
        "diverged": False,
        "train_loss": np.array(history.train_loss),
        "val_loss": np.array(history.val_loss),
        "val_accuracy": np.array(history.val_accuracy),
        "test_loss": model.loss(params, data.X_test, data.y_test),
        "test_accuracy": model.accuracy(params, data.X_test, data.y_test),
    }


def mean_std(values) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    return float(values.mean()), float(values.std(ddof=1))


def main() -> None:
    data = load_split()
    raw: dict[str, np.ndarray] = {}
    rows, best_rows = [], []
    start = time.perf_counter()
    for name, make in OPTIMIZERS.items():
        best = None
        for lr in LEARNING_RATES:
            runs = [one_run(data, make, lr, seed) for seed in SEEDS]
            ok = [r for r in runs if not r["diverged"]]
            row = {"optimizer": name, "lr": f"{lr:.4e}", "seeds": len(runs), "diverged": len(runs) - len(ok)}
            if ok:
                key = f"{name}|{lr:.4e}"
                for metric in ("train_loss", "val_loss", "val_accuracy"):
                    raw[f"{key}|{metric}"] = np.stack([r[metric] for r in ok])
                final_val_loss = [r["val_loss"][-1] for r in ok]
                # Losses span several decades across rates: the spread is also
                # reported on log10, which is what the figure draws.
                log_loss = np.log10(final_val_loss)
                row.update(
                    {
                        "val_loss_mean": "%.6e" % np.mean(final_val_loss),
                        "val_loss_std": "%.6e" % np.std(final_val_loss, ddof=1),
                        "log10_val_loss_mean": "%.6f" % log_loss.mean(),
                        "log10_val_loss_std": "%.6f" % log_loss.std(ddof=1),
                        "val_accuracy_mean": "%.6f" % mean_std([r["val_accuracy"][-1] for r in ok])[0],
                        "val_accuracy_std": "%.6f" % mean_std([r["val_accuracy"][-1] for r in ok])[1],
                    }
                )
                # Only a rate where all 10 seeds trained can be chosen.
                if len(ok) == len(runs) and (best is None or np.mean(final_val_loss) < best[1]):
                    best = (lr, float(np.mean(final_val_loss)), ok)
            rows.append(row)
        lr, val_loss, ok = best
        test_loss = mean_std([r["test_loss"] for r in ok])
        test_acc = mean_std([r["test_accuracy"] for r in ok])
        best_rows.append(
            {
                "optimizer": name,
                "chosen_lr": f"{lr:.4e}",
                "val_loss_mean": f"{val_loss:.4f}",
                "test_loss_mean": f"{test_loss[0]:.4f}",
                "test_loss_std": f"{test_loss[1]:.4f}",
                "test_accuracy_mean": f"{test_acc[0]:.4f}",
                "test_accuracy_std": f"{test_acc[1]:.4f}",
                "seeds": len(ok),
            }
        )
        usable = usable_rates(rows, name)
        # The grid is a third of a decade apart, so n consecutive usable
        # rates cover (n - 1) / 3 decades.
        best_rows[-1].update(
            {
                "usable_lr_min": f"{min(usable):.4e}",
                "usable_lr_max": f"{max(usable):.4e}",
                "usable_lr_count": len(usable),
                "usable_decades": f"{np.log10(max(usable) / min(usable)):.2f}",
            }
        )
        print(f"{name:9s} lr {lr:.2e}: test accuracy {test_acc[0]:.4f} +- {test_acc[1]:.4f}, "
              f"usable {min(usable):.1e}..{max(usable):.1e}  ({time.perf_counter() - start:.0f} s)")

    (RESULTS / "raw").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(RESULTS / "raw" / "lr_sweep.npz", **raw)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(RESULTS / "lr_sweep_digits.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, restval="")
        writer.writeheader()
        writer.writerows(rows)
    with open(RESULTS / "lr_sweep_digits_best.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(best_rows[0]))
        writer.writeheader()
        writer.writerows(best_rows)


if __name__ == "__main__":
    main()

"""Minimal training loop shared by the experiments."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from lpopt.numerics import NonRepresentableError
from lpopt.optimizers import Optimizer

# A loss above this is treated as a divergence. Rosenbrock starts at 24.2,
# so no run that is still converging ever gets near it.
DIVERGENCE_LOSS = 1e12


class Task(Protocol):
    def initial_params(self) -> list[np.ndarray]: ...
    def loss(self, params: list[np.ndarray]) -> float: ...
    def grads(self, params: list[np.ndarray]) -> list[np.ndarray]: ...


@dataclass
class RunResult:
    losses: np.ndarray  # loss before step 1, then after each step: shape (n_steps + 1,)
    trajectory: np.ndarray  # flattened parameters at the same instants
    diverged: bool

    @property
    def final_loss(self) -> float:
        return float("inf") if self.diverged else float(self.losses[-1])


def run(task: Task, make_optimizer: Callable[[list[np.ndarray]], Optimizer], n_steps: int) -> RunResult:
    """Train `task` for `n_steps`, stopping early if the loss blows up.

    On divergence the arrays are cut at the last finite point instead of
    being padded, so a plot never shows values that were not computed.
    """
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}")
    params = task.initial_params()
    opt = make_optimizer(params)
    losses = [task.loss(params)]
    trajectory = [np.concatenate([p.ravel() for p in params])]
    diverged = False
    # Overflow is the expected way for a too-large learning rate to fail,
    # it is detected below instead of printing a warning per step.
    with np.errstate(over="ignore", invalid="ignore"):
        for _ in range(n_steps):
            try:
                opt.step(task.grads(params))
            except NonRepresentableError:  # NaN in fixed point: the run failed
                diverged = True
                break
            loss = task.loss(params)
            if not np.isfinite(loss) or loss > DIVERGENCE_LOSS:
                diverged = True
                break
            losses.append(loss)
            trajectory.append(np.concatenate([p.ravel() for p in params]))
    return RunResult(np.array(losses), np.array(trajectory), diverged)


@dataclass(frozen=True)
class Split:
    """Train / validation / test arrays. Labels are integers in [0, n_classes)."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray


@dataclass
class ClassifierHistory:
    """Metrics at epoch 0 (before training) and after each epoch.

    On divergence the lists stop at the last epoch whose loss was finite.
    """

    train_loss: list[float]
    val_loss: list[float]
    val_accuracy: list[float]
    diverged: bool
    n_steps: int


def train_classifier(
    model,
    params: list[np.ndarray],
    optimizer: Optimizer,
    data: Split,
    epochs: int,
    batch_size: int,
    rng: np.random.Generator,
    on_epoch: Callable[[int], None] | None = None,
) -> ClassifierHistory:
    """Mini-batch training. `rng` only draws the order of the samples.

    The last incomplete batch of an epoch is kept, so every sample is seen
    once per epoch. `params` is updated in place. `on_epoch(epoch)` is
    called after each completed epoch (1-based), for instance to read the
    optimizer's counters.
    """
    if epochs < 1 or batch_size < 1:
        raise ValueError(f"epochs and batch_size must be >= 1, got {epochs} and {batch_size}")
    n = len(data.y_train)
    history = ClassifierHistory([], [], [], diverged=False, n_steps=0)

    def evaluate() -> bool:
        train_loss = model.loss(params, data.X_train, data.y_train)
        if not np.isfinite(train_loss) or train_loss > DIVERGENCE_LOSS:
            return False
        history.train_loss.append(train_loss)
        history.val_loss.append(model.loss(params, data.X_val, data.y_val))
        history.val_accuracy.append(model.accuracy(params, data.X_val, data.y_val))
        return True

    with np.errstate(over="ignore", invalid="ignore"):
        evaluate()
        for epoch in range(1, epochs + 1):
            order = rng.permutation(n)
            try:
                for start in range(0, n, batch_size):
                    batch = order[start : start + batch_size]
                    _, grads = model.loss_and_grads(params, data.X_train[batch], data.y_train[batch])
                    optimizer.step(grads)
                    history.n_steps += 1
            except NonRepresentableError:  # NaN in fixed point: the run failed
                history.diverged = True
                break
            if not evaluate():
                history.diverged = True
                break
            if on_epoch is not None:
                on_epoch(epoch)
    return history

"""Minimal training loop shared by the experiments."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np

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
            opt.step(task.grads(params))
            loss = task.loss(params)
            if not np.isfinite(loss) or loss > DIVERGENCE_LOSS:
                diverged = True
                break
            losses.append(loss)
            trajectory.append(np.concatenate([p.ravel() for p in params]))
    return RunResult(np.array(losses), np.array(trajectory), diverged)

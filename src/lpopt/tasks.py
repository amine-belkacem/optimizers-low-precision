"""Small optimization problems with a known answer.

Each task works on a list of parameter arrays, the same list the optimizers
update, and returns the loss and the gradients for it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Quadratic:
    """f(x) = 1/2 * sum(a_i * x_i^2), minimum 0 at x = 0.

    With plain SGD each coordinate follows x_k = (1 - lr * a_i)^k * x_0,
    which gives the tests an exact reference.
    """

    curvatures: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.curvatures or any(a <= 0.0 for a in self.curvatures):
            raise ValueError(f"curvatures must be > 0, got {self.curvatures}")

    def initial_params(self, x0: float = 1.0) -> list[np.ndarray]:
        return [np.full(len(self.curvatures), x0, dtype=np.float64)]

    def loss(self, params: list[np.ndarray]) -> float:
        (x,) = params
        return float(0.5 * np.sum(np.asarray(self.curvatures) * x * x))

    def grads(self, params: list[np.ndarray]) -> list[np.ndarray]:
        (x,) = params
        return [np.asarray(self.curvatures, dtype=x.dtype) * x]


@dataclass(frozen=True)
class Rosenbrock:
    """f(x, y) = (1 - x)^2 + 100 (y - x^2)^2, minimum 0 at (1, 1).

    The minimum sits at the bottom of a long curved valley: easy to find,
    slow to follow, which is what separates the optimizers.
    """

    start: tuple[float, float] = (-1.2, 1.0)  # classic start point (Rosenbrock, 1960)
    valley_stiffness: float = 100.0

    minimum = (1.0, 1.0)

    def initial_params(self) -> list[np.ndarray]:
        return [np.array(self.start, dtype=np.float64)]

    def loss(self, params: list[np.ndarray]) -> float:
        x, y = params[0]
        return float((1.0 - x) ** 2 + self.valley_stiffness * (y - x * x) ** 2)

    def grads(self, params: list[np.ndarray]) -> list[np.ndarray]:
        x, y = params[0]
        dx = -2.0 * (1.0 - x) - 4.0 * self.valley_stiffness * x * (y - x * x)
        dy = 2.0 * self.valley_stiffness * (y - x * x)
        return [np.array([dx, dy], dtype=params[0].dtype)]

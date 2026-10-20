"""SGD, Momentum, Adam and AdamW written from scratch in NumPy.

Each class follows the update rule documented for torch.optim (PyTorch 2.x),
including its less obvious choices, so the two can be compared step by step:

* SGD with momentum stores the raw gradient as the first buffer, without
  applying dampening on that first step.
* Adam's `weight_decay` is an L2 term added to the gradient, AdamW's is a
  multiplicative decay of the parameters applied before the Adam update.

The arithmetic happens in the dtype of the parameters: float64 parameters
give a float64 optimizer, float32 parameters a float32 one. Hyperparameters
stay Python floats, which NumPy 2 treats as "weak" scalars, so they never
promote a float32 array to float64.

Parameters are updated in place, like torch.optim does, so the arrays the
caller holds always see the latest values.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

import numpy as np


def _check_range(name: str, value: float, low: float, high: float | None = None) -> float:
    value = float(value)
    if not math.isfinite(value) or value < low or (high is not None and value >= high):
        bound = f"[{low}, {high})" if high is not None else f">= {low}"
        raise ValueError(f"{name} must be {bound}, got {value}")
    return value


class Optimizer:
    """Shared bookkeeping: parameter checks, step counter, per-parameter state."""

    def __init__(self, params: Iterable[np.ndarray], lr: float) -> None:
        self.params: list[np.ndarray] = list(params)
        if not self.params:
            raise ValueError("optimizer got an empty parameter list")
        for i, p in enumerate(self.params):
            if not isinstance(p, np.ndarray):
                raise TypeError(f"parameter {i} is a {type(p).__name__}, expected a numpy array")
            if not np.issubdtype(p.dtype, np.floating):
                raise TypeError(f"parameter {i} has dtype {p.dtype}, expected a floating dtype")
            if not p.flags.writeable:
                raise ValueError(f"parameter {i} is read-only, it cannot be updated in place")
        self.lr = _check_range("lr", lr, 0.0)
        self.state: list[dict[str, np.ndarray]] = [{} for _ in self.params]
        self.step_count = 0

    def step(self, grads: Sequence[np.ndarray]) -> None:
        """Apply one update, `grads[i]` being the gradient of `params[i]`."""
        grads = list(grads)
        if len(grads) != len(self.params):
            raise ValueError(f"got {len(grads)} gradients for {len(self.params)} parameters")
        for i, (p, g) in enumerate(zip(self.params, grads)):
            if np.shape(g) != p.shape:
                raise ValueError(f"gradient {i} has shape {np.shape(g)}, parameter has {p.shape}")
        self.step_count += 1
        for p, g, state in zip(self.params, grads, self.state):
            # asarray never copies a gradient already in the right dtype,
            # so _update must not modify g in place.
            self._update(p, np.asarray(g, dtype=p.dtype), state)

    def _update(self, p: np.ndarray, g: np.ndarray, state: dict[str, np.ndarray]) -> None:
        raise NotImplementedError


class SGD(Optimizer):
    """Stochastic gradient descent, optionally with (Nesterov) momentum.

        g <- g + weight_decay * p
        b <- g                                  first step
        b <- momentum * b + (1 - dampening) * g  next steps
        g <- g + momentum * b   if nesterov, else  g <- b
        p <- p - lr * g
    """

    def __init__(
        self,
        params: Iterable[np.ndarray],
        lr: float = 1e-3,
        momentum: float = 0.0,
        dampening: float = 0.0,
        nesterov: bool = False,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(params, lr)
        self.momentum = _check_range("momentum", momentum, 0.0)
        self.dampening = _check_range("dampening", dampening, 0.0)
        if self.dampening > 1.0:
            raise ValueError(f"dampening must be <= 1, got {self.dampening}")
        self.weight_decay = _check_range("weight_decay", weight_decay, 0.0)
        self.nesterov = bool(nesterov)
        if self.nesterov and (self.momentum == 0.0 or self.dampening != 0.0):
            raise ValueError("nesterov needs momentum > 0 and dampening = 0")

    def _update(self, p, g, state):
        if self.weight_decay != 0.0:
            g = g + self.weight_decay * p  # L2 term of SGD
        if self.momentum != 0.0:
            buf = state.get("momentum_buffer")
            if buf is None:
                buf = g.copy()
                state["momentum_buffer"] = buf
            else:
                buf *= self.momentum
                buf += (1.0 - self.dampening) * g
            g = g + self.momentum * buf if self.nesterov else buf
        p -= self.lr * g


class Momentum(SGD):
    """SGD with momentum (heavy ball), the same as torch.optim.SGD(momentum=0.9)."""

    def __init__(
        self,
        params: Iterable[np.ndarray],
        lr: float = 1e-3,
        momentum: float = 0.9,
        dampening: float = 0.0,
        nesterov: bool = False,
        weight_decay: float = 0.0,
    ) -> None:
        if not momentum > 0.0:
            raise ValueError(f"Momentum needs momentum > 0, got {momentum}, use SGD instead")
        super().__init__(params, lr, momentum, dampening, nesterov, weight_decay)


class Adam(Optimizer):
    """Adam (Kingma and Ba, 2015), with an optional L2 term.

        g <- g + weight_decay * p
        m <- beta1 * m + (1 - beta1) * g
        v <- beta2 * v + (1 - beta2) * g^2
        p <- p - lr * m_hat / (sqrt(v_hat) + eps)
             with m_hat = m / (1 - beta1^t) and v_hat = v / (1 - beta2^t)

    The last line is computed as PyTorch arranges it, which is the same
    value written differently:

        p <- p - (lr / (1 - beta1^t)) * m / (sqrt(v) / sqrt(1 - beta2^t) + eps)
    """

    decoupled_weight_decay = False

    def __init__(
        self,
        params: Iterable[np.ndarray],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(params, lr)
        if len(betas) != 2:
            raise ValueError(f"betas must hold two values, got {betas}")
        self.beta1 = _check_range("beta1", betas[0], 0.0, 1.0)
        self.beta2 = _check_range("beta2", betas[1], 0.0, 1.0)
        self.eps = _check_range("eps", eps, 0.0)
        self.weight_decay = _check_range("weight_decay", weight_decay, 0.0)

    def _update(self, p, g, state):
        if "exp_avg" not in state:
            state["exp_avg"] = np.zeros_like(p)
            state["exp_avg_sq"] = np.zeros_like(p)
        m = state["exp_avg"]
        v = state["exp_avg_sq"]
        beta1, beta2 = self.beta1, self.beta2

        if self.weight_decay != 0.0:
            if self.decoupled_weight_decay:
                p *= 1.0 - self.lr * self.weight_decay
            else:
                g = g + self.weight_decay * p

        m *= beta1
        m += (1.0 - beta1) * g
        v *= beta2
        v += (1.0 - beta2) * g * g

        t = self.step_count
        bias_correction1 = 1.0 - beta1**t
        bias_correction2 = 1.0 - beta2**t
        denom = np.sqrt(v) / math.sqrt(bias_correction2) + self.eps
        p -= (self.lr / bias_correction1) * (m / denom)


class AdamW(Adam):
    """Adam with decoupled weight decay (Loshchilov and Hutter, 2019).

        p <- p * (1 - lr * weight_decay)
        then the Adam update, without any L2 term in the gradient.
    """

    decoupled_weight_decay = True

    def __init__(
        self,
        params: Iterable[np.ndarray],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
    ) -> None:
        super().__init__(params, lr, betas, eps, weight_decay)

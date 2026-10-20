"""The four optimizers running in an emulated number format.

Each class inherits from its float64 version in lpopt.optimizers (same
hyperparameters, same checks) and replaces the update with one where every
elementary operation is rounded to a format, in exactly the order the
float64 code evaluates it. Constants (lr, 1 - beta1, the bias corrections)
are computed in float64 and rounded to the format once, which is what NumPy
does with a Python float applied to a float16 array.

Because the order is the same, running this code in FP16 gives, bit for
bit, the float64 code applied to np.float16 arrays (tested). The emulation
adds what NumPy cannot do: bf16, fixed point, stochastic rounding, and a
different format for the parameters and for the optimizer state.

A precision policy says which format holds what:

  P1  parameters, gradients, optimizer state and arithmetic all in `fmt`,
      round to nearest.
  P2  parameters and gradients in `fmt`, optimizer state and arithmetic in
      fp32. Only the new parameter value is rounded to `fmt`.
  P3  like P1, with stochastic rounding everywhere.

The model itself is not emulated: it computes in float64 from parameters
that hold `fmt` values, and its gradients are rounded to `fmt` when they
reach the optimizer. The study is about the optimizer, so that is the part
whose every operation is rounded.

The optimizer also counts what goes wrong, for the analysis:

  stalled_params   the parameter did not change during the step although
                   its float64 gradient was not zero. This one covers every
                   cause and compares across formats.
  lost_updates     the update, computed in the format, is not zero, but
                   p - update rounds back to p (absorption). In fixed point
                   a small update usually rounds to 0 before that, so it
                   shows up in stalled_params rather than here.
  frozen_v         beta2 * v rounds back to v (v != 0): the decay of
                   Adam's second moment is lost
  frozen_buffer    momentum * buf rounds back to buf (buf != 0): the same
                   loss of decay, on the momentum buffer
  decay_lost       AdamW: p * (1 - lr * weight_decay) rounds back to p (p != 0)
  zero_denominator sqrt(v_hat) + eps is exactly 0
  grad_underflow   a non-zero float64 gradient rounds to 0
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from lpopt import optimizers
from lpopt.numerics import FP32, FP64

POLICY_NAMES = ("P1", "P2", "P3")


@dataclass(frozen=True)
class Policy:
    name: str
    param_format: object
    state_format: object
    rounding: str

    @classmethod
    def make(cls, name: str, fmt) -> Policy:
        if name == "P1":
            return cls(name, fmt, fmt, "nearest")
        if name == "P2":
            return cls(name, fmt, FP32, "nearest")
        if name == "P3":
            return cls(name, fmt, fmt, "stochastic")
        raise ValueError(f"policy must be one of {POLICY_NAMES}, got {name!r}")

    @property
    def label(self) -> str:
        return f"{self.param_format.name} {self.name}"


FLOAT64_REFERENCE = Policy("P1", FP64, FP64, "nearest")

STAT_KEYS = (
    "stalled_params",
    "param_updates",
    "lost_updates",
    "decay_values",
    "decay_lost",
    "v_values",
    "frozen_v",
    "buffer_values",
    "frozen_buffer",
    "denominators",
    "zero_denominator",
    "grad_values",
    "grad_underflow",
)


class _Emulated:
    """Shared part: policy, rounding helpers, gradient rounding, counters."""

    def __init__(self, params: Iterable[np.ndarray], policy: Policy, rng: np.random.Generator | None = None,
                 **hyperparameters) -> None:
        params = list(params)
        for i, p in enumerate(params):
            if not isinstance(p, np.ndarray) or p.dtype != np.float64:
                raise TypeError(f"parameter {i}: emulated optimizers store values in float64 arrays")
        super().__init__(params, **hyperparameters)
        if policy.rounding == "stochastic" and rng is None:
            raise ValueError("a stochastic-rounding policy needs rng=np.random.Generator")
        self.policy = policy
        self.rng = rng
        self.stats = dict.fromkeys(STAT_KEYS, 0)
        for p in self.params:  # start from values the format can hold
            p[...] = policy.param_format.round(p)

    # Rounding helpers. s(): optimizer arithmetic and state. const(): a
    # constant, rounded once to nearest. store(): a new parameter value.
    def s(self, x):
        return self.policy.state_format.round(x, self.policy.rounding, self.rng)

    def const(self, x: float):
        return self.policy.state_format.round(x)

    def store(self, x):
        return self.policy.param_format.round(x, self.policy.rounding, self.rng)

    def step(self, grads: Sequence[np.ndarray]) -> None:
        raw = [np.asarray(g, dtype=np.float64) for g in grads]
        rounded = []
        for g in raw:
            r = self.policy.param_format.round(g, self.policy.rounding, self.rng)
            self.stats["grad_values"] += int(np.count_nonzero(g))
            self.stats["grad_underflow"] += int(np.count_nonzero((r == 0) & (g != 0)))
            rounded.append(r)
        before = [p.copy() for p in self.params]
        super().step(rounded)
        for p, old, g in zip(self.params, before, raw):
            self.stats["stalled_params"] += int(np.count_nonzero((p == old) & (g != 0)))

    def _apply(self, p: np.ndarray, update: np.ndarray) -> None:
        """p <- store(p - update), counting the updates the format swallows."""
        new = self.store(self.s(p - update))
        lost = (new == p) & (update != 0)
        self.stats["param_updates"] += int(np.count_nonzero(update))
        self.stats["lost_updates"] += int(np.count_nonzero(lost))
        p[...] = new


class LowPrecSGD(_Emulated, optimizers.SGD):
    def _update(self, p, g, state):
        s, c = self.s, self.const
        if self.weight_decay != 0.0:
            g = s(g + s(c(self.weight_decay) * p))
        if self.momentum != 0.0:
            buf = state.get("momentum_buffer")
            if buf is None:
                buf = s(g)  # g is already a value of the format: no change in P1/P3
            else:
                decayed = s(buf * c(self.momentum))
                self.stats["buffer_values"] += int(np.count_nonzero(buf))
                self.stats["frozen_buffer"] += int(np.count_nonzero((decayed == buf) & (buf != 0)))
                buf = s(decayed + s(c(1.0 - self.dampening) * g))
            state["momentum_buffer"] = buf
            g = s(g + s(c(self.momentum) * buf)) if self.nesterov else buf
        self._apply(p, s(c(self.lr) * g))


class LowPrecMomentum(_Emulated, optimizers.Momentum):
    _update = LowPrecSGD._update


class LowPrecAdam(_Emulated, optimizers.Adam):
    def _update(self, p, g, state):
        s, c = self.s, self.const
        if "exp_avg" not in state:
            state["exp_avg"] = np.zeros_like(p)
            state["exp_avg_sq"] = np.zeros_like(p)
        m, v = state["exp_avg"], state["exp_avg_sq"]

        if self.weight_decay != 0.0:
            if self.decoupled_weight_decay:
                decayed = self.store(s(p * c(1.0 - self.lr * self.weight_decay)))
                self.stats["decay_values"] += int(np.count_nonzero(p))
                self.stats["decay_lost"] += int(np.count_nonzero((decayed == p) & (p != 0)))
                p[...] = decayed
            else:
                g = s(g + s(c(self.weight_decay) * p))

        m = s(s(m * c(self.beta1)) + s(c(1.0 - self.beta1) * g))
        decayed = s(v * c(self.beta2))
        self.stats["v_values"] += int(np.count_nonzero(v))
        self.stats["frozen_v"] += int(np.count_nonzero((decayed == v) & (v != 0)))
        v = s(decayed + s(s(c(1.0 - self.beta2) * g) * g))
        state["exp_avg"], state["exp_avg_sq"] = m, v

        t = self.step_count
        bias_correction1 = 1.0 - self.beta1**t
        bias_correction2 = 1.0 - self.beta2**t
        denom = s(s(s(np.sqrt(v)) / c(math.sqrt(bias_correction2))) + c(self.eps))
        self.stats["denominators"] += denom.size
        self.stats["zero_denominator"] += int(np.count_nonzero(denom == 0))
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = s(m / denom)  # m / 0 is the failure being measured
        self._apply(p, s(c(self.lr / bias_correction1) * ratio))


class LowPrecAdamW(_Emulated, optimizers.AdamW):
    _update = LowPrecAdam._update


CLASSES = {
    "SGD": LowPrecSGD,
    "Momentum": LowPrecMomentum,
    "Adam": LowPrecAdam,
    "AdamW": LowPrecAdamW,
}


def make(name: str, params: Iterable[np.ndarray], policy: Policy, rng: np.random.Generator | None = None,
         **hyperparameters):
    if name not in CLASSES:
        raise ValueError(f"optimizer must be one of {list(CLASSES)}, got {name!r}")
    return CLASSES[name](params, policy, rng, **hyperparameters)

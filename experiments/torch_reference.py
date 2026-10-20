"""Run an lpopt optimizer and its torch.optim counterpart on the same gradients.

The gradients come from a seeded generator and do not depend on the
parameters (open loop). Both optimizers therefore see exactly the same
inputs at every step, and any difference in the parameters comes from the
optimizers alone, not from a feedback loop that would amplify it.

The error at a step is the normwise relative error, over all parameters:

    max_i |p_lpopt_i - p_torch_i| / max_i |p_torch_i|

An element-wise relative error would explode for a parameter that happens
to pass close to zero, without the optimizer being any less correct.

Why the error is not exactly zero: on an AVX2 CPU, torch computes
`p.add_(g, alpha=-lr)` with a fused multiply-add (one rounding) on full
vector blocks, 8 elements in float64 and 16 in float32, and with a multiply
then an add (two roundings) on the remaining tail. lpopt always rounds
after each operation. The (64,) parameter below is there so that both torch
code paths are exercised in both dtypes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import ModuleType

import numpy as np
import torch

import lpopt.optimizers

PARAM_SHAPES: tuple[tuple[int, ...], ...] = ((5, 3), (7,), (64,))

TORCH_CLASSES = {
    "SGD": torch.optim.SGD,
    "Momentum": torch.optim.SGD,
    "Adam": torch.optim.Adam,
    "AdamW": torch.optim.AdamW,
}

TORCH_DTYPES = {np.float64: torch.float64, np.float32: torch.float32}


@dataclass(frozen=True)
class Scenario:
    name: str
    optimizer: str
    kwargs: dict = field(default_factory=dict)

    def torch_kwargs(self) -> dict:
        kwargs = dict(self.kwargs)
        if self.optimizer == "Momentum":
            kwargs.setdefault("momentum", 0.9)
        # The single-tensor loop: it is the code path the lpopt classes
        # mirror, and pinning it keeps the reference stable across versions.
        kwargs["foreach"] = False
        return kwargs


# Default settings of each optimizer, used for the figure.
DEFAULT_SCENARIOS: tuple[Scenario, ...] = (
    Scenario("SGD", "SGD", {"lr": 0.05}),
    Scenario("Momentum", "Momentum", {"lr": 0.05, "momentum": 0.9}),
    Scenario("Adam", "Adam", {"lr": 1e-2}),
    Scenario("AdamW", "AdamW", {"lr": 1e-2}),
)

# Every option the lpopt classes implement, at least once. Dampening > 0 is
# needed to catch a wrong first momentum step: with dampening = 0 the two
# textbook variants give the same buffer.
ALL_SCENARIOS: tuple[Scenario, ...] = DEFAULT_SCENARIOS + (
    Scenario("SGD + L2", "SGD", {"lr": 0.05, "weight_decay": 0.1}),
    Scenario("Momentum + dampening", "Momentum", {"lr": 0.05, "momentum": 0.9, "dampening": 0.5}),
    Scenario(
        "Nesterov + L2", "Momentum", {"lr": 0.05, "momentum": 0.8, "nesterov": True, "weight_decay": 0.01}
    ),
    Scenario("Adam + L2", "Adam", {"lr": 1e-2, "betas": (0.8, 0.99), "eps": 1e-6, "weight_decay": 0.1}),
    Scenario("AdamW strong decay", "AdamW", {"lr": 1e-2, "betas": (0.85, 0.995), "weight_decay": 0.1}),
)


def gradient_sequence(n_steps: int, seed: int, dtype=np.float64) -> list[list[np.ndarray]]:
    rng = np.random.default_rng(seed)
    return [[rng.standard_normal(shape).astype(dtype) for shape in PARAM_SHAPES] for _ in range(n_steps)]


def initial_params(seed: int, dtype=np.float64) -> list[np.ndarray]:
    # Offset the seed so the parameters are not the first gradient.
    rng = np.random.default_rng(seed + 10_000)
    return [rng.standard_normal(shape).astype(dtype) for shape in PARAM_SHAPES]


def normwise_relative_error(ours: list[np.ndarray], reference: list[np.ndarray]) -> float:
    diff = max(float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))) for a, b in zip(ours, reference))
    scale = max(float(np.max(np.abs(b.astype(np.float64)))) for b in reference)
    return diff / scale


def compare_with_torch(
    scenario: Scenario,
    n_steps: int,
    seed: int = 0,
    dtype=np.float64,
    module: ModuleType = lpopt.optimizers,
) -> np.ndarray:
    """Error after each step, shape (n_steps,).

    `module` is the module the lpopt class is taken from. The mutation tests
    pass a mutated copy of lpopt.optimizers here.
    """
    torch_dtype = TORCH_DTYPES[dtype]
    params = initial_params(seed, dtype)
    torch_params = [torch.tensor(p.copy(), dtype=torch_dtype) for p in params]

    ours = getattr(module, scenario.optimizer)(params, **scenario.kwargs)
    reference = TORCH_CLASSES[scenario.optimizer](torch_params, **scenario.torch_kwargs())

    errors = np.empty(n_steps)
    for k, grads in enumerate(gradient_sequence(n_steps, seed, dtype)):
        ours.step(grads)
        for tp, g in zip(torch_params, grads):
            tp.grad = torch.tensor(g, dtype=torch_dtype)
        reference.step()
        errors[k] = normwise_relative_error(params, [tp.numpy() for tp in torch_params])
    return errors

"""T3: differential tests against torch.optim, same gradients for both.

Thresholds (success criteria 1 and 2 of the specification):
* float64: normwise relative error <= 1e-10 after 1000 steps;
* float32: <= 1e-5 after 1000 steps.
One rounding costs about 1.1e-16 in float64 and 6e-8 in float32, and
torch does not round at the same places (see experiments/torch_reference.py),
so the error grows slowly with the number of steps. The thresholds leave
several orders of magnitude for that growth and still catch any real
mistake in an update rule, which shows up at 1e-3 or above.
"""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from experiments.torch_reference import ALL_SCENARIOS, Scenario, compare_with_torch

N_STEPS = 1000
THRESHOLDS = {np.float64: 1e-10, np.float32: 1e-5}


@pytest.mark.parametrize("dtype", [np.float64, np.float32], ids=["fp64", "fp32"])
@pytest.mark.parametrize("scenario", ALL_SCENARIOS, ids=[s.name for s in ALL_SCENARIOS])
def test_matches_torch_over_1000_steps(scenario, dtype):
    errors = compare_with_torch(scenario, N_STEPS, seed=1, dtype=dtype)
    worst = float(errors.max())
    assert worst <= THRESHOLDS[dtype], f"ECHEC {scenario.name}: obtenu {worst:.3e}, attendu <= {THRESHOLDS[dtype]:.0e}"


@st.composite
def scenarios(draw):
    name = draw(st.sampled_from(["SGD", "Momentum", "Adam", "AdamW"]))
    lr = draw(st.floats(1e-4, 1.0))
    weight_decay = draw(st.sampled_from([0.0, draw(st.floats(1e-4, 0.5))]))
    kwargs = {"lr": lr, "weight_decay": weight_decay}
    if name == "Momentum":
        kwargs["momentum"] = draw(st.floats(0.05, 0.99))
        if draw(st.booleans()):
            kwargs["nesterov"] = True
        else:
            kwargs["dampening"] = draw(st.floats(0.0, 1.0))
    elif name in ("Adam", "AdamW"):
        kwargs["betas"] = (draw(st.floats(0.0, 0.99)), draw(st.floats(0.5, 0.9999)))
        kwargs["eps"] = draw(st.floats(1e-12, 1e-3))
    return Scenario(name, name, kwargs)


@settings(max_examples=60, deadline=None)
@given(scenario=scenarios(), seed=st.integers(0, 2**31 - 1))
def test_matches_torch_for_random_settings(scenario, seed):
    errors = compare_with_torch(scenario, n_steps=30, seed=seed)
    worst = float(errors.max())
    assert worst <= THRESHOLDS[np.float64], f"ECHEC {scenario.kwargs}: obtenu {worst:.3e}, attendu <= 1e-10"

"""The stall point of SGD on a quadratic, predicted by hand.

Step k: x <- x - round(eta * x). The step is 0 as soon as eta * x <= q / 2,
q being the spacing of the format around x. With a constant spacing q
(fixed point, or the subnormal range of a float), x stops just under
x* = q / (2 eta), within one spacing: x_final in (x* - q, x*].
"""

import numpy as np
import pytest

from lpopt.lowprec import Policy, make
from lpopt.numerics import BF16, FP16, Q7_8
from lpopt.tasks import Quadratic
from lpopt.train import run


def final_x(fmt, eta, n_steps=500, policy="P1", rng=None):
    result = run(Quadratic((eta,)), lambda p: make("SGD", p, Policy.make(policy, fmt), rng, lr=1.0), n_steps)
    return float(result.trajectory[-1, 0])


@pytest.mark.parametrize(
    "fmt, eta, q",
    [
        (Q7_8, 0.05, 2.0**-8),  # x* = 2^-8 / 0.1 = 0.0390625
        (FP16, 0.25, 2.0**-24),  # x* = 2^-24 / 0.5 = 1.19e-7, in the subnormal range
    ],
    ids=["Q7.8", "fp16 subnormals"],
)
def test_sgd_stops_just_under_the_predicted_point(fmt, eta, q):
    x_star = q / (2 * eta)
    got = final_x(fmt, eta)
    assert x_star - q < got <= x_star, f"ECHEC: obtenu x = {got!r}, attendu dans ]{x_star - q!r}, {x_star!r}]"


@pytest.mark.parametrize(
    "fmt, eta",
    [(FP16, 1e-4), (BF16, 1e-3)],
    ids=["fp16, eta below 2^-12", "bf16, eta below 2^-9"],
)
def test_sgd_never_leaves_the_start_when_eta_is_below_half_the_relative_spacing(fmt, eta):
    # Just under 1 the spacing is 2^-(p-1) / 2: 2^-11 in fp16, 2^-8 in bf16.
    # Half of it, 2.44e-4 and 1.95e-3, is above eta, so the first step is lost.
    got = final_x(fmt, eta, n_steps=100)
    assert got == 1.0, f"ECHEC: obtenu {got!r}, attendu 1.0"


def test_stochastic_rounding_goes_past_the_fixed_point_floor():
    # Same setting as the Q7.8 case above: nearest stops at 0.039, stochastic
    # rounding keeps an expected step of eta * x and gets much closer to 0.
    got = abs(final_x(Q7_8, 0.05, policy="P3", rng=np.random.default_rng(0)))
    assert got < 0.0390625 / 4, f"ECHEC: obtenu |x| = {got!r}, attendu < 0.0098"

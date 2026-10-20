"""T2: one or two steps on a scalar, compared with values computed by hand.

The expected values were derived by hand, then checked with exact rational
arithmetic (fractions.Fraction) and 40-digit decimals. They are written as
fractions or decimals so the derivation stays readable. The tolerance only
absorbs float64 rounding: a dozen operations, each off by at most 2^-53.
"""

import numpy as np
import pytest

from lpopt.optimizers import SGD, Adam, AdamW, Momentum

FLOAT64_REL_TOL = 1e-14


def run(optimizer_class, grads, p0=1.0, **kwargs):
    p = np.array([p0])
    opt = optimizer_class([p], **kwargs)
    history = []
    for g in grads:
        opt.step([np.array([g])])
        history.append(float(p[0]))
    return history


def check(got, expected):
    assert got == pytest.approx(expected, rel=FLOAT64_REL_TOL), f"ECHEC: obtenu {got!r}, attendu {expected!r}"


def test_sgd_with_l2():
    # g' = 1 + 0.5 * 1 = 1.5 ; p1 = 1 - 0.1 * 1.5 = 0.85
    check(run(SGD, [1.0], lr=0.1, weight_decay=0.5), [17 / 20])


def test_momentum_two_steps():
    # b1 = g1 = 1           ; p1 = 1 - 0.1 * 1     = 0.9
    # b2 = 0.9 * 1 + 2 = 2.9 ; p2 = 0.9 - 0.1 * 2.9 = 0.61
    check(run(Momentum, [1.0, 2.0], lr=0.1, momentum=0.9), [0.9, 0.61])


def test_nesterov_two_steps():
    # b1 = 1   ; g' = 1 + 0.9 * 1   = 1.9  ; p1 = 1 - 0.19     = 0.81
    # b2 = 2.9 ; g' = 2 + 0.9 * 2.9 = 4.61 ; p2 = 0.81 - 0.461 = 0.349
    check(run(Momentum, [1.0, 2.0], lr=0.1, momentum=0.9, nesterov=True), [0.81, 0.349])


def test_first_momentum_step_ignores_dampening():
    # PyTorch convention: b1 = g1 (dampening not applied), so p1 = 0.9.
    # The textbook b1 = (1 - 0.5) * g1 would give p1 = 0.95.
    # b2 = 0.9 * 1 + 0.5 * 2 = 1.9 ; p2 = 0.9 - 0.19 = 0.71
    check(run(Momentum, [1.0, 2.0], lr=0.1, momentum=0.9, dampening=0.5), [0.9, 0.71])


def test_adam_first_step_is_close_to_lr_times_sign():
    # m1 = 0.1 * 0.5 = 0.05 ; v1 = 0.001 * 0.25 = 0.00025
    # m_hat = 0.05 / 0.1 = 0.5 ; v_hat = 0.00025 / 0.001 = 0.25
    # p1 = 1 - 0.1 * 0.5 / (0.5 + 1e-8) = 1 - 0.1 / (1 + 2e-8) = 0.90000000199999996...
    check(run(Adam, [0.5], lr=0.1), [0.9000000019999999600000008])


def test_adam_two_steps():
    # m2 = 0.9 * 0.05 + 0.1 * (-0.25) = 0.02
    # v2 = 0.999 * 0.00025 + 0.001 * 0.0625 = 0.00031225
    # m_hat = 0.02 / (1 - 0.9^2) = 0.02 / 0.19 = 0.10526315789473684...
    # v_hat = 0.00031225 / (1 - 0.999^2) = 0.00031225 / 0.001999 = 0.15620310155077539...
    # p2 = p1 - 0.1 * m_hat / (sqrt(v_hat) + 1e-8) = 0.87336629870784616...
    check(run(Adam, [0.5, -0.25], lr=0.1), [0.9000000019999999600000008, 0.8733662987078461625593764])


def test_adamw_decays_the_parameter_before_the_update():
    # p <- 1 * (1 - 0.1 * 0.5) = 0.95, then the Adam step of the test above:
    # p1 = 0.95 - 0.099999998000000040 = 0.85000000199999996...
    check(run(AdamW, [0.5], lr=0.1, weight_decay=0.5), [0.8500000019999999600000008])


def test_adam_l2_is_normalized_away_on_the_first_step():
    # g' = 0.5 + 0.5 * 1 = 1.0 ; m_hat = 1, v_hat = 1
    # p1 = 1 - 0.1 * 1 / (1 + 1e-8) = 0.90000000099999999...
    # The L2 term only changed the update by 1e-9: Adam divides it away.
    # This is the reason AdamW exists.
    check(run(Adam, [0.5], lr=0.1, weight_decay=0.5), [0.9000000009999999900000001])


@pytest.mark.parametrize("optimizer_class", [SGD, Momentum, Adam, AdamW])
def test_zero_learning_rate_leaves_the_parameters_unchanged(optimizer_class):
    rng = np.random.default_rng(0)
    p = rng.standard_normal(10)
    expected = p.copy()
    opt = optimizer_class([p], lr=0.0)
    for _ in range(5):
        opt.step([rng.standard_normal(10)])
    assert np.array_equal(p, expected), f"ECHEC: obtenu {p}, attendu {expected}"


@pytest.mark.parametrize("optimizer_class", [SGD, Momentum, Adam, AdamW])
def test_float32_parameters_stay_float32(optimizer_class):
    p = np.ones(4, dtype=np.float32)
    opt = optimizer_class([p], lr=0.1)
    for _ in range(3):
        opt.step([np.full(4, 0.3, dtype=np.float32)])
    dtypes = {p.dtype, *(a.dtype for a in opt.state[0].values())}
    assert dtypes == {np.dtype(np.float32)}, f"ECHEC: obtenu {dtypes}, attendu {{float32}}"


@pytest.mark.parametrize("optimizer_class", [SGD, Momentum, Adam, AdamW])
def test_step_does_not_modify_the_gradient(optimizer_class):
    p = np.ones(3)
    g = np.array([0.1, -0.2, 0.3])
    expected = g.copy()
    opt = optimizer_class([p], lr=0.1, weight_decay=0.1)
    opt.step([g])
    opt.step([g])
    assert np.array_equal(g, expected), f"ECHEC: obtenu {g}, attendu {expected}"


@pytest.mark.parametrize(
    "make",
    [
        lambda p: SGD([], lr=0.1),
        lambda p: SGD([[1.0, 2.0]], lr=0.1),
        lambda p: SGD([np.arange(3)], lr=0.1),
        lambda p: SGD([p], lr=-0.1),
        lambda p: SGD([p], lr=float("nan")),
        lambda p: SGD([p], lr=0.1, dampening=1.5),
        lambda p: SGD([p], lr=0.1, nesterov=True),
        lambda p: SGD([p], lr=0.1, momentum=0.9, dampening=0.1, nesterov=True),
        lambda p: Momentum([p], lr=0.1, momentum=0.0),
        lambda p: Adam([p], betas=(0.9, 1.0)),
        lambda p: Adam([p], betas=(0.9,)),
        lambda p: Adam([p], eps=-1e-8),
        lambda p: AdamW([p], weight_decay=-0.01),
    ],
)
def test_invalid_settings_are_rejected(make):
    with pytest.raises((ValueError, TypeError)):
        make(np.ones(3))


def test_read_only_parameter_is_rejected():
    p = np.ones(3)
    p.flags.writeable = False
    with pytest.raises(ValueError):
        SGD([p], lr=0.1)


@pytest.mark.parametrize(
    "grads", [[np.ones(3), np.ones(3)], [np.ones(4)]], ids=["too many gradients", "wrong shape"]
)
def test_mismatched_gradients_are_rejected(grads):
    opt = Adam([np.ones(3)])
    with pytest.raises(ValueError):
        opt.step(grads)
    assert opt.step_count == 0, f"ECHEC: obtenu step_count={opt.step_count}, attendu 0"

"""T6: the tasks themselves, and SGD against its closed-form trajectory."""

import numpy as np
import pytest

from lpopt.optimizers import SGD
from lpopt.tasks import Quadratic, Rosenbrock


def test_sgd_halves_exactly_on_a_unit_quadratic():
    # a = 1, lr = 0.5: x_{k+1} = x_k - 0.5 * x_k = x_k / 2, and dividing a
    # float by 2 is exact, so x_k = 2^-k bit for bit, without any tolerance.
    task = Quadratic((1.0,))
    params = task.initial_params(1.0)
    opt = SGD(params, lr=0.5)
    for k in range(1, 60):
        opt.step(task.grads(params))
        got, expected = float(params[0][0]), 2.0**-k
        assert got == expected, f"ECHEC k={k}: obtenu {got!r}, attendu {expected!r}"


def test_sgd_follows_the_closed_form_on_a_quadratic():
    curvatures = (0.5, 2.0, 7.0)
    lr = 0.1
    task = Quadratic(curvatures)
    params = task.initial_params(3.0)
    opt = SGD(params, lr=lr)
    n_steps = 200
    for _ in range(n_steps):
        opt.step(task.grads(params))
    expected = 3.0 * (1.0 - lr * np.array(curvatures)) ** n_steps
    got = params[0]
    # About 2 roundings per step on each coordinate: 400 * 1.1e-16 < 1e-13.
    assert np.allclose(got, expected, rtol=1e-13, atol=0.0), f"ECHEC: obtenu {got}, attendu {expected}"


def test_sgd_diverges_above_the_stability_limit():
    # |1 - lr * a| > 1 when lr > 2 / a: here 1 - 0.3 * 10 = -2.
    task = Quadratic((10.0,))
    params = task.initial_params(1.0)
    opt = SGD(params, lr=0.3)
    for _ in range(10):
        opt.step(task.grads(params))
    got, expected = float(params[0][0]), (-2.0) ** 10
    assert got == expected, f"ECHEC: obtenu {got!r}, attendu {expected!r}"


def test_quadratic_rejects_non_positive_curvature():
    with pytest.raises(ValueError):
        Quadratic((1.0, 0.0))


def test_rosenbrock_is_zero_with_zero_gradient_at_its_minimum():
    task = Rosenbrock()
    params = [np.array(Rosenbrock.minimum)]
    loss, grad = task.loss(params), task.grads(params)[0]
    assert loss == 0.0, f"ECHEC: obtenu {loss!r}, attendu 0.0"
    assert np.array_equal(grad, [0.0, 0.0]), f"ECHEC: obtenu {grad}, attendu [0, 0]"


def test_rosenbrock_value_at_the_classic_start():
    # (1 - (-1.2))^2 + 100 * (1 - 1.44)^2 = 4.84 + 19.36 = 24.2
    got = Rosenbrock().loss(Rosenbrock().initial_params())
    assert got == pytest.approx(24.2, rel=1e-14), f"ECHEC: obtenu {got!r}, attendu 24.2"


@pytest.mark.parametrize("point", [(-1.2, 1.0), (0.3, -0.7), (2.0, 3.5)])
def test_rosenbrock_gradient_matches_central_differences(point):
    task = Rosenbrock()
    x = np.array(point)
    h = 1e-6
    numeric = np.empty(2)
    for i in range(2):
        step = np.zeros(2)
        step[i] = h
        numeric[i] = (task.loss([x + step]) - task.loss([x - step])) / (2 * h)
    analytic = task.grads([x])[0]
    # Central differences are accurate to O(h^2) ~ 1e-12 times the third
    # derivative, plus about 1e-16 * |f| / h of cancellation: 1e-6 relative is safe.
    assert np.allclose(analytic, numeric, rtol=1e-6, atol=1e-6), f"ECHEC: obtenu {analytic}, attendu {numeric}"

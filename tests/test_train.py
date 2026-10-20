import numpy as np
import pytest

from lpopt.optimizers import SGD
from lpopt.tasks import Quadratic
from lpopt.train import run


class UnitQuadratic(Quadratic):
    def initial_params(self):
        return super().initial_params(1.0)


def test_run_records_the_loss_before_and_after_each_step():
    # a = 1, lr = 0.5: x_k = 2^-k, loss_k = 0.5 * 4^-k
    result = run(UnitQuadratic((1.0,)), lambda p: SGD(p, lr=0.5), n_steps=3)
    expected = [0.5, 0.125, 0.03125, 0.0078125]
    assert result.losses.tolist() == expected, f"ECHEC: obtenu {result.losses.tolist()}, attendu {expected}"
    assert result.trajectory.shape == (4, 1), f"ECHEC: obtenu {result.trajectory.shape}, attendu (4, 1)"
    assert result.final_loss == 0.0078125, f"ECHEC: obtenu {result.final_loss}, attendu 0.0078125"


def test_run_stops_on_divergence_and_keeps_only_computed_points():
    # 1 - lr * a = 1 - 30 = -29: |x| grows by 29 per step, loss by 841.
    # loss_k = 0.5 * 841^k exceeds 1e12 at k = 5 (2.1e14), so 4 steps survive.
    result = run(UnitQuadratic((1.0,)), lambda p: SGD(p, lr=30.0), n_steps=100)
    assert result.diverged, "ECHEC: obtenu diverged=False, attendu True"
    assert len(result.losses) == 5, f"ECHEC: obtenu {len(result.losses)} points, attendu 5"
    assert result.final_loss == float("inf"), f"ECHEC: obtenu {result.final_loss}, attendu inf"
    assert np.all(np.isfinite(result.losses)), f"ECHEC: obtenu {result.losses}, attendu que des valeurs finies"


def test_run_treats_a_fixed_point_nan_as_a_failure():
    # Q7.8 Adam at the optimum: g = 0, m = v = 0, eps rounds to 0, 0 / 0 = NaN.
    from lpopt.lowprec import Policy, make
    from lpopt.numerics import Q7_8

    class AtOptimum(Quadratic):
        def initial_params(self):
            return super().initial_params(0.0)

    result = run(AtOptimum((1.0,)), lambda p: make("Adam", p, Policy.make("P1", Q7_8)), n_steps=10)
    assert result.diverged, "ECHEC: obtenu diverged=False, attendu True"
    assert len(result.losses) == 1, f"ECHEC: obtenu {len(result.losses)} points, attendu 1"


def test_run_rejects_zero_steps():
    with pytest.raises(ValueError):
        run(UnitQuadratic((1.0,)), lambda p: SGD(p, lr=0.1), n_steps=0)

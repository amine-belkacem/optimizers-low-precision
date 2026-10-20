"""The emulated optimizers: equivalence with the verified ones, then H1 and H2."""

import numpy as np
import pytest

from experiments.torch_reference import ALL_SCENARIOS, gradient_sequence, initial_params
from lpopt import lowprec, optimizers
from lpopt.lowprec import FLOAT64_REFERENCE, Policy
from lpopt.numerics import BF16, FP16, FP32, Q7_8, NonRepresentableError

N_STEPS = 200


def run_pair(scenario, policy, native_dtype, n_steps=N_STEPS, seed=4):
    """The emulated optimizer and the J1 one on native arrays, same gradients."""
    grads = gradient_sequence(n_steps, seed)  # float64
    emu_params = initial_params(seed)
    native_params = [p.astype(native_dtype) for p in initial_params(seed)]
    emu = lowprec.make(scenario.optimizer, emu_params, policy, **scenario.kwargs)
    native = getattr(optimizers, scenario.optimizer)(native_params, **scenario.kwargs)
    with np.errstate(all="ignore"):
        for g in grads:
            emu.step(g)
            native.step([x.astype(native_dtype) for x in g])
    return emu_params, native_params


def bit_identical(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return np.array_equal(a, b, equal_nan=True) and np.array_equal(np.signbit(a), np.signbit(b))


@pytest.mark.parametrize(
    "policy, native_dtype",
    [(FLOAT64_REFERENCE, np.float64), (Policy.make("P1", FP32), np.float32), (Policy.make("P1", FP16), np.float16)],
    ids=["fp64", "fp32", "fp16"],
)
@pytest.mark.parametrize("scenario", ALL_SCENARIOS, ids=[s.name for s in ALL_SCENARIOS])
def test_emulation_matches_native_arrays_bit_for_bit(scenario, policy, native_dtype):
    # fp64: the emulation with rounding switched off is the J1 code itself.
    # fp32, fp16: rounding every operation equals NumPy's own float32 and
    # float16 arithmetic (shown in test_arithmetic.py), as long as the order
    # of operations is the same. 200 steps, 9 settings, 3 parameter arrays.
    ours, theirs = run_pair(scenario, policy, native_dtype)
    for k, (a, b) in enumerate(zip(ours, theirs)):
        diff = int(np.sum(~((a == b) | (np.isnan(a) & np.isnan(b)))))
        assert bit_identical(a, b), f"ECHEC {scenario.name} param {k}: {diff} valeurs differentes, attendu 0"


def one_adam_step(policy, g, p0=1.0, lr=1e-3):
    p = np.array([p0])
    opt = lowprec.make("Adam", [p], policy, lr=lr)
    opt.step([np.array([g])])
    return float(p[0]), opt


def test_h1_fp16_adam_divides_by_zero():
    # Hand derivation (see the J4 notes in the README):
    # v = (fp16(0.001) * 1e-4 -> 1.19e-7) * 1e-4 -> 1.2e-11 -> 0 in fp16
    # m = fp16(0.1) * 1e-4 = 1.0e-5, not 0
    # denom = sqrt(0) / c + fp16(1e-8) = 0 + 0 = 0, so m / denom = +inf
    p, opt = one_adam_step(Policy.make("P1", FP16), g=1e-4)
    assert p == -np.inf, f"ECHEC: obtenu p = {p!r}, attendu -inf"
    assert opt.stats["zero_denominator"] == 1, f"ECHEC: obtenu {opt.stats['zero_denominator']}, attendu 1"
    v = float(opt.state[0]["exp_avg_sq"][0])
    assert v == 0.0, f"ECHEC: obtenu v = {v!r}, attendu 0.0"


def test_h1_fp32_state_keeps_eps_and_v():
    # P2: v = 1e-11 and eps = 1e-8 exist in fp32. The first Adam step is
    # lr * m_hat / (sqrt(v_hat) + eps) ~ lr * 1e-4 / (1e-4 + 1e-8) ~ lr.
    p, opt = one_adam_step(Policy.make("P2", FP16), g=1e-4)
    expected = float(FP16.round(1.0 - 1e-3 / (1 + 1e-4)))
    assert p == expected, f"ECHEC: obtenu {p!r}, attendu {expected!r}"
    assert opt.stats["zero_denominator"] == 0, "ECHEC: denominateur nul en P2"


def decay_v(policy, n_steps=100):
    """One step with g = 1, then n_steps with g = 0: only the decay acts on v."""
    p = np.array([1.0])
    opt = lowprec.make("Adam", [p], policy, lr=1e-3)
    opt.step([np.array([1.0])])
    v_after_first = float(opt.state[0]["exp_avg_sq"][0])
    for _ in range(n_steps):
        opt.step([np.array([0.0])])
    return v_after_first, float(opt.state[0]["exp_avg_sq"][0]), opt


def test_h2_bf16_v_never_decays():
    # After the first step v = round(0.001 * 1 * 1) != 0. With zero
    # gradients, v <- round(0.999 * v) each step, and in bf16 that is v.
    v0, v100, opt = decay_v(Policy.make("P1", BF16))
    assert v100 == v0, f"ECHEC: obtenu v = {v100!r}, attendu {v0!r} (aucune decroissance)"
    assert opt.stats["frozen_v"] == 100, f"ECHEC: obtenu {opt.stats['frozen_v']}, attendu 100"


def test_h2_fp32_state_decays_v():
    # P2 keeps v in fp32: after 100 steps v = 0.001 * 0.999^100 = 9.05e-4,
    # to fp32 rounding (about 100 roundings of 6e-8 relative each).
    v0, v100, opt = decay_v(Policy.make("P2", BF16))
    expected = v0 * 0.999**100
    assert v100 == pytest.approx(expected, rel=1e-5), f"ECHEC: obtenu {v100!r}, attendu {expected!r}"
    assert opt.stats["frozen_v"] == 0, f"ECHEC: obtenu {opt.stats['frozen_v']}, attendu 0"


def test_small_sgd_updates_are_lost_with_nearest_and_kept_with_stochastic():
    # lr * g = 1e-4, half the fp16 spacing at 1 is 2^-11 = 4.9e-4: every step
    # of round-to-nearest is swallowed. Stochastic rounding moves p by 1e-4
    # per step on average: after 1000 steps p ~ 1 - 0.1 = 0.9.
    def run(policy, rng=None):
        p = np.array([1.0])
        opt = lowprec.make("SGD", [p], policy, rng, lr=1e-4)
        for _ in range(1000):
            opt.step([np.array([1.0])])
        return float(p[0]), opt

    p_nearest, opt = run(Policy.make("P1", FP16))
    assert p_nearest == 1.0, f"ECHEC: obtenu {p_nearest!r}, attendu 1.0"
    assert opt.stats["lost_updates"] == 1000, f"ECHEC: obtenu {opt.stats['lost_updates']}, attendu 1000"
    assert opt.stats["stalled_params"] == 1000, f"ECHEC: obtenu {opt.stats['stalled_params']}, attendu 1000"
    p_sr, _ = run(Policy.make("P3", FP16), np.random.default_rng(0))
    # The spacing below 1 is 2^-11 = 4.9e-4; each step goes down one spacing
    # with probability ~0.2. Std of the final value ~ sqrt(1000 * 0.2 * 0.8) * 4.9e-4 = 6e-3.
    assert abs(p_sr - 0.9) < 0.03, f"ECHEC: obtenu {p_sr!r}, attendu 0.9 +- 0.03"


def test_fixed_point_small_update_counts_as_stalled_not_lost():
    # Q7.8: lr * g = 1e-4 rounds to 0 (resolution 2^-8 = 3.9e-3) before the
    # subtraction, so the update itself is 0: stalled, but not "absorbed".
    p = np.array([1.0])
    opt = lowprec.make("SGD", [p], Policy.make("P1", Q7_8), lr=1e-4)
    opt.step([np.array([1.0])])
    got = (opt.stats["stalled_params"], opt.stats["lost_updates"])
    assert got == (1, 0), f"ECHEC: obtenu {got}, attendu (1, 0)"


LR_DIGITS, WD = 2.1544e-02, 1e-2  # Adam's rate from the digits sweep, AdamW's default decay


@pytest.mark.parametrize("policy_name", ["P1", "P2"])
def test_adamw_decay_vanishes_for_every_normal_fp16(policy_name):
    # 1 - lr * wd = 0.99978. Half the relative fp16 spacing is between 2^-12
    # = 2.44e-4 and 2^-11, always above 2.15e-4: p * 0.99978 rounds back to p
    # for every normal p, whether the constant is fp16 (P1) or fp32 (P2).
    bits = np.arange(2**16, dtype=np.uint16).view(np.float16).astype(np.float64)
    p = bits[np.isfinite(bits) & (np.abs(bits) >= FP16.min_normal)]
    opt = lowprec.make("AdamW", [p.copy()], Policy.make(policy_name, FP16), lr=LR_DIGITS, weight_decay=WD)
    decayed = opt.store(opt.s(p * opt.const(1.0 - LR_DIGITS * WD)))
    changed = int(np.sum(decayed != p))
    assert changed == 0, f"ECHEC: {changed} valeurs decroissent sur {p.size}, attendu 0"


def test_adamw_is_adam_in_fp16():
    # Consequence of the test above: with the same gradients, AdamW and Adam
    # give bit-identical parameters, and every decay is counted as lost.
    grads = gradient_sequence(50, seed=5)
    ps_adam, ps_adamw = initial_params(5), initial_params(5)
    adam = lowprec.make("Adam", ps_adam, Policy.make("P1", FP16), lr=LR_DIGITS)
    adamw = lowprec.make("AdamW", ps_adamw, Policy.make("P1", FP16), lr=LR_DIGITS, weight_decay=WD)
    for g in grads:
        adam.step(g)
        adamw.step(g)
    for a, b in zip(ps_adam, ps_adamw):
        assert np.array_equal(a, b), "ECHEC: Adam et AdamW different en fp16"
    got = (adamw.stats["decay_lost"], adamw.stats["decay_values"])
    assert got[0] == got[1] > 0, f"ECHEC: obtenu {got}, attendu decay_lost == decay_values"


def test_q78_momentum_buffer_freezes_and_keeps_pushing():
    # Q7.8, resolution 2^-8. Step 1: g = 4 * 2^-8 = 0.015625, buf = g.
    # lr rounds to 55/256 = 0.21484375; lr * buf = 0.00336 = 0.86 resolution
    # rounds to one resolution. Steps 2-21, g = 0: 0.9 * 4 = 3.6 rounds to 4,
    # the buffer never decays, and each step removes one more resolution:
    # p = 1 - 21/256 = 0.91796875. In float64 the buffer decays and p ends
    # at 1 - 0.21544 * 0.015625 * (1 - 0.9^21) / 0.1 = 0.970.
    p = np.array([1.0])
    opt = lowprec.make("Momentum", [p], Policy.make("P1", Q7_8), lr=0.21544, momentum=0.9)
    opt.step([np.array([4 * 2**-8])])
    for _ in range(20):
        opt.step([np.array([0.0])])
    assert float(p[0]) == 0.91796875, f"ECHEC: obtenu {float(p[0])!r}, attendu 0.91796875"
    got = (opt.stats["frozen_buffer"], opt.stats["buffer_values"])
    assert got == (20, 20), f"ECHEC: obtenu {got}, attendu (20, 20)"


def test_fixed_point_adam_fails_on_zero_over_zero():
    # Q7.8: eps = 1e-8 rounds to 0. A zero gradient on the first step gives
    # m = 0, v = 0, denom = 0 and m / denom = NaN, which fixed point cannot hold.
    p = np.array([1.0])
    opt = lowprec.make("Adam", [p], Policy.make("P1", Q7_8), lr=1e-3)
    with pytest.raises(NonRepresentableError):
        opt.step([np.array([0.0])])


def test_gradient_underflow_is_counted():
    # 1e-9 is under half the smallest fp16 subnormal (2^-25 = 3e-8): it rounds to 0.
    p = np.ones(4)
    opt = lowprec.make("SGD", [p], Policy.make("P1", FP16), lr=0.1)
    opt.step([np.array([1e-9, 1e-9, 0.5, 0.0])])
    got = (opt.stats["grad_values"], opt.stats["grad_underflow"])
    assert got == (3, 2), f"ECHEC: obtenu {got}, attendu (3, 2)"


def test_parameters_are_rounded_to_the_format_at_start():
    p = np.array([1.0 + 2**-12, 0.1])
    lowprec.make("SGD", [p], Policy.make("P1", FP16), lr=0.1)
    expected = FP16.round([1.0 + 2**-12, 0.1])
    assert np.array_equal(p, expected), f"ECHEC: obtenu {p}, attendu {expected}"


@pytest.mark.parametrize(
    "call",
    [
        lambda: Policy.make("P4", FP16),
        lambda: lowprec.make("Adagrad", [np.ones(2)], Policy.make("P1", FP16)),
        lambda: lowprec.make("SGD", [np.ones(2)], Policy.make("P3", FP16), lr=0.1),  # no rng
        lambda: lowprec.make("SGD", [np.ones(2, dtype=np.float32)], Policy.make("P1", FP16), lr=0.1),
    ],
    ids=["unknown policy", "unknown optimizer", "stochastic without rng", "float32 storage"],
)
def test_invalid_use_is_rejected(call):
    with pytest.raises((ValueError, TypeError)):
        call()


def test_policy_labels():
    got = [Policy.make(n, BF16).label for n in ("P1", "P2", "P3")]
    assert got == ["bf16 P1", "bf16 P2", "bf16 P3"], f"ECHEC: obtenu {got}"

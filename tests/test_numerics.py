"""T1: the number format emulator, against NumPy, PyTorch and hand values."""

from fractions import Fraction

import numpy as np
import pytest
import torch

from lpopt.numerics import BF16, FORMATS, FP16, FP32, FP64, Q7_8, FixedFormat, FloatFormat

N_VALUES = 1_000_000


def wide_range_values(seed=0, n=N_VALUES):
    """Random float64 values from 1e-45 to 1e42: subnormals, normals and overflow."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal(n) * np.exp2(rng.uniform(-150, 140, n))


def same_values(a, b):
    """Equal as floats, with NaN == NaN and the sign of zero taken into account."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    both_nan = np.isnan(a) & np.isnan(b)
    return both_nan | ((a == b) & (np.signbit(a) == np.signbit(b)))


# --- conversion of a million values -----------------------------------------


@pytest.mark.parametrize("fmt, native", [(FP16, np.float16), (FP32, np.float32)], ids=["fp16", "fp32"])
def test_conversion_matches_numpy_bit_for_bit(fmt, native):
    x = wide_range_values()
    with np.errstate(over="ignore"):
        expected = x.astype(native).astype(np.float64)
    ok = same_values(fmt.round(x), expected)
    assert ok.all(), f"ECHEC {fmt.name}: {np.sum(~ok)} valeurs differentes, attendu 0 (ex. x={x[~ok][:3]})"


def test_bf16_conversion_matches_torch_on_float32_inputs():
    with np.errstate(over="ignore"):  # values past float32's range become inf on purpose
        x = wide_range_values().astype(np.float32)
    expected = torch.from_numpy(x).to(torch.bfloat16).double().numpy()
    ok = same_values(BF16.round(x.astype(np.float64)), expected)
    assert ok.all(), f"ECHEC: {np.sum(~ok)} valeurs differentes, attendu 0"


def test_bf16_from_float64_is_rounded_once_unlike_torch():
    # 1 + 2^-8 is the midpoint between 1 and 1 + 2^-7. Adding 2^-30 puts x just
    # above it, so the nearest bf16 is 1 + 2^-7 = 1.0078125.
    # torch converts float64 -> float32 -> bf16. The float32 step rounds x
    # onto the midpoint itself, then the tie goes to the even value 1.0.
    x = 1.0 + 2.0**-8 + 2.0**-30
    got = float(BF16.round(x))
    assert got == 1.0078125, f"ECHEC: obtenu {got!r}, attendu 1.0078125"
    torch_value = torch.tensor(x, dtype=torch.float64).to(torch.bfloat16).item()
    assert torch_value == 1.0, f"ECHEC: torch donne {torch_value!r}, attendu 1.0 (double arrondi)"


def test_where_torch_differs_on_float64_inputs_we_are_the_nearest():
    x = wide_range_values(seed=1)
    ours = BF16.round(x)
    theirs = torch.from_numpy(x).to(torch.bfloat16).double().numpy()
    differ = np.nonzero(~same_values(ours, theirs))[0]
    assert differ.size < 100, f"ECHEC: {differ.size} differences, attendu quelques-unes (double arrondi rare)"
    for i in differ:
        exact = Fraction(float(x[i]))
        d_ours = abs(exact - Fraction(float(ours[i])))
        d_theirs = abs(exact - Fraction(float(theirs[i])))
        assert d_ours < d_theirs, f"ECHEC x={x[i]!r}: obtenu {ours[i]!r}, torch {theirs[i]!r} est plus proche"


# --- hand values -------------------------------------------------------------


@pytest.mark.parametrize(
    "fmt, finfo",
    [(FP16, np.finfo(np.float16)), (FP32, np.finfo(np.float32))],
    ids=["fp16", "fp32"],
)
def test_limits_match_numpy_finfo(fmt, finfo):
    got = (fmt.max_value, fmt.min_normal, fmt.min_subnormal)
    expected = (float(finfo.max), float(finfo.smallest_normal), float(finfo.smallest_subnormal))
    assert got == expected, f"ECHEC {fmt.name}: obtenu {got}, attendu {expected}"


def test_bf16_limits_by_hand():
    # 8 exponent bits like fp32: emax = 127, emin = -126; 7 stored bits.
    # max = (2 - 2^-7) * 2^127, min normal = 2^-126, min subnormal = 2^-133
    got = (BF16.max_value, BF16.min_normal, BF16.min_subnormal)
    expected = ((2 - 2**-7) * 2.0**127, 2.0**-126, 2.0**-133)
    assert got == expected, f"ECHEC: obtenu {got}, attendu {expected}"
    finfo = torch.finfo(torch.bfloat16)
    assert BF16.max_value == finfo.max, f"ECHEC: obtenu {BF16.max_value}, torch {finfo.max}"
    sizes = (BF16.precision, BF16.bits, FP16.precision, FP16.bits)
    assert sizes == (8, 16, 11, 16), f"ECHEC: obtenu {sizes}, attendu (8, 16, 11, 16)"


@pytest.mark.parametrize(
    "x, expected",
    [
        (1.0 + 2**-11, 1.0),  # midpoint of 1 and 1 + 2^-10: tie, 1 is even
        (1.0 + 3 * 2**-11, 1.0 + 2**-9),  # midpoint of 1 + 2^-10 (odd) and 1 + 2^-9 (even)
        (65519.0, 65504.0),  # below max + ulp/2 = 65504 + 16
        (65520.0, np.inf),  # tie between 65504 (odd) and 65536 (even, overflow): inf
        (1e-8, 0.0),  # Adam's eps: under 2^-25 = 2.98e-8, half the smallest subnormal
        (3e-8, 2.0**-24),  # above 2^-25: rounds up to the smallest subnormal 5.96e-8
        (-1e-8, -0.0),  # the sign of zero is kept
    ],
)
def test_fp16_by_hand(x, expected):
    got = float(FP16.round(x))
    assert same_values(got, expected), f"ECHEC x={x!r}: obtenu {got!r}, attendu {expected!r}"


@pytest.mark.parametrize("fmt", [FP16, BF16, FP32], ids=lambda f: f.name)
@pytest.mark.parametrize("x", [0.0, -0.0, np.inf, -np.inf, np.nan])
def test_special_values_pass_through(fmt, x):
    got = float(fmt.round(x))
    assert same_values(got, x), f"ECHEC {fmt.name}: obtenu {got!r}, attendu {x!r}"


def test_adam_eps_survives_in_bf16_but_not_in_fp16():
    # bf16 has fp32's exponent range: 1e-8 is a normal number there, rounded
    # with 8 bits of precision (relative error <= 2^-8).
    bf = float(BF16.round(1e-8))
    assert abs(bf - 1e-8) <= 2**-8 * 1e-8, f"ECHEC: obtenu {bf!r}, attendu 1e-8 a 2^-8 pres"
    fp = float(FP16.round(1e-8))
    assert fp == 0.0, f"ECHEC: obtenu {fp!r}, attendu 0.0"


def all_positive_normals(fmt):
    """Every positive normal value of a 16-bit float format, exhaustively."""
    bits = np.arange(2**16, dtype=np.uint32)
    if fmt is FP16:
        values = bits.astype(np.uint16).view(np.float16).astype(np.float64)
    else:  # bf16 is the top half of a float32
        with np.errstate(invalid="ignore"):  # the NaN bit patterns, filtered below
            values = (bits << 16).view(np.float32).astype(np.float64)
    return values[np.isfinite(values) & (values >= fmt.min_normal)]


def test_multiplying_by_0999_never_changes_a_normal_bf16():
    # beta2 = 0.999 changes v by 0.1 %. bf16 values are spaced by 2^-8 to
    # 2^-7 of their magnitude (half of that below a power of two), so the
    # nearest bf16 to 0.999 * v is always v itself: v * beta2 rounds back to v.
    # Checked on all 32512 positive normal bf16 values.
    v = all_positive_normals(BF16)
    assert v.size == 254 * 128, f"ECHEC: obtenu {v.size} valeurs, attendu 32512"
    changed = BF16.round(0.999 * v) != v
    assert not changed.any(), f"ECHEC: {changed.sum()} valeurs changent, attendu 0"


def test_multiplying_by_0999_always_changes_a_normal_fp16():
    # fp16 spacing is 2^-11 to 2^-10 of the magnitude: 0.1 % is more than
    # half a spacing, so every normal value moves. (Subnormals are not
    # included: their relative spacing is larger.)
    v = all_positive_normals(FP16)
    assert v.size == 30 * 1024, f"ECHEC: obtenu {v.size} valeurs, attendu 30720"
    unchanged = FP16.round(0.999 * v) == v
    assert not unchanged.any(), f"ECHEC: {unchanged.sum()} valeurs inchangees, attendu 0"


# --- fixed point ---------------------------------------------------------------


@pytest.mark.parametrize(
    "x, expected",
    [
        (0.1, 26 / 256),  # 0.1 * 256 = 25.6 -> 26
        (2**-9, 0.0),  # half a resolution: tie, 0 is even
        (3 * 2**-9, 2**-7),  # 1.5 resolution: tie, 2 is even
        (1000.0, 127.99609375),  # saturates at 2^7 - 2^-8
        (127.998046875, 127.99609375),  # max + half resolution rounds up to 128, clipped back
        (-1000.0, -128.0),
        (np.inf, 127.99609375),
        (-np.inf, -128.0),
    ],
)
def test_q7_8_by_hand(x, expected):
    got = float(Q7_8.round(x))
    assert got == expected, f"ECHEC x={x!r}: obtenu {got!r}, attendu {expected!r}"


def test_fixed_point_rejects_nan():
    with pytest.raises(ValueError):
        Q7_8.round([1.0, np.nan])


def test_fixed_point_ulp_and_bits():
    got = (Q7_8.bits, float(Q7_8.ulp(3.0)), Q7_8.max_value, Q7_8.min_value)
    expected = (16, 2**-8, 127.99609375, -128.0)
    assert got == expected, f"ECHEC: obtenu {got}, attendu {expected}"


# --- stochastic rounding -------------------------------------------------------------


@pytest.mark.parametrize("fmt", [FP16, BF16, Q7_8], ids=lambda f: f.name)
def test_stochastic_rounding_is_unbiased(fmt):
    # x sits 30 % of the way between two neighbors. Each draw rounds up with
    # probability 0.3. Over N draws the mean has a standard deviation of
    # ulp * sqrt(0.3 * 0.7 / N): with N = 200 000 that is 1.0e-3 ulp, and the
    # test allows 5 of them (a false failure has probability ~6e-7).
    n = 200_000
    base = 1.0
    ulp = float(fmt.ulp(base))
    x = np.full(n, base + 0.3 * ulp)
    rounded = fmt.round(x, "stochastic", np.random.default_rng(0))
    values = set(np.unique(rounded).tolist())
    assert values == {base, base + ulp}, f"ECHEC: obtenu {values}, attendu {{{base}, {base + ulp}}}"
    bias = abs(rounded.mean() - x[0]) / ulp
    bound = 5 * np.sqrt(0.3 * 0.7 / n)
    assert bias < bound, f"ECHEC: biais {bias:.2e} ulp, attendu < {bound:.2e}"


def test_stochastic_rounding_keeps_representable_values():
    x = all_positive_normals(BF16)
    got = BF16.round(x, "stochastic", np.random.default_rng(0))
    assert np.array_equal(got, x), "ECHEC: une valeur deja representable a ete modifiee"


def test_nearest_rounding_drops_small_updates_that_stochastic_keeps_on_average():
    # 1000 updates of +0.1 ulp from 1.0: round-to-nearest loses every one,
    # stochastic rounding moves by 100 ulp on average.
    ulp = float(BF16.ulp(1.0))
    rng = np.random.default_rng(0)
    nearest = np.array(1.0)
    stochastic = np.array(1.0)
    for _ in range(1000):
        nearest = BF16.round(nearest + 0.1 * ulp)
        stochastic = BF16.round(stochastic + 0.1 * ulp, "stochastic", rng)
    assert float(nearest) == 1.0, f"ECHEC: obtenu {float(nearest)!r}, attendu 1.0"
    moved = (float(stochastic) - 1.0) / ulp
    # Binomial(1000, 0.1): mean 100, std 9.5; 5 std allowed.
    assert abs(moved - 100) < 5 * 9.5, f"ECHEC: obtenu {moved} ulp, attendu 100 +- 47"


# --- argument checks -----------------------------------------------------------------


@pytest.mark.parametrize("fmt", list(FORMATS.values()), ids=list(FORMATS))
def test_rounding_mode_is_checked(fmt):
    with pytest.raises(ValueError):
        fmt.round(1.0, "truncate")
    with pytest.raises(ValueError):
        fmt.round(1.0, "stochastic")  # no rng


@pytest.mark.parametrize(
    "make",
    [
        lambda: FloatFormat("too precise", 8, 30),
        lambda: FloatFormat("no exponent", 1, 10),
        lambda: FixedFormat("negative", -1, 8),
        lambda: FixedFormat("too wide", 30, 30),
    ],
)
def test_invalid_formats_are_rejected(make):
    with pytest.raises(ValueError):
        make()


def test_fp64_is_the_identity():
    x = wide_range_values(n=1000)
    got = FP64.round(x)
    assert np.array_equal(got, x), "ECHEC: FP64.round modifie les valeurs"
    assert np.array_equal(FP64.ulp(x), np.spacing(np.abs(x))), "ECHEC: FP64.ulp"


@pytest.mark.parametrize("fmt", [FP16, BF16, FP32], ids=lambda f: f.name)
def test_ulp_is_the_gap_to_the_next_value(fmt):
    x = fmt.round(np.abs(wide_range_values(n=10_000)))
    x = x[np.isfinite(x) & (x < fmt.max_value)]
    nxt = x + fmt.ulp(x)
    assert np.array_equal(fmt.round(nxt), nxt), "ECHEC: x + ulp(x) n'est pas representable"
    mid = x + 0.75 * fmt.ulp(x)  # past the midpoint: must round up to x + ulp, not stay at x
    assert np.array_equal(fmt.round(mid), nxt), "ECHEC: il existe une valeur entre x et x + ulp(x)"

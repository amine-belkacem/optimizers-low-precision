"""Number formats emulated in float64: fp32, fp16, bf16 and fixed point Qm.n.

A format is applied by rounding: `fmt.round(x)` returns the float64 array
of the values of `fmt` closest to `x`. To emulate an operation in a format,
compute it in float64 and round the result:

    fp16 sum of a and b  ==  FP16.round(a + b)    (a, b already fp16 values)

This is exact, not an approximation. For +, -, *, / and sqrt, rounding the
float64 result to a format with p bits of precision gives the same value as
rounding the exact result directly, as long as 53 >= 2p + 2 (Figueroa,
"When is double rounding innocuous?", SIGNUM Newsletter, 1995). fp32 has
p = 24, fp16 p = 11, bf16 p = 8: all three qualify. tests/test_numerics.py
checks it against the native NumPy and PyTorch arithmetic.

Two rounding modes:
* "nearest": round to nearest, ties to even, the IEEE 754 default.
* "stochastic": round up with a probability equal to the distance to the
  lower neighbor, in units of the spacing. The result is unbiased:
  E[round(x)] = x. This is the usual fix for updates smaller than half a
  spacing, which round-to-nearest drops every time.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ROUNDING_MODES = ("nearest", "stochastic")


class NonRepresentableError(ValueError):
    """A value has no representation in the format (NaN in fixed point).

    In training this is a failure of the run, like an infinite loss in
    floating point, and the training loops treat it that way.
    """


def _check_mode(rounding: str, rng: np.random.Generator | None) -> None:
    if rounding not in ROUNDING_MODES:
        raise ValueError(f"rounding must be one of {ROUNDING_MODES}, got {rounding!r}")
    if rounding == "stochastic" and rng is None:
        raise ValueError("stochastic rounding needs a numpy Generator (rng=...)")


def _round_integers(scaled: np.ndarray, rounding: str, rng: np.random.Generator | None) -> np.ndarray:
    """Round float64 values to integers. `scaled` must hold exact values."""
    if rounding == "nearest":
        return np.rint(scaled)  # IEEE round half to even
    low = np.floor(scaled)
    # scaled - low is exact (Sterbenz), u < frac happens with probability frac
    return low + (rng.random(scaled.shape) < (scaled - low))


@dataclass(frozen=True)
class FloatFormat:
    """IEEE-style binary floating point with subnormals, infinities and NaN.

    `mantissa_bits` counts the stored fraction bits (the leading 1 of a
    normal number is implicit), so the precision is mantissa_bits + 1.
    """

    name: str
    exponent_bits: int
    mantissa_bits: int

    def __post_init__(self) -> None:
        # 2p + 2 <= 53, i.e. at most 24 stored bits, keeps the float64
        # emulation exact (see module docstring). The exponent must also fit
        # comfortably inside float64's 11 bits.
        if not 2 <= self.exponent_bits <= 10 or not 1 <= self.mantissa_bits <= 24:
            raise ValueError(f"{self.name}: unsupported format e={self.exponent_bits}, m={self.mantissa_bits}")

    @property
    def precision(self) -> int:
        return self.mantissa_bits + 1

    @property
    def emax(self) -> int:
        return 2 ** (self.exponent_bits - 1) - 1

    @property
    def emin(self) -> int:
        return 1 - self.emax

    @property
    def max_value(self) -> float:
        return (2.0 - 2.0**-self.mantissa_bits) * 2.0**self.emax

    @property
    def min_normal(self) -> float:
        return 2.0**self.emin

    @property
    def min_subnormal(self) -> float:
        return 2.0 ** (self.emin - self.mantissa_bits)

    @property
    def bits(self) -> int:
        return 1 + self.exponent_bits + self.mantissa_bits

    def _exponent(self, a: np.ndarray) -> np.ndarray:
        """floor(log2(a)) for a > 0, clamped to emin (subnormal range)."""
        _, k = np.frexp(a)  # a = f * 2^k with f in [0.5, 1)
        return np.maximum(k - 1, self.emin)

    def ulp(self, x) -> np.ndarray:
        """Spacing between |x| and the next larger value of the format."""
        a = np.abs(np.asarray(x, dtype=np.float64))
        e = self._exponent(np.where(a > 0, a, self.min_normal))
        return np.ldexp(1.0, e - self.mantissa_bits)

    def round(self, x, rounding: str = "nearest", rng: np.random.Generator | None = None) -> np.ndarray:
        _check_mode(rounding, rng)
        x = np.asarray(x, dtype=np.float64)
        a = np.abs(x)
        finite = np.isfinite(a) & (a > 0)
        # Work only on finite non-zero values: zeros keep their sign, inf and
        # NaN pass through unchanged, exactly as an IEEE conversion does.
        safe = np.where(finite, a, 1.0)
        quantum = np.ldexp(1.0, self._exponent(safe) - self.mantissa_bits)
        # Dividing by a power of two is exact, and the quotient is below
        # 2^(p+1), so it holds an exact float64 value.
        n = _round_integers(safe / quantum, rounding, rng)
        rounded = n * quantum
        # A value that rounds past max_value overflows to infinity. The tie at
        # max_value + ulp/2 rounds to the even neighbor 2^(emax+1), i.e. to inf,
        # as IEEE 754 specifies.
        rounded = np.where(rounded > self.max_value, np.inf, rounded)
        return np.where(finite, np.copysign(rounded, x), x)


@dataclass(frozen=True)
class FixedFormat:
    """Signed two's complement fixed point Qm.n: m integer bits, n fraction bits.

    m + n + 1 bits in total (with the sign). Values are k * 2^-n with k an
    integer, from -2^m to 2^m - 2^-n. Out-of-range values saturate, which is
    what a DSP or an FPGA datapath with saturation logic does: there is no
    infinity in fixed point.
    """

    name: str
    integer_bits: int
    fraction_bits: int

    def __post_init__(self) -> None:
        if self.integer_bits < 0 or self.fraction_bits < 0 or self.bits > 52:
            raise ValueError(f"{self.name}: unsupported Q{self.integer_bits}.{self.fraction_bits}")

    @property
    def bits(self) -> int:
        return self.integer_bits + self.fraction_bits + 1

    @property
    def resolution(self) -> float:
        return 2.0**-self.fraction_bits

    @property
    def max_value(self) -> float:
        return 2.0**self.integer_bits - self.resolution

    @property
    def min_value(self) -> float:
        return -(2.0**self.integer_bits)

    def ulp(self, x) -> np.ndarray:
        return np.full(np.shape(x), self.resolution)

    def round(self, x, rounding: str = "nearest", rng: np.random.Generator | None = None) -> np.ndarray:
        _check_mode(rounding, rng)
        x = np.asarray(x, dtype=np.float64)
        if np.isnan(x).any():
            raise NonRepresentableError(f"{self.name}: NaN has no fixed-point representation")
        clipped = np.clip(x, self.min_value, self.max_value)  # also saturates +-inf
        n = _round_integers(np.ldexp(clipped, self.fraction_bits), rounding, rng)
        # Rounding can step one resolution past max_value (e.g. 2^m - 2^-(n+1)),
        # clip again after rounding.
        return np.clip(np.ldexp(n, -self.fraction_bits), self.min_value, self.max_value)


@dataclass(frozen=True)
class Float64:
    """The reference: values are already float64, rounding does nothing."""

    name: str = "fp64"
    bits: int = 64

    def round(self, x, rounding: str = "nearest", rng: np.random.Generator | None = None) -> np.ndarray:
        _check_mode(rounding, rng)
        return np.asarray(x, dtype=np.float64)

    def ulp(self, x) -> np.ndarray:
        return np.spacing(np.abs(np.asarray(x, dtype=np.float64)))


FP64 = Float64()
FP32 = FloatFormat("fp32", exponent_bits=8, mantissa_bits=23)
FP16 = FloatFormat("fp16", exponent_bits=5, mantissa_bits=10)
BF16 = FloatFormat("bf16", exponent_bits=8, mantissa_bits=7)
Q7_8 = FixedFormat("Q7.8", integer_bits=7, fraction_bits=8)
Q15_16 = FixedFormat("Q15.16", integer_bits=15, fraction_bits=16)

FORMATS = {f.name: f for f in (FP64, FP32, FP16, BF16, Q7_8, Q15_16)}

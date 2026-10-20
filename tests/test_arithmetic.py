"""Emulated arithmetic against native low-precision arithmetic.

The emulator's claim (src/lpopt/numerics.py): computing an operation in
float64 and rounding the result to the format gives the correctly rounded
result in that format, for +, -, *, / and sqrt. Here it is checked bit for
bit on a million random operand pairs per operation and format, against:

* NumPy float32 (native IEEE arithmetic);
* NumPy float16, which computes in float32 and rounds back. That double
  rounding is itself innocuous since 24 >= 2 * 11 + 2;
* PyTorch bfloat16 on CPU, which also computes in float32 and rounds back
  (24 >= 2 * 8 + 2).

Operands are random bit patterns, not random reals, so every exponent of
the format shows up, subnormals and the largest values included.
"""

import numpy as np
import pytest
import torch

from lpopt.numerics import BF16, FP16, FP32

N_PAIRS = 1_000_000
OPS = ("add", "sub", "mul", "div", "sqrt")


def random_operands(fmt, n, seed):
    rng = np.random.default_rng(seed)
    with np.errstate(invalid="ignore"):
        if fmt is FP16:
            v = rng.integers(0, 2**16, n, dtype=np.uint16).view(np.float16).astype(np.float64)
        elif fmt is BF16:
            bits = rng.integers(0, 2**16, n, dtype=np.uint32) << 16
            v = bits.view(np.float32).astype(np.float64)
        else:
            v = rng.integers(0, 2**32, n, dtype=np.uint64).astype(np.uint32).view(np.float32).astype(np.float64)
    return v[np.isfinite(v)]


def emulated(op, a, b):
    with np.errstate(all="ignore"):
        return {
            "add": lambda: a + b,
            "sub": lambda: a - b,
            "mul": lambda: a * b,
            "div": lambda: a / b,
            "sqrt": lambda: np.sqrt(np.abs(a)),
        }[op]()


def native(fmt, op, a, b):
    if fmt is BF16:
        ta = torch.from_numpy(a.astype(np.float32)).to(torch.bfloat16)
        tb = torch.from_numpy(b.astype(np.float32)).to(torch.bfloat16)
        result = {"add": ta + tb, "sub": ta - tb, "mul": ta * tb, "div": ta / tb, "sqrt": torch.sqrt(ta.abs())}[op]
        return result.double().numpy()
    dtype = np.float16 if fmt is FP16 else np.float32
    with np.errstate(all="ignore"):
        return emulated(op, a.astype(dtype), b.astype(dtype)).astype(np.float64)


def same_values(a, b):
    both_nan = np.isnan(a) & np.isnan(b)
    return both_nan | ((a == b) & (np.signbit(a) == np.signbit(b)))


@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize("fmt", [FP32, FP16, BF16], ids=lambda f: f.name)
def test_float64_then_round_equals_native_arithmetic(fmt, op):
    a = random_operands(fmt, N_PAIRS, seed=1)
    b = random_operands(fmt, N_PAIRS, seed=2)
    n = min(a.size, b.size)
    a, b = a[:n], b[:n]
    ours = fmt.round(emulated(op, a, b))
    theirs = native(fmt, op, a, b)
    ok = same_values(ours, theirs)
    bad = np.nonzero(~ok)[0][:3]
    assert ok.all(), (
        f"ECHEC {fmt.name} {op}: {np.sum(~ok)} resultats differents sur {n}, attendu 0 "
        f"(ex. a={a[bad]}, b={b[bad]}, obtenu {ours[bad]}, attendu {theirs[bad]})"
    )

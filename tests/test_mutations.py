"""T4: inject plausible bugs into the optimizers and check the bench sees them.

Each mutant is a copy of src/lpopt/optimizers.py with one line changed, the
kind of mistake that still trains and converges a bit worse, so nobody
notices. The mutated source is compiled into a throwaway module and run
through the same differential comparison as test_against_torch.py. A
mutant that stays under the float64 threshold on every scenario would mean
the bench has a blind spot, and the test fails.
"""

import types
from pathlib import Path

import numpy as np
import pytest

import lpopt.optimizers
from experiments.torch_reference import ALL_SCENARIOS, compare_with_torch

SOURCE = Path(lpopt.optimizers.__file__).read_text(encoding="utf-8")
DETECTION_THRESHOLD = 1e-10  # same as the float64 threshold of T3
N_STEPS = 50

MUTANTS = {
    "Adam without bias correction of m": ("bias_correction1 = 1.0 - beta1**t", "bias_correction1 = 1.0"),
    "Adam without bias correction of v": ("bias_correction2 = 1.0 - beta2**t", "bias_correction2 = 1.0"),
    "eps inside the square root": (
        "denom = np.sqrt(v) / math.sqrt(bias_correction2) + self.eps",
        "denom = np.sqrt(v / bias_correction2 + self.eps)",
    ),
    "AdamW decay coupled to the gradient": ("decoupled_weight_decay = True", "decoupled_weight_decay = False"),
    "AdamW decay not scaled by lr": (
        "p *= 1.0 - self.lr * self.weight_decay",
        "p *= 1.0 - self.weight_decay",
    ),
    "beta2 used in the first moment": ("m += (1.0 - beta1) * g", "m += (1.0 - beta2) * g"),
    "dampening applied on the first momentum step": ("buf = g.copy()", "buf = (1.0 - self.dampening) * g"),
    "Nesterov look-ahead missing": (
        "g = g + self.momentum * buf if self.nesterov else buf",
        "g = buf",
    ),
    "SGD weight decay with the wrong sign": (
        "g = g + self.weight_decay * p  # L2 term of SGD",
        "g = g - self.weight_decay * p  # L2 term of SGD",
    ),
    "step counter stuck at 1": ("self.step_count += 1", "self.step_count = 1"),
}


def build_mutant(original: str, mutated: str) -> types.ModuleType:
    count = SOURCE.count(original)
    assert count == 1, f"ECHEC: la ligne a muter apparait {count} fois, attendu 1: {original!r}"
    module = types.ModuleType("lpopt_mutant")
    exec(compile(SOURCE.replace(original, mutated), "lpopt_mutant", "exec"), module.__dict__)
    return module


def worst_error(module: types.ModuleType) -> float:
    return max(float(np.max(compare_with_torch(s, N_STEPS, seed=3, module=module))) for s in ALL_SCENARIOS)


def test_unmutated_copy_passes():
    # Guards the harness itself: if the copy failed, every mutant would be
    # "detected" for the wrong reason.
    got = worst_error(build_mutant("import math", "import math"))
    assert got <= DETECTION_THRESHOLD, f"ECHEC: obtenu {got:.3e}, attendu <= {DETECTION_THRESHOLD:.0e}"


@pytest.mark.parametrize("name", list(MUTANTS))
def test_mutant_is_detected(name):
    module = build_mutant(*MUTANTS[name])
    got = worst_error(module)
    assert got > DETECTION_THRESHOLD, f"ECHEC {name}: obtenu {got:.3e}, attendu > {DETECTION_THRESHOLD:.0e} (mutant non detecte)"

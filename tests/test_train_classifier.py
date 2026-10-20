"""T7 and the classifier training loop, on the real digits split."""

import numpy as np
import pytest

from experiments.digits import load_split
from lpopt.mlp import MLP
from lpopt.optimizers import SGD, Adam
from lpopt.train import train_classifier


@pytest.fixture(scope="module")
def data():
    return load_split()


def train(data, seed, optimizer_class=Adam, epochs=3, **kwargs):
    rng = np.random.default_rng(seed)
    model = MLP((64, 32, 10))
    params = model.init_params(rng)
    history = train_classifier(model, params, optimizer_class(params, **kwargs), data, epochs, 64, rng)
    return history, params


def test_split_sizes_are_disjoint_and_stratified(data):
    # 1797 samples: test_size 0.4 -> ceil(718.8) = 719 held out, 1078 train;
    # then 719 split in half -> ceil(359.5) = 360 test, 359 validation.
    sizes = (len(data.y_train), len(data.y_val), len(data.y_test))
    assert sizes == (1078, 359, 360), f"ECHEC: obtenu {sizes}, attendu (1078, 359, 360)"
    assert sum(sizes) == 1797, f"ECHEC: obtenu {sum(sizes)} echantillons, attendu 1797"
    # Each class has 174 to 183 images, and 20 % of 174 is 34.8: a stratified
    # split cannot leave fewer than 34 images of a class in val or test.
    for name, y in (("val", data.y_val), ("test", data.y_test)):
        counts = np.bincount(y, minlength=10)
        assert counts.min() >= 34, f"ECHEC {name}: obtenu {counts}, attendu au moins 34 par classe"
    assert 0.0 <= data.X_train.min() and data.X_train.max() <= 1.0, "ECHEC: pixels hors de [0, 1]"


def test_same_seed_gives_bit_identical_runs(data):
    h1, p1 = train(data, seed=7, lr=1e-2)
    h2, p2 = train(data, seed=7, lr=1e-2)
    assert h1.train_loss == h2.train_loss, f"ECHEC: obtenu {h1.train_loss} puis {h2.train_loss}"
    for a, b in zip(p1, p2):
        assert np.array_equal(a, b), "ECHEC: parametres differents pour la meme graine"


def test_different_seeds_give_different_runs(data):
    h1, _ = train(data, seed=7, lr=1e-2)
    h2, _ = train(data, seed=8, lr=1e-2)
    assert h1.train_loss != h2.train_loss, "ECHEC: deux graines differentes donnent le meme entrainement"


def test_training_counts_steps_and_learns(data):
    history, _ = train(data, seed=0, epochs=3, lr=1e-2)
    # ceil(1078 / 64) = 17 mini-batches per epoch, the last one has 54 samples
    assert history.n_steps == 3 * 17, f"ECHEC: obtenu {history.n_steps} pas, attendu 51"
    assert len(history.train_loss) == 4, f"ECHEC: obtenu {len(history.train_loss)} points, attendu 4"
    assert history.train_loss[-1] < history.train_loss[0], f"ECHEC: la perte n'a pas baisse: {history.train_loss}"


def test_divergence_is_detected(data):
    history, _ = train(data, seed=0, optimizer_class=SGD, epochs=5, lr=1e15)
    assert history.diverged, "ECHEC: obtenu diverged=False, attendu True"
    assert all(np.isfinite(history.train_loss)), f"ECHEC: valeurs non finies conservees: {history.train_loss}"


def test_on_epoch_is_called_after_each_epoch(data):
    rng = np.random.default_rng(0)
    model = MLP((64, 32, 10))
    params = model.init_params(rng)
    seen = []
    train_classifier(model, params, Adam(params, lr=1e-2), data, 3, 64, rng, on_epoch=seen.append)
    assert seen == [1, 2, 3], f"ECHEC: obtenu {seen}, attendu [1, 2, 3]"


def test_fixed_point_adam_fails_cleanly(data):
    # Adam with Q7.8 everywhere divides by a zero denominator during the
    # first epoch (see test_lowprec.py): the run must stop, not crash.
    from lpopt.lowprec import Policy, make
    from lpopt.numerics import Q7_8

    rng = np.random.default_rng(0)
    model = MLP((64, 32, 10))
    params = model.init_params(rng)
    opt = make("Adam", params, Policy.make("P1", Q7_8), lr=2e-2)
    history = train_classifier(model, params, opt, data, 2, 64, rng)
    assert history.diverged, "ECHEC: obtenu diverged=False, attendu True"
    assert len(history.val_accuracy) == 1, f"ECHEC: obtenu {len(history.val_accuracy)} points, attendu 1"


def test_invalid_epochs_rejected(data):
    with pytest.raises(ValueError):
        train(data, seed=0, epochs=0, lr=1e-2)

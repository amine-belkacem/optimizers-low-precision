"""T5: the hand-written backward pass of the MLP."""

import math

import numpy as np
import pytest
import torch

from lpopt.mlp import MLP


def test_zero_network_gives_log10_loss_and_hand_gradients():
    # Zero weights: logits = 0, softmax = 1/10 everywhere, loss = ln(10).
    # dz2 = (0.1 - onehot(y)) / 4. h1 = tanh(0) = 0 so dW2 = 0. W2 = 0 so the
    # signal sent back to layer 1 is 0: dW1 = db1 = 0.
    # db2 = mean over the batch of (0.1 - onehot(y)), with y = [0, 0, 1, 2]:
    #   class 0: 0.1 - 2/4 = -0.4 ; classes 1 and 2: 0.1 - 1/4 = -0.15 ; others: 0.1
    model = MLP((3, 4, 10))
    params = [np.zeros((3, 4)), np.zeros(4), np.zeros((4, 10)), np.zeros(10)]
    X = np.array([[0.2, -1.0, 3.0], [1.0, 1.0, 1.0], [0.0, 0.5, -0.5], [2.0, 0.0, 1.0]])
    y = np.array([0, 0, 1, 2])
    loss, grads = model.loss_and_grads(params, X, y)
    expected_loss = math.log(10)
    assert loss == pytest.approx(expected_loss, rel=1e-15), f"ECHEC: obtenu {loss!r}, attendu {expected_loss!r}"
    expected_db2 = np.array([-0.4, -0.15, -0.15] + [0.1] * 7)
    assert np.allclose(grads[3], expected_db2, rtol=0, atol=1e-16), f"ECHEC: obtenu {grads[3]}, attendu {expected_db2}"
    for i in (0, 1, 2):
        assert not grads[i].any(), f"ECHEC: obtenu grads[{i}] = {grads[i]}, attendu 0"


def test_single_layer_softmax_by_hand():
    # logits = [1, 0], p0 = e / (e + 1) = 0.7310585786300049
    # loss = -ln(p0) = ln(1 + e^-1) = 0.31326168751822286
    # dW = x^T (p - onehot) = [[p0 - 1, 1 - p0], [0, 0]], db = [p0 - 1, 1 - p0]
    model = MLP((2, 2))
    params = [np.array([[1.0, 0.0], [0.0, 0.0]]), np.zeros(2)]
    loss, grads = model.loss_and_grads(params, np.array([[1.0, 0.0]]), np.array([0]))
    assert loss == pytest.approx(0.31326168751822286, rel=1e-15), f"ECHEC: obtenu {loss!r}, attendu 0.31326168751822286"
    a = 0.2689414213699951  # 1 - p0
    expected = [np.array([[-a, a], [0.0, 0.0]]), np.array([-a, a])]
    for got, exp in zip(grads, expected):
        assert np.allclose(got, exp, rtol=1e-15, atol=0), f"ECHEC: obtenu {got}, attendu {exp}"


def test_cross_entropy_does_not_overflow_on_huge_logits():
    # logits [1000, 0], label 1: loss = logsumexp - z_1 = 1000 + ln(1 + e^-1000) - 0 = 1000
    model = MLP((1, 2))
    params = [np.array([[1000.0, 0.0]]), np.zeros(2)]
    loss = model.loss(params, np.array([[1.0]]), np.array([1]))
    assert loss == 1000.0, f"ECHEC: obtenu {loss!r}, attendu 1000.0"


def random_problem(seed=0, sizes=(5, 4, 3), n=6):
    rng = np.random.default_rng(seed)
    model = MLP(sizes)
    params = [p + 0.1 * rng.standard_normal(p.shape) for p in model.init_params(rng)]  # non-zero biases
    X = rng.standard_normal((n, sizes[0]))
    y = rng.integers(0, sizes[-1], n)
    return model, params, X, y


@pytest.mark.parametrize("sizes", [(5, 3), (5, 4, 3), (5, 6, 4, 3)], ids=["no hidden", "1 hidden", "2 hidden"])
def test_gradient_matches_central_differences(sizes):
    model, params, X, y = random_problem(sizes=sizes)
    _, grads = model.loss_and_grads(params, X, y)
    h = 1e-6
    for k, p in enumerate(params):
        numeric = np.empty_like(p)
        for idx in np.ndindex(p.shape):
            saved = p[idx]
            p[idx] = saved + h
            up = model.loss(params, X, y)
            p[idx] = saved - h
            down = model.loss(params, X, y)
            p[idx] = saved
            numeric[idx] = (up - down) / (2 * h)
        # Central differences: O(h^2) truncation plus ~1e-16 / h cancellation,
        # both well under 1e-8 here.
        err = np.max(np.abs(grads[k] - numeric))
        assert err < 1e-8, f"ECHEC param {k}: obtenu ecart {err:.2e}, attendu < 1e-8"


def test_gradient_matches_torch_autograd():
    model, params, X, y = random_problem(seed=1, sizes=(64, 32, 10), n=50)
    _, grads = model.loss_and_grads(params, X, y)

    t = [torch.tensor(p, requires_grad=True) for p in params]
    h = torch.tanh(torch.tensor(X) @ t[0] + t[1])
    loss = torch.nn.functional.cross_entropy(h @ t[2] + t[3], torch.tensor(y))
    loss.backward()
    for k, (ours, ref) in enumerate(zip(grads, t)):
        ref = ref.grad.numpy()
        err = np.max(np.abs(ours - ref)) / np.max(np.abs(ref))
        assert err < 1e-12, f"ECHEC param {k}: obtenu ecart relatif {err:.2e}, attendu < 1e-12"
    assert model.loss(params, X, y) == pytest.approx(loss.item(), rel=1e-13), "ECHEC: perte differente de torch"


def test_float32_stays_float32():
    model, params, X, y = random_problem()
    params32 = [p.astype(np.float32) for p in params]
    loss, grads = model.loss_and_grads(params32, X, y)
    dtypes = {g.dtype for g in grads}
    assert dtypes == {np.dtype(np.float32)}, f"ECHEC: obtenu {dtypes}, attendu {{float32}}"


def test_accuracy_by_hand():
    # identity weights: predicted class = argmax of the input
    model = MLP((3, 3))
    params = [np.eye(3), np.zeros(3)]
    X = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 1.0], [0.0, 0.0, 5.0], [3.0, 4.0, 0.0]])
    # argmax per row: 0, 1, 2, 1 against labels 0, 1, 1, 0: 2 right out of 4
    got = model.accuracy(params, X, np.array([0, 1, 1, 0]))
    assert got == 0.5, f"ECHEC: obtenu {got}, attendu 0.5"


def test_glorot_init():
    rng = np.random.default_rng(0)
    params = MLP((64, 32, 10)).init_params(rng)
    shapes = [p.shape for p in params]
    assert shapes == [(64, 32), (32,), (32, 10), (10,)], f"ECHEC: obtenu {shapes}"
    limit = math.sqrt(6 / (64 + 32))
    assert np.abs(params[0]).max() <= limit, f"ECHEC: poids hors de [-{limit}, {limit}]"
    assert not params[1].any() and not params[3].any(), "ECHEC: biais non nuls"


@pytest.mark.parametrize(
    "call",
    [
        lambda m, p: MLP((5,)),
        lambda m, p: MLP((5, 0, 3)),
        lambda m, p: m.loss_and_grads(p[:2], np.ones((2, 5)), np.array([0, 1])),
        lambda m, p: m.loss_and_grads(p, np.ones((2, 4)), np.array([0, 1])),
        lambda m, p: m.loss_and_grads(p, np.ones((2, 5)), np.array([0, 3])),
        lambda m, p: m.loss_and_grads(p, np.ones((2, 5)), np.array([0])),
    ],
    ids=["one size", "zero width", "missing params", "wrong width", "label too big", "wrong y length"],
)
def test_invalid_inputs_are_rejected(call):
    model, params, _, _ = random_problem()
    with pytest.raises(ValueError):
        call(model, params)

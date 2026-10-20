"""Multilayer perceptron with the backward pass written by hand.

    h_0 = X
    z_l = h_{l-1} @ W_l + b_l
    h_l = tanh(z_l)                      hidden layers
    loss = mean_i( logsumexp(z_L[i]) - z_L[i, y_i] )   softmax cross-entropy

Backward, with N the batch size and p = softmax(z_L):

    dz_L      = (p - onehot(y)) / N
    dW_l      = h_{l-1}^T @ dz_l
    db_l      = sum over the batch of dz_l
    dz_{l-1}  = (dz_l @ W_l^T) * (1 - h_{l-1}^2)      since tanh' = 1 - tanh^2

tanh rather than ReLU: it is smooth everywhere, so the gradient check
against finite differences has no kink to trip on.

Everything runs in the dtype of the parameters, inputs included, so the
same code will be reused unchanged for the low-precision study.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


class MLP:
    """Fully connected tanh network ending with a softmax cross-entropy.

    The parameters are not stored in the object: they are a plain list
    [W_1, b_1, W_2, b_2, ...] that the optimizer updates in place.
    """

    def __init__(self, layer_sizes: Sequence[int]) -> None:
        sizes = tuple(int(s) for s in layer_sizes)
        if len(sizes) < 2 or any(s < 1 for s in sizes):
            raise ValueError(f"need at least an input and an output size, all >= 1, got {layer_sizes}")
        self.layer_sizes = sizes

    @property
    def n_layers(self) -> int:
        return len(self.layer_sizes) - 1

    def init_params(self, rng: np.random.Generator, dtype=np.float64) -> list[np.ndarray]:
        """Glorot uniform weights (suited to tanh), zero biases."""
        params = []
        for fan_in, fan_out in zip(self.layer_sizes[:-1], self.layer_sizes[1:]):
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            params.append(rng.uniform(-limit, limit, size=(fan_in, fan_out)).astype(dtype))
            params.append(np.zeros(fan_out, dtype=dtype))
        return params

    def _check(self, params: Sequence[np.ndarray], X: np.ndarray) -> None:
        if len(params) != 2 * self.n_layers:
            raise ValueError(f"expected {2 * self.n_layers} parameter arrays, got {len(params)}")
        if X.ndim != 2 or X.shape[1] != self.layer_sizes[0]:
            raise ValueError(f"X must have shape (N, {self.layer_sizes[0]}), got {X.shape}")

    def logits(self, params: Sequence[np.ndarray], X: np.ndarray) -> np.ndarray:
        self._check(params, X)
        h = X.astype(params[0].dtype, copy=False)
        for layer in range(self.n_layers):
            z = h @ params[2 * layer] + params[2 * layer + 1]
            h = np.tanh(z) if layer < self.n_layers - 1 else z
        return h

    @staticmethod
    def _cross_entropy(z: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
        """Mean cross-entropy and softmax probabilities, computed stably."""
        shifted = z - z.max(axis=1, keepdims=True)  # exp never overflows
        log_sum = np.log(np.exp(shifted).sum(axis=1))
        loss = float(np.mean(log_sum - shifted[np.arange(len(y)), y]))
        probs = np.exp(shifted - log_sum[:, None])
        return loss, probs

    def loss(self, params: Sequence[np.ndarray], X: np.ndarray, y: np.ndarray) -> float:
        return self._cross_entropy(self.logits(params, X), y)[0]

    def accuracy(self, params: Sequence[np.ndarray], X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(np.argmax(self.logits(params, X), axis=1) == y))

    def loss_and_grads(
        self, params: Sequence[np.ndarray], X: np.ndarray, y: np.ndarray
    ) -> tuple[float, list[np.ndarray]]:
        self._check(params, X)
        if y.shape != (X.shape[0],):
            raise ValueError(f"y must have shape ({X.shape[0]},), got {y.shape}")
        if y.min() < 0 or y.max() >= self.layer_sizes[-1]:
            raise ValueError(f"labels must be in [0, {self.layer_sizes[-1]}), got [{y.min()}, {y.max()}]")

        # Forward, keeping every activation for the backward pass.
        activations = [X.astype(params[0].dtype, copy=False)]
        for layer in range(self.n_layers):
            z = activations[-1] @ params[2 * layer] + params[2 * layer + 1]
            activations.append(np.tanh(z) if layer < self.n_layers - 1 else z)
        loss, probs = self._cross_entropy(activations[-1], y)

        # Backward.
        n = X.shape[0]
        dz = probs
        dz[np.arange(n), y] -= 1.0
        dz /= n
        grads: list[np.ndarray] = [None] * len(params)  # type: ignore[list-item]
        for layer in reversed(range(self.n_layers)):
            h_prev = activations[layer]
            grads[2 * layer] = h_prev.T @ dz
            grads[2 * layer + 1] = dz.sum(axis=0)
            if layer > 0:
                dz = (dz @ params[2 * layer].T) * (1.0 - h_prev * h_prev)
        return loss, grads

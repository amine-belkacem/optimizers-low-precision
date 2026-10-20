"""The `digits` dataset, split once and for all.

1797 grayscale 8x8 images of handwritten digits, pixel values 0 to 16.
Source: UCI "Optical Recognition of Handwritten Digits" (Alpaydin and
Kaynak, 1998), licensed CC BY 4.0, shipped with scikit-learn as
`sklearn.datasets.load_digits`, so nothing is downloaded.

The split is fixed (seed SPLIT_SEED) and stratified: 60 % train, 20 %
validation to pick the learning rate, 20 % test touched only once, at the
chosen learning rate. The training seeds change the initial weights and the
order of the mini-batches, never the split.
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from lpopt.train import Split

SPLIT_SEED = 0
PIXEL_MAX = 16.0
N_CLASSES = 10
N_FEATURES = 64


def load_split() -> Split:
    X, y = load_digits(return_X_y=True)
    X = X.astype(np.float64) / PIXEL_MAX  # pixels in [0, 1]
    y = y.astype(np.int64)
    X_train, X_rest, y_train, y_rest = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=SPLIT_SEED
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_rest, y_rest, test_size=0.5, stratify=y_rest, random_state=SPLIT_SEED
    )
    return Split(X_train, y_train, X_val, y_val, X_test, y_test)

from __future__ import annotations

import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


def load_digits_data(seed: int = 0, batch_size: int = 128):
    ds = load_digits()
    X = ds.images.astype(np.float32) / 16.0
    y = ds.target.astype(np.int64)

    # Preserve the 8x8 image representation. Global standardization is not used,
    # because spatial perturbations are easier to interpret on [0, 1] pixels.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=seed
    )

    train = TensorDataset(torch.from_numpy(X_train[:, None]), torch.from_numpy(y_train))
    test = TensorDataset(torch.from_numpy(X_test[:, None]), torch.from_numpy(y_test))
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True),
        DataLoader(test, batch_size=batch_size, shuffle=False),
        train,
        test,
    )


def subset_dataset(dataset: TensorDataset, n: int, seed: int) -> TensorDataset:
    rng = np.random.default_rng(seed)
    y = dataset.tensors[1].numpy()
    idx = []
    classes = np.unique(y)
    per_class = n // len(classes)
    remainder = n - per_class * len(classes)
    for k, c in enumerate(classes):
        take = per_class + (1 if k < remainder else 0)
        candidates = np.flatnonzero(y == c)
        idx.extend(rng.choice(candidates, size=min(take, len(candidates)), replace=False).tolist())
    idx = np.array(idx, dtype=np.int64)
    return TensorDataset(dataset.tensors[0][idx], dataset.tensors[1][idx])


def shifted_images(x: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
    """Zero-padded integer translation for BCHW tensors."""
    out = torch.zeros_like(x)
    h, w = x.shape[-2:]
    ys = max(0, -dy)
    ye = min(h, h - dy) if dy >= 0 else h
    yd = max(0, dy)
    xs = max(0, -dx)
    xe = min(w, w - dx) if dx >= 0 else w
    xd = max(0, dx)
    hh = min(ye - ys, h - yd)
    ww = min(xe - xs, w - xd)
    if hh > 0 and ww > 0:
        out[..., yd:yd+hh, xd:xd+ww] = x[..., ys:ys+hh, xs:xs+ww]
    return out

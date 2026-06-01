from __future__ import annotations

import numpy as np


def softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x - np.nanmax(x) if len(x) else x
    ex = np.exp(x)
    denom = np.nansum(ex)
    if denom <= 0 or np.isnan(denom):
        return np.ones_like(x) / max(1, len(x))
    return ex / denom


def attention_summary(seq, wq: float = 0.046):
    arr = np.asarray([x for x in seq if x == x], dtype=float)
    if len(arr) == 0:
        return {'weighted': np.nan, 'entropy': np.nan, 'recency_bias': np.nan, 'n': 0}
    scores = wq * arr
    weights = softmax(scores)
    weighted = float(np.sum(weights * arr))
    entropy = float(-np.sum(weights * np.log(np.clip(weights, 1e-12, 1.0))))
    if len(arr) == 1:
        recency = 1.0
    else:
        tnorm = np.linspace(0, 1, len(arr))
        recency = float(np.sum(weights * tnorm))
    return {'weighted': weighted, 'entropy': entropy, 'recency_bias': recency, 'n': int(len(arr))}


def train_global_wq(sequences, targets, lr: float = 0.01, epochs: int = 500, clip: float = 1.0) -> float:
    """Tiny numpy SGD for single-query attention. Falls back to 0.046 when data is sparse."""
    clean = []
    for seq, y in zip(sequences, targets):
        arr = np.asarray(seq, dtype=float)
        arr = arr[~np.isnan(arr)]
        if len(arr) >= 2 and y == y:
            clean.append((arr, float(y)))
    if len(clean) < 20:
        return 0.046
    wq = 0.0
    for _ in range(epochs):
        grad = 0.0
        for x, y in clean:
            a = softmax(wq * x)
            pred = float(np.sum(a * x))
            # d pred / d wq for softmax(wq*x) is weighted second moment - weighted mean^2.
            dp = float(np.sum(a * x * x) - pred * pred)
            grad += 2.0 * (pred - y) * dp / len(clean)
        grad = float(np.clip(grad, -clip, clip))
        wq -= lr * grad
    return float(wq)

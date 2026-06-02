from __future__ import annotations

import numpy as np


DEFAULT_WQ = 0.046


def softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax that tolerates NaN values."""
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.asarray([], dtype=float)
    z = x - np.nanmax(x)
    ex = np.exp(z)
    denom = np.nansum(ex)
    if denom <= 0 or not np.isfinite(denom):
        return np.ones_like(x, dtype=float) / max(1, len(x))
    return ex / denom


def attention_summary(seq, wq: float = DEFAULT_WQ) -> dict:
    """
    Single-query attention summary over a numeric history sequence.

    Returns the three Layer-1 attention features:
      - weighted: attention-weighted level
      - entropy: how concentrated/uniform the history weights are
      - recency_bias: weighted position in [0, 1], where >0.5 means newer
        observations dominate the summary
    """
    arr = np.asarray([x for x in seq if x == x and np.isfinite(x)], dtype=float)
    if len(arr) == 0:
        return {'weighted': np.nan, 'entropy': np.nan, 'recency_bias': np.nan, 'n': 0}

    weights = softmax(float(wq) * arr)
    if len(weights) != len(arr) or len(weights) == 0:
        weights = np.ones_like(arr, dtype=float) / len(arr)

    weighted = float(np.sum(weights * arr))
    entropy = float(-np.sum(weights * np.log(np.clip(weights, 1e-12, 1.0))))

    if len(arr) == 1:
        recency = 1.0
    else:
        tnorm = np.linspace(0.0, 1.0, len(arr))
        recency = float(np.sum(weights * tnorm))

    return {'weighted': weighted, 'entropy': entropy, 'recency_bias': recency, 'n': int(len(arr))}


def train_global_wq(
    sequences,
    targets,
    lr: float = 0.01,
    epochs: int = 500,
    clip: float = 1.0,
    fallback: float = DEFAULT_WQ,
) -> float:
    """
    Tiny NumPy SGD for the scalar attention query w_q.

    The task is self-supervised next-step prediction: summarize the previous
    sequence and predict the next observed utilization. If the data is sparse or
    degenerate, return the documented fallback value.
    """
    clean: list[tuple[np.ndarray, float]] = []
    for seq, y in zip(sequences, targets):
        arr = np.asarray(seq, dtype=float)
        arr = arr[np.isfinite(arr)]
        if len(arr) >= 2 and y == y and np.isfinite(y):
            clean.append((arr, float(y)))

    if len(clean) < 20:
        return float(fallback)

    # Keep the scalar attention training lightweight.  The feature builder can
    # produce thousands of next-step pairs; full Python-loop SGD over all of
    # them for 500 epochs is unnecessarily slow and can make Layer 1 appear
    # stuck.  A deterministic subsample is enough for learning this single
    # global scalar.
    max_pairs = 2000
    if len(clean) > max_pairs:
        idx = np.linspace(0, len(clean) - 1, max_pairs).astype(int)
        clean = [clean[i] for i in idx]

    epochs = min(int(epochs), 75)

    wq = 0.0
    for _ in range(epochs):
        grad = 0.0
        for x, y in clean:
            a = softmax(wq * x)
            if len(a) != len(x) or len(a) == 0:
                continue
            pred = float(np.sum(a * x))
            # d pred / d wq for softmax(wq*x) is E[x^2] - E[x]^2.
            dp = float(np.sum(a * x * x) - pred * pred)
            grad += 2.0 * (pred - y) * dp / len(clean)
        grad = float(np.clip(grad, -clip, clip))
        wq -= float(lr) * grad
        if not np.isfinite(wq):
            return float(fallback)

    return float(wq)

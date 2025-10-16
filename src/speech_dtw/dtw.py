from __future__ import annotations

import math
from typing import Callable, Iterable, List, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

Frame = NDArray[np.float32]
SequenceFrames = Sequence[Frame]
DistanceFunction = Callable[[Frame, Frame], float]


def euclidean_distance(a: Frame, b: Frame) -> float:
    return float(np.linalg.norm(a - b))


def dtw_path(
    ref: SequenceFrames,
    seq: SequenceFrames,
    distance: DistanceFunction = euclidean_distance,
) -> Tuple[float, List[Tuple[int, int]]]:
    """Compute DTW distance and alignment path between two sequences."""
    n, m = len(ref), len(seq)
    cost = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
    cost[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d = distance(ref[i - 1], seq[j - 1])
            cost[i, j] = d + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])

    path: List[Tuple[int, int]] = []
    i, j = n, m
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        prev = (cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])
        step = int(np.argmin(prev))
        if step == 0:
            i -= 1
        elif step == 1:
            j -= 1
        else:
            i -= 1
            j -= 1

    while i > 0:
        i -= 1
        path.append((i, 0))
    while j > 0:
        j -= 1
        path.append((0, j))

    path.reverse()
    return float(cost[n, m]), path


def dtw_distance(
    ref: SequenceFrames,
    seq: SequenceFrames,
    distance: DistanceFunction,
) -> float:
    cost, _ = dtw_path(ref, seq, distance=distance)
    return cost


def dtw_template_distance(
    sample: SequenceFrames,
    means: Sequence[Frame],
    cov_inverses: Sequence[NDArray[np.float64]],
    log_dets: Sequence[float],
) -> float:
    """
    Compute DTW distance between a sample and a generalized template
    parameterized by per-frame mean and covariance.
    """
    n, m = len(sample), len(means)
    cost = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
    cost[0, 0] = 0.0

    for i in range(1, n + 1):
        x = sample[i - 1]
        for j in range(1, m + 1):
            diff = x - means[j - 1]
            inv = cov_inverses[j - 1]
            # Mahalanobis distance with log-det penalty for covariance volume
            dist = math.sqrt(float(diff.T @ inv @ diff)) + 0.5 * log_dets[j - 1]
            cost[i, j] = dist + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])

    return float(cost[n, m])

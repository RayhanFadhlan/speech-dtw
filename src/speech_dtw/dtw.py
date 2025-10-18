from __future__ import annotations

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

            min_cost = cost[i, j - 1]

            min_cost = min(min_cost, cost[i - 1, j - 1])

            if i > 1:
                min_cost = min(min_cost, cost[i - 2, j - 1])

            cost[i, j] = d + min_cost

    path: List[Tuple[int, int]] = []
    i, j = n, m
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))


        prev_costs = [cost[i, j - 1]]
        prev_steps = [(i, j - 1)]

        prev_costs.append(cost[i - 1, j - 1])
        prev_steps.append((i - 1, j - 1))

        if i > 1:
            prev_costs.append(cost[i - 2, j - 1])
            prev_steps.append((i - 2, j - 1))

        step_idx = int(np.argmin(prev_costs))
        i, j = prev_steps[step_idx]

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

    D = sample[0].shape[0] if n > 0 else means[0].shape[0]

    log_2pi_D = D * np.log(2 * np.pi)

    for i in range(1, n + 1):
        x = sample[i - 1]
        for j in range(1, m + 1):

            # formula Negative Gaussian Log Likelihood
            # d(x,m_j) = 0.5 * log((2*pi)^D * |C_j|) + 0.5 * (x-m_j)^T * C_j^-1 * (x-m_j)

            diff = x - means[j - 1]
            inv = cov_inverses[j - 1]
            log_det = log_dets[j - 1]

            #0.5 * (x-m_j)^T * C_j^-1 * (x-m_j)
            mahalanobis_part = 0.5 * float(diff.T @ inv @ diff)

            # 0.5 * log((2*pi)^D * |C_j|)
            # =  0.5 * (D * log(2*pi) + log|C_j|)
            # using log(a*b) = log(a) + log(b)
            log_det_part = 0.5 * (log_2pi_D + log_det)

            dist = mahalanobis_part + log_det_part

            min_cost = cost[i, j - 1]

            min_cost = min(min_cost, cost[i - 1, j - 1])

            if i > 1:
                min_cost = min(min_cost, cost[i - 2, j - 1])

            cost[i, j] = dist + min_cost

    return float(cost[n, m])

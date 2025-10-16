from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
from numpy.typing import NDArray

from .dtw import dtw_path, dtw_template_distance, euclidean_distance


FrameVector = NDArray[np.float32]
SequenceArray = NDArray[np.float32]


@dataclass
class GeneralizedTemplate:
    vowel: str
    means: List[FrameVector]
    covariances: List[NDArray[np.float64]]
    cov_inverses: List[NDArray[np.float64]]
    log_dets: List[float]

    def distance(self, sample: Sequence[FrameVector]) -> float:
        return dtw_template_distance(sample, self.means, self.cov_inverses, self.log_dets)


def _ensure_positive_definite(cov: NDArray[np.float64], epsilon: float = 1e-2) -> NDArray[np.float64]:
    return cov + np.eye(cov.shape[0]) * epsilon


def _choose_reference(sequences: Sequence[SequenceArray]) -> int:
    lengths = [len(seq) for seq in sequences]
    sorted_idx = np.argsort(lengths)
    return int(sorted_idx[len(sorted_idx) // 2])


def build_generalized_template(vowel: str, sequences: Sequence[SequenceArray]) -> GeneralizedTemplate:
    if not sequences:
        raise ValueError(f"No sequences provided for vowel '{vowel}'")

    ref_idx = _choose_reference(sequences)
    ref = sequences[ref_idx]
    num_frames = len(ref)
    buckets: List[List[FrameVector]] = [[frame] for frame in ref]

    for idx, seq in enumerate(sequences):
        if idx == ref_idx:
            continue
        _, path = dtw_path(ref, seq, distance=euclidean_distance)
        for ref_pos, seq_pos in path:
            buckets[ref_pos].append(seq[seq_pos])

    means: List[FrameVector] = []
    covariances: List[NDArray[np.float64]] = []
    cov_inverses: List[NDArray[np.float64]] = []
    log_dets: List[float] = []

    for frames in buckets:
        stacked = np.vstack(frames).astype(np.float64)
        mean = stacked.mean(axis=0).astype(np.float32)
        centered = stacked - mean
        if len(frames) > 1:
            cov = (centered.T @ centered) / (len(frames) - 1)
        else:
            cov = np.eye(stacked.shape[1], dtype=np.float64) * 1e-2
        cov = _ensure_positive_definite(cov, epsilon=1e-2)
        cov_inv = np.linalg.inv(cov)
        sign, log_det = np.linalg.slogdet(cov)
        if sign <= 0:
            log_det = float(np.log(np.abs(np.linalg.det(cov)) + 1e-12))
        means.append(mean)
        covariances.append(cov)
        cov_inverses.append(cov_inv)
        log_dets.append(log_det)

    return GeneralizedTemplate(
        vowel=vowel,
        means=means,
        covariances=covariances,
        cov_inverses=cov_inverses,
        log_dets=log_dets,
    )


def build_templates_by_vowel(
    vowel_to_sequences: Dict[str, Sequence[SequenceArray]]
) -> Dict[str, GeneralizedTemplate]:
    return {vowel: build_generalized_template(vowel, seqs) for vowel, seqs in vowel_to_sequences.items()}

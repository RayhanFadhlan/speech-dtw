from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

from .audio_processing import compute_mfcc_features, load_wav
from .dataset import Recording, discover_recordings, filter_recordings
from .recognizer import DTWRecognizer
from .template_builder import build_templates_by_vowel


def compute_features(recordings: Iterable[Recording]) -> Dict[Recording, np.ndarray]:
    feature_map: Dict[Recording, np.ndarray] = {}
    for rec in recordings:
        rate, signal = load_wav(str(rec.path))
        features = compute_mfcc_features(signal, rate)
        feature_map[rec] = features
    return feature_map


def evaluate(
    recognizer: DTWRecognizer,
    records: Iterable[Recording],
    features: Dict[Recording, np.ndarray],
) -> Tuple[int, int, List[Tuple[Recording, str]]]:
    correct = 0
    total = 0
    predictions: List[Tuple[Recording, str]] = []
    for rec in records:
        total += 1
        seq = features[rec]
        result = recognizer.predict(seq)
        predictions.append((rec, result.predicted_vowel))
        if result.predicted_vowel == rec.vowel:
            correct += 1
    return correct, total, predictions


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate DTW-based vowel recognizer.")
    parser.add_argument("--data-root", type=Path, default=Path("data"), help="Path to data directory.")
    parser.add_argument(
        "--train-max-index",
        type=int,
        default=3,
        help="Maximum recording index to include when building templates.",
    )
    parser.add_argument(
        "--closed-min-index",
        type=int,
        default=4,
        help="Minimum recording index to include in closed-set evaluation.",
    )
    args = parser.parse_args(argv)

    train_dir = args.data_root / "train"
    test_dir = args.data_root / "test"
    train_records = discover_recordings(train_dir, group="train")
    test_records = discover_recordings(test_dir, group="test")

    template_records = filter_recordings(train_records, max_index=args.train_max_index)
    closed_records = filter_recordings(train_records, min_index=args.closed_min_index)

    print(f"Templates per vowel: {len(template_records)} recordings")
    print(f"Closed-set evaluation recordings: {len(closed_records)}")
    print(f"Open-set evaluation recordings: {len(test_records)}")

    all_records = list({*template_records, *closed_records, *test_records})
    features = compute_features(all_records)

    vowel_sequences: Dict[str, List[np.ndarray]] = defaultdict(list)
    for rec in template_records:
        vowel_sequences[rec.vowel].append(features[rec])

    templates = build_templates_by_vowel(vowel_sequences)
    recognizer = DTWRecognizer(templates)

    closed_correct, closed_total, _ = evaluate(recognizer, closed_records, features)
    open_correct, open_total, _ = evaluate(recognizer, test_records, features)

    closed_acc = closed_correct / closed_total if closed_total else 0.0
    open_acc = open_correct / open_total if open_total else 0.0
    avg_acc = (closed_acc + open_acc) / 2 if closed_total and open_total else 0.0

    print(f"Closed-set accuracy: {closed_correct}/{closed_total} = {closed_acc:.3f}")
    print(f"Open-set accuracy:   {open_correct}/{open_total} = {open_acc:.3f}")
    print(f"Average accuracy:    {avg_acc:.3f}")

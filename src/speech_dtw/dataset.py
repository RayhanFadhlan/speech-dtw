from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

VOWELS = ("a", "i", "u", "e", "o")


@dataclass(frozen=True)
class Recording:
    path: Path
    speaker: str
    vowel: str
    index: int | None
    group: str  # e.g., "train" or "test"


_VOWEL_PATTERN = re.compile(r"\b([aiueo])\d*\b", re.IGNORECASE)


def _extract_vowel(stem: str) -> str:
    lowered = stem.lower()
    match = _VOWEL_PATTERN.search(lowered)
    if match:
        return match.group(1)
    tokens = re.split(r"[\s_\-]+", lowered)
    for token in tokens:
        for char in token:
            if char in VOWELS:
                return char
    raise ValueError(f"Unable to determine vowel from filename '{stem}'")


def _extract_index(stem: str) -> int | None:
    match = re.search(r"(\d+)", stem)
    if match:
        return int(match.group(1))
    return None


def discover_recordings(root: Path, group: str) -> List[Recording]:
    recordings: List[Recording] = []
    for path in sorted(root.rglob("*.wav")):
        stem = path.stem
        try:
            relative = path.relative_to(root)
            parts = relative.parts
        except ValueError:
            parts = ()

        speaker = parts[0] if len(parts) >= 2 else stem.split()[0]
        if len(parts) >= 2:
            vowel = parts[1].lower()
        else:
            vowel = _extract_vowel(stem)
        idx = _extract_index(stem)
        recordings.append(Recording(path=path, speaker=speaker, vowel=vowel, index=idx, group=group))
    return recordings


def filter_recordings(
    recordings: Iterable[Recording],
    *,
    vowels: Iterable[str] | None = None,
    min_index: int | None = None,
    max_index: int | None = None,
) -> List[Recording]:
    vowels_lower = tuple(v.lower() for v in vowels) if vowels else VOWELS
    results: List[Recording] = []
    for rec in recordings:
        if rec.vowel not in vowels_lower:
            continue
        if min_index is not None and (rec.index is None or rec.index < min_index):
            continue
        if max_index is not None and (rec.index is None or rec.index > max_index):
            continue
        results.append(rec)
    return results

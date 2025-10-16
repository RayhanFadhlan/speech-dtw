from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np

from .template_builder import GeneralizedTemplate


@dataclass
class RecognitionResult:
    predicted_vowel: str
    distances: Dict[str, float]


class DTWRecognizer:
    def __init__(self, templates: Dict[str, GeneralizedTemplate]):
        self.templates = templates

    def predict(self, sequence: Sequence[np.ndarray]) -> RecognitionResult:
        distances = {vowel: template.distance(sequence) for vowel, template in self.templates.items()}
        predicted = min(distances, key=distances.get)
        return RecognitionResult(predicted_vowel=predicted, distances=distances)

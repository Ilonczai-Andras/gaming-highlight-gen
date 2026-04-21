"""Moment detectors for the gaming highlight pipeline."""

from gaming_highlight_gen.detectors.base import (
    BaseDetector,
    DetectorInputError,
    DetectorProcessingError,
)
from gaming_highlight_gen.detectors.audio_detector import AudioDetector
from gaming_highlight_gen.detectors.visual_detector import VisualDetector
from gaming_highlight_gen.detectors.composite_scorer import CompositeScorer
from gaming_highlight_gen.detectors.ml_detector import MLDetector

__all__ = [
    "BaseDetector",
    "DetectorInputError",
    "DetectorProcessingError",
    "AudioDetector",
    "VisualDetector",
    "CompositeScorer",
    "MLDetector",
]

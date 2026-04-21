"""Abstract base detector and shared exceptions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from gaming_highlight_gen.models.moment import DetectionResult


class DetectorInputError(ValueError):
    """Raised when the detector receives invalid or missing input."""


class DetectorProcessingError(RuntimeError):
    """Raised when the detector fails during analysis."""


class BaseDetector(ABC):
    """Abstract base class for all moment detectors.

    Subclasses must implement :meth:`detect`.
    """

    @property
    @abstractmethod
    def detector_type(self) -> str:
        """Unique identifier string for this detector type."""

    @abstractmethod
    def detect(self, video_path: Path) -> DetectionResult:
        """Analyse *video_path* and return a :class:`DetectionResult`.

        Args:
            video_path: Path to the video file to analyse.

        Returns:
            :class:`DetectionResult` containing all discovered signals.

        Raises:
            DetectorInputError: If *video_path* does not exist or is not a
                readable video file.
            DetectorProcessingError: If an internal error occurs during
                analysis.
        """

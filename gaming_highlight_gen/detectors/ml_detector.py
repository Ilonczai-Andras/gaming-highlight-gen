"""ML detector stub — placeholder for Sprint 5."""

from __future__ import annotations

import time
from pathlib import Path

import structlog

from gaming_highlight_gen.detectors.base import BaseDetector, DetectorInputError
from gaming_highlight_gen.models.moment import DetectionResult

logger: structlog.BoundLogger = structlog.get_logger(__name__)


class MLDetector(BaseDetector):
    """Stub ML detector that always returns an empty result.

    This detector is reserved for Sprint 5 when a trained model will be
    integrated.  It logs a ``skipped`` event so operators are aware it is
    not yet active.
    """

    @property
    def detector_type(self) -> str:
        return "ml"

    def detect(self, video_path: Path) -> DetectionResult:
        """Return an empty :class:`DetectionResult` without doing any work.

        Args:
            video_path: Path to the video file (existence is still validated).

        Returns:
            Empty :class:`DetectionResult` with ``processing_time_sec=0.0``.

        Raises:
            DetectorInputError: If *video_path* does not exist.
        """
        if not video_path.exists():
            raise DetectorInputError(f"Video file not found: {video_path}")

        logger.info(
            "ml_detector.skipped",
            reason="not_yet_implemented",
            video=str(video_path),
        )
        return DetectionResult(
            detector_type=self.detector_type,
            signals=[],
            processing_time_sec=0.0,
            video_duration_sec=0.0,
            config_snapshot={"status": "stub"},
        )

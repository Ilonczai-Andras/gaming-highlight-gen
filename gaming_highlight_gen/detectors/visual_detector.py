"""Visual frame-difference detector for gameplay moment detection."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import structlog

from gaming_highlight_gen.config.game_config import VisualDetectorConfig
from gaming_highlight_gen.detectors.base import (
    BaseDetector,
    DetectorInputError,
    DetectorProcessingError,
)
from gaming_highlight_gen.models.moment import DetectionResult, DetectorSignal

logger: structlog.BoundLogger = structlog.get_logger(__name__)


class VisualDetector(BaseDetector):
    """Detect gameplay moments from sudden visual changes between frames.

    The detector:
    1. Opens the video with ``cv2.VideoCapture``.
    2. Samples every ``sample_every_n_frames``-th frame.
    3. Converts frames to greyscale and computes normalised absolute diff.
    4. Applies a rolling-average smoothing window.
    5. Finds peaks above an adaptive threshold.
    6. Tags peaks above ``hard_cut_threshold`` as ``"scene_cut"``.
    7. Optionally removes scene-cut signals (``filter_scene_cuts=True``).
    """

    def __init__(self, config: VisualDetectorConfig) -> None:
        self._config = config

    @property
    def detector_type(self) -> str:
        return "visual"

    def detect(self, video_path: Path) -> DetectionResult:
        """Analyse *video_path* for visual motion spikes.

        Args:
            video_path: Path to the source video file.

        Returns:
            :class:`DetectionResult` with detected :class:`DetectorSignal` items.

        Raises:
            DetectorInputError: If the file does not exist or cannot be opened.
            DetectorProcessingError: If frame analysis fails.
        """
        if not video_path.exists():
            raise DetectorInputError(f"Video file not found: {video_path}")

        if not self._config.enabled:
            logger.info("visual_detector.skipped", reason="disabled_in_config")
            return DetectionResult(
                detector_type=self.detector_type,
                signals=[],
                processing_time_sec=0.0,
                video_duration_sec=0.0,
                config_snapshot=self._config.model_dump(),
            )

        t_start = time.perf_counter()

        try:
            duration, signals = self._analyze_frames(video_path)
        except DetectorInputError:
            raise
        except Exception as exc:
            raise DetectorProcessingError(
                f"Visual detection failed for {video_path}: {exc}"
            ) from exc

        processing_time = time.perf_counter() - t_start
        realtime_ratio = processing_time / duration if duration > 0 else 0.0

        logger.info(
            "visual_detector.complete",
            video=str(video_path),
            signals_found=len(signals),
            processing_time_sec=round(processing_time, 3),
            realtime_ratio=round(realtime_ratio, 3),
        )

        return DetectionResult(
            detector_type=self.detector_type,
            signals=signals,
            processing_time_sec=processing_time,
            video_duration_sec=duration,
            config_snapshot=self._config.model_dump(),
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _analyze_frames(self, video_path: Path) -> tuple[float, list[DetectorSignal]]:
        import cv2  # deferred import — optional dependency
        from scipy.signal import find_peaks

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise DetectorInputError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0.0

        frame_diffs: list[float] = []
        timestamps: list[float] = []
        prev_gray: np.ndarray | None = None
        frame_idx = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % self._config.sample_every_n_frames == 0:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
                    if prev_gray is not None:
                        diff = float(np.mean(np.abs(gray - prev_gray)))
                        frame_diffs.append(diff)
                        timestamps.append(frame_idx / fps)
                    prev_gray = gray
                frame_idx += 1
        finally:
            cap.release()

        if len(frame_diffs) < 2:
            return duration, []

        scores = np.array(frame_diffs, dtype=np.float64)

        # Rolling average smoothing
        smooth_frames = max(1, int(self._config.smooth_window_sec * fps / self._config.sample_every_n_frames))
        if smooth_frames > 1 and len(scores) > smooth_frames:
            kernel = np.ones(smooth_frames) / smooth_frames
            smoothed = np.convolve(scores, kernel, mode="same")
        else:
            smoothed = scores

        # Adaptive threshold
        mean_score = float(np.mean(smoothed))
        std_score = float(np.std(smoothed))

        if std_score == 0:
            return duration, []

        threshold = mean_score + self._config.threshold_sigma * std_score

        # Minimum peak distance in samples
        samples_per_sec = fps / self._config.sample_every_n_frames
        min_distance_samples = max(1, int(self._config.min_peak_distance_sec * samples_per_sec))

        peak_indices, _ = find_peaks(smoothed, height=threshold, distance=min_distance_samples)

        signals: list[DetectorSignal] = []
        for idx in peak_indices:
            ts = timestamps[idx]
            raw = float(smoothed[idx])
            is_hard_cut = raw >= self._config.hard_cut_threshold
            event = "scene_cut" if is_hard_cut else "motion_spike"

            if is_hard_cut and self._config.filter_scene_cuts:
                continue

            sigmas_above = (raw - mean_score) / std_score
            confidence = float(
                np.clip(sigmas_above / (self._config.threshold_sigma * 3), 0.0, 1.0)
            )
            signals.append(
                DetectorSignal(
                    timestamp_sec=ts,
                    confidence=confidence,
                    detector_type=self.detector_type,
                    event_type=event,
                    raw_value=raw,
                )
            )

        return duration, signals

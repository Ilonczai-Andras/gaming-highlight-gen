"""Audio-based moment detector using RMS energy spike detection."""

from __future__ import annotations

import logging
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import structlog

from gaming_highlight_gen.config.game_config import AudioDetectorConfig
from gaming_highlight_gen.detectors.base import (
    BaseDetector,
    DetectorInputError,
    DetectorProcessingError,
)
from gaming_highlight_gen.models.moment import DetectionResult, DetectorSignal

if TYPE_CHECKING:
    pass

logger: structlog.BoundLogger = structlog.get_logger(__name__)


class AudioDetector(BaseDetector):
    """Detect gameplay moments from audio spikes in RMS energy.

    The detector:
    1. Extracts audio from the video to a temporary WAV file via FFmpeg.
    2. Loads the WAV with ``librosa`` and computes frame-level RMS energy.
    3. Applies a Savitzky-Golay smoothing filter.
    4. Finds peaks that exceed ``mean + spike_threshold_sigma * std``.
    5. Enforces a ``min_peak_distance_sec`` cooldown between peaks.
    6. Caps output to ``max_signals_per_minute`` density.
    """

    def __init__(
        self,
        config: AudioDetectorConfig,
        ffmpeg_wrapper: "gaming_highlight_gen.core.ffmpeg_wrapper.FFmpegWrapper | None" = None,
    ) -> None:
        from gaming_highlight_gen.core.ffmpeg_wrapper import FFmpegWrapper  # local import avoids cycles
        from gaming_highlight_gen.config.global_config import GlobalConfig

        self._config = config
        self._ffmpeg = ffmpeg_wrapper or FFmpegWrapper(GlobalConfig())

    @property
    def detector_type(self) -> str:
        return "audio"

    def detect(self, video_path: Path) -> DetectionResult:
        """Analyse *video_path* for audio spikes.

        Args:
            video_path: Path to the source video file.

        Returns:
            :class:`DetectionResult` with detected :class:`DetectorSignal` items.

        Raises:
            DetectorInputError: If the file does not exist.
            DetectorProcessingError: If audio extraction or analysis fails.
        """
        if not video_path.exists():
            raise DetectorInputError(f"Video file not found: {video_path}")

        if not self._config.enabled:
            logger.info("audio_detector.skipped", reason="disabled_in_config")
            duration = self._get_duration(video_path)
            return DetectionResult(
                detector_type=self.detector_type,
                signals=[],
                processing_time_sec=0.0,
                video_duration_sec=duration,
                config_snapshot=self._config.model_dump(),
            )

        t_start = time.perf_counter()

        try:
            duration = self._get_duration(video_path)
            signals = self._analyze_audio(video_path, duration)
        except DetectorInputError:
            raise
        except Exception as exc:
            raise DetectorProcessingError(
                f"Audio detection failed for {video_path}: {exc}"
            ) from exc

        processing_time = time.perf_counter() - t_start
        realtime_ratio = processing_time / duration if duration > 0 else 0.0

        logger.info(
            "audio_detector.complete",
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

    def _get_duration(self, video_path: Path) -> float:
        try:
            return self._ffmpeg.get_duration(video_path)
        except Exception as exc:
            raise DetectorInputError(f"Cannot read video duration: {exc}") from exc

    def _analyze_audio(self, video_path: Path, duration: float) -> list[DetectorSignal]:
        import librosa  # deferred import — optional dependency
        from scipy.signal import find_peaks, savgol_filter

        wav_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        wav_path = Path(wav_file.name)
        wav_file.close()

        try:
            self._ffmpeg.extract_audio(video_path, wav_path)
            y, sr = librosa.load(str(wav_path), sr=None, mono=True)
        finally:
            if wav_path.exists():
                wav_path.unlink()

        if len(y) == 0:
            return []

        # Compute RMS energy per frame
        hop_length = 512
        rms: np.ndarray = librosa.feature.rms(y=y, hop_length=hop_length)[0]
        times: np.ndarray = librosa.frames_to_time(
            np.arange(len(rms)), sr=sr, hop_length=hop_length
        )

        # Smooth with Savitzky-Golay (window ≈ 0.5 s, must be odd)
        window_frames = max(5, int(0.5 * sr / hop_length) | 1)  # ensure odd
        if len(rms) > window_frames:
            smoothed = savgol_filter(rms.astype(np.float64), window_frames, polyorder=3)
            smoothed = np.clip(smoothed, 0.0, None)
        else:
            smoothed = rms.astype(np.float64)

        # Adaptive threshold
        mean_rms = float(np.mean(smoothed))
        std_rms = float(np.std(smoothed))
        threshold = mean_rms + self._config.spike_threshold_sigma * std_rms

        if std_rms == 0:
            return []

        # Find peaks above threshold
        frames_per_sec = sr / hop_length
        min_distance_frames = max(1, int(self._config.min_peak_distance_sec * frames_per_sec))
        peak_indices, properties = find_peaks(
            smoothed,
            height=threshold,
            distance=min_distance_frames,
        )

        if len(peak_indices) == 0:
            return []

        # Build signals
        signals: list[DetectorSignal] = []
        for idx in peak_indices:
            ts = float(times[idx])
            rms_value = float(smoothed[idx])
            # Normalise confidence: how many sigmas above threshold?
            sigmas_above = (rms_value - mean_rms) / std_rms
            confidence = float(np.clip(sigmas_above / (self._config.spike_threshold_sigma * 3), 0.0, 1.0))
            signals.append(
                DetectorSignal(
                    timestamp_sec=ts,
                    confidence=confidence,
                    detector_type=self.detector_type,
                    event_type="audio_spike",
                    raw_value=rms_value,
                )
            )

        # Cap density
        if duration > 0:
            max_allowed = int(self._config.max_signals_per_minute * (duration / 60.0))
            max_allowed = max(1, max_allowed)
            if len(signals) > max_allowed:
                signals = sorted(signals, key=lambda s: s.confidence, reverse=True)[:max_allowed]
                signals = sorted(signals, key=lambda s: s.timestamp_sec)

        return signals

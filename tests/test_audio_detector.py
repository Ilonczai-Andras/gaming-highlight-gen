"""Tests for the AudioDetector."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gaming_highlight_gen.config.game_config import AudioDetectorConfig
from gaming_highlight_gen.detectors.audio_detector import AudioDetector
from gaming_highlight_gen.detectors.base import DetectorInputError, DetectorProcessingError
from gaming_highlight_gen.models.moment import DetectionResult, DetectorSignal
from tests.fixtures.generate_fixtures import generate_sine_audio_with_spikes


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_detector(
    enabled: bool = True,
    spike_threshold_sigma: float = 2.0,
    min_peak_distance_sec: float = 1.5,
    max_signals_per_minute: int = 60,
    ffmpeg_wrapper: MagicMock | None = None,
) -> AudioDetector:
    cfg = AudioDetectorConfig(
        enabled=enabled,
        spike_threshold_sigma=spike_threshold_sigma,
        min_peak_distance_sec=min_peak_distance_sec,
        max_signals_per_minute=max_signals_per_minute,
    )
    mock_ffmpeg = ffmpeg_wrapper or MagicMock()
    mock_ffmpeg.get_duration.return_value = 10.0
    return AudioDetector(config=cfg, ffmpeg_wrapper=mock_ffmpeg)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_detect_raises_for_missing_file() -> None:
    detector = _make_detector()
    with pytest.raises(DetectorInputError, match="not found"):
        detector.detect(Path("/nonexistent/video.mp4"))


# ---------------------------------------------------------------------------
# Disabled detector
# ---------------------------------------------------------------------------


def test_detect_returns_empty_when_disabled(tmp_path: Path) -> None:
    # Create a dummy file so existence check passes
    video = tmp_path / "video.mp4"
    video.write_bytes(b"dummy")

    mock_ffmpeg = MagicMock()
    mock_ffmpeg.get_duration.return_value = 10.0
    detector = _make_detector(enabled=False, ffmpeg_wrapper=mock_ffmpeg)

    result = detector.detect(video)

    assert isinstance(result, DetectionResult)
    assert result.signals == []
    assert result.processing_time_sec == 0.0
    assert result.detector_type == "audio"


# ---------------------------------------------------------------------------
# Real audio analysis (uses synthetic WAV)
# ---------------------------------------------------------------------------


def test_detect_finds_spikes_in_synthetic_audio(tmp_path: Path) -> None:
    """AudioDetector should find the 3 amplitude spikes in the test WAV."""
    wav_path = tmp_path / "spiky.wav"
    generate_sine_audio_with_spikes(wav_path, duration_sec=10.0, spike_times=[2.0, 5.0, 8.0])

    video = tmp_path / "video.mp4"
    video.write_bytes(b"dummy")

    mock_ffmpeg = MagicMock()
    mock_ffmpeg.get_duration.return_value = 10.0

    # Mock extract_audio to copy the prepared WAV to the requested temp path
    def fake_extract_audio(src: Path, dst: Path) -> None:
        import shutil
        shutil.copy(wav_path, dst)

    mock_ffmpeg.extract_audio.side_effect = fake_extract_audio

    detector = AudioDetector(
        config=AudioDetectorConfig(
            enabled=True,
            spike_threshold_sigma=1.5,
            min_peak_distance_sec=0.5,
            max_signals_per_minute=60,
        ),
        ffmpeg_wrapper=mock_ffmpeg,
    )

    result = detector.detect(video)

    assert isinstance(result, DetectionResult)
    assert result.detector_type == "audio"
    assert len(result.signals) >= 1  # at least one spike should be detected
    for sig in result.signals:
        assert isinstance(sig, DetectorSignal)
        assert sig.detector_type == "audio"
        assert 0.0 <= sig.confidence <= 1.0
        assert sig.timestamp_sec >= 0.0


def test_detect_returns_empty_for_silent_audio(tmp_path: Path) -> None:
    """No spikes in a silent (all-zero) WAV — should return empty signals."""
    import struct, wave

    wav_path = tmp_path / "silent.wav"
    n = 44100 * 5
    with wave.open(str(wav_path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(44100)
        wf.writeframes(struct.pack(f"<{n}h", *([0] * n)))

    video = tmp_path / "video.mp4"
    video.write_bytes(b"dummy")

    mock_ffmpeg = MagicMock()
    mock_ffmpeg.get_duration.return_value = 5.0

    def fake_extract_audio(src: Path, dst: Path) -> None:
        import shutil
        shutil.copy(wav_path, dst)

    mock_ffmpeg.extract_audio.side_effect = fake_extract_audio

    detector = AudioDetector(
        config=AudioDetectorConfig(enabled=True),
        ffmpeg_wrapper=mock_ffmpeg,
    )

    result = detector.detect(video)
    assert result.signals == []


# ---------------------------------------------------------------------------
# Density cap
# ---------------------------------------------------------------------------


def test_density_cap_limits_signals(tmp_path: Path) -> None:
    """max_signals_per_minute should cap the number of returned signals."""
    # Spikes at every half-second → 20 spikes in 10 s video
    spike_times = [i * 0.5 for i in range(20)]
    wav_path = tmp_path / "many_spikes.wav"
    generate_sine_audio_with_spikes(wav_path, duration_sec=10.0, spike_times=spike_times)

    video = tmp_path / "video.mp4"
    video.write_bytes(b"dummy")

    mock_ffmpeg = MagicMock()
    mock_ffmpeg.get_duration.return_value = 10.0

    def fake_extract_audio(src: Path, dst: Path) -> None:
        import shutil
        shutil.copy(wav_path, dst)

    mock_ffmpeg.extract_audio.side_effect = fake_extract_audio

    detector = AudioDetector(
        config=AudioDetectorConfig(
            enabled=True,
            spike_threshold_sigma=1.0,
            min_peak_distance_sec=0.1,
            max_signals_per_minute=6,  # cap at ~1/10s video → max 1
        ),
        ffmpeg_wrapper=mock_ffmpeg,
    )

    result = detector.detect(video)
    max_allowed = max(1, int(6 * (10.0 / 60.0)))
    assert len(result.signals) <= max_allowed


# ---------------------------------------------------------------------------
# FFmpeg extraction failure
# ---------------------------------------------------------------------------


def test_detect_raises_processing_error_on_ffmpeg_failure(tmp_path: Path) -> None:
    from gaming_highlight_gen.core.ffmpeg_wrapper import FFmpegError

    video = tmp_path / "video.mp4"
    video.write_bytes(b"dummy")

    mock_ffmpeg = MagicMock()
    mock_ffmpeg.get_duration.return_value = 10.0
    mock_ffmpeg.extract_audio.side_effect = FFmpegError("fail", returncode=1, stderr="err")

    detector = AudioDetector(
        config=AudioDetectorConfig(enabled=True),
        ffmpeg_wrapper=mock_ffmpeg,
    )

    with pytest.raises(DetectorProcessingError):
        detector.detect(video)

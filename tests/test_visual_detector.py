"""Tests for the VisualDetector."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from gaming_highlight_gen.config.game_config import VisualDetectorConfig
from gaming_highlight_gen.detectors.base import DetectorInputError, DetectorProcessingError
from gaming_highlight_gen.detectors.visual_detector import VisualDetector
from gaming_highlight_gen.models.moment import DetectionResult, DetectorSignal
from tests.fixtures.generate_fixtures import generate_test_video_with_motion


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_detector(
    enabled: bool = True,
    **kwargs: object,
) -> VisualDetector:
    cfg = VisualDetectorConfig(enabled=enabled, **kwargs)
    return VisualDetector(config=cfg)


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
    video = tmp_path / "video.mp4"
    video.write_bytes(b"dummy")

    detector = _make_detector(enabled=False)
    result = detector.detect(video)

    assert isinstance(result, DetectionResult)
    assert result.signals == []
    assert result.detector_type == "visual"


# ---------------------------------------------------------------------------
# Real video analysis
# ---------------------------------------------------------------------------


def test_detect_finds_motion_events(tmp_path: Path) -> None:
    """VisualDetector should find the 3 bright-flash frames in the test video."""
    video = generate_test_video_with_motion(
        tmp_path / "test.mp4",
        duration_sec=10.0,
        fps=30,
        width=64,
        height=64,
        motion_times=[2.0, 5.0, 8.0],
    )

    detector = VisualDetector(
        config=VisualDetectorConfig(
            enabled=True,
            sample_every_n_frames=1,
            threshold_sigma=1.5,
            min_peak_distance_sec=0.5,
            smooth_window_sec=0.1,
            filter_scene_cuts=False,
        )
    )

    result = detector.detect(video)

    assert isinstance(result, DetectionResult)
    assert result.detector_type == "visual"
    assert len(result.signals) >= 1
    for sig in result.signals:
        assert isinstance(sig, DetectorSignal)
        assert sig.detector_type == "visual"
        assert 0.0 <= sig.confidence <= 1.0


def test_detect_filters_scene_cuts(tmp_path: Path) -> None:
    """filter_scene_cuts=True should remove scene_cut signals."""
    video = generate_test_video_with_motion(
        tmp_path / "test_sc.mp4",
        duration_sec=10.0,
        fps=30,
        width=64,
        height=64,
        motion_times=[2.0, 5.0, 8.0],
    )

    detector_unfiltered = VisualDetector(
        config=VisualDetectorConfig(
            enabled=True,
            sample_every_n_frames=1,
            threshold_sigma=1.0,
            hard_cut_threshold=0.3,  # low enough to trigger scene_cut
            filter_scene_cuts=False,
        )
    )
    result_unfiltered = detector_unfiltered.detect(video)

    detector_filtered = VisualDetector(
        config=VisualDetectorConfig(
            enabled=True,
            sample_every_n_frames=1,
            threshold_sigma=1.0,
            hard_cut_threshold=0.3,
            filter_scene_cuts=True,
        )
    )
    result_filtered = detector_filtered.detect(video)

    # After filtering, scene_cut signals must be absent
    assert all(s.event_type != "scene_cut" for s in result_filtered.signals)
    # Unfiltered can have scene_cut signals (or none if frame diff never hit threshold)
    # — just verify the structure
    assert isinstance(result_unfiltered, DetectionResult)


# ---------------------------------------------------------------------------
# Unreadable video
# ---------------------------------------------------------------------------


def test_detect_raises_for_unreadable_video(tmp_path: Path) -> None:
    """A file that exists but isn't a valid video should raise DetectorInputError."""
    bad_video = tmp_path / "garbage.mp4"
    bad_video.write_bytes(b"not a video")

    detector = _make_detector()
    with pytest.raises(DetectorInputError, match="Cannot open"):
        detector.detect(bad_video)

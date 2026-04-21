"""Tests for the CompositeScorer."""

from __future__ import annotations

from pathlib import Path

import pytest

from gaming_highlight_gen.config.game_config import CompositeConfig
from gaming_highlight_gen.detectors.composite_scorer import CompositeScorer
from gaming_highlight_gen.models.moment import DetectionResult, DetectorSignal, Moment


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SOURCE = Path("game.mp4")


def _result(
    detector_type: str,
    signals: list[DetectorSignal],
    duration: float = 30.0,
) -> DetectionResult:
    return DetectionResult(
        detector_type=detector_type,
        signals=signals,
        processing_time_sec=0.1,
        video_duration_sec=duration,
    )


def _signal(
    ts: float,
    confidence: float = 0.8,
    detector_type: str = "audio",
    event_type: str = "audio_spike",
) -> DetectorSignal:
    return DetectorSignal(
        timestamp_sec=ts,
        confidence=confidence,
        detector_type=detector_type,
        event_type=event_type,
    )


def _scorer(**kwargs: object) -> CompositeScorer:
    return CompositeScorer(config=CompositeConfig(**kwargs))


# ---------------------------------------------------------------------------
# No signals
# ---------------------------------------------------------------------------


def test_score_returns_empty_for_no_signals() -> None:
    scorer = _scorer()
    result = _result("audio", signals=[])
    moments = scorer.score([result], _SOURCE)
    assert moments == []


def test_score_returns_empty_for_no_results() -> None:
    scorer = _scorer()
    moments = scorer.score([], _SOURCE)
    assert moments == []


# ---------------------------------------------------------------------------
# Basic scoring
# ---------------------------------------------------------------------------


def test_score_produces_moment_from_single_signal() -> None:
    scorer = _scorer(min_moment_score=0.0)
    result = _result("audio", [_signal(5.0, confidence=0.9)])
    moments = scorer.score([result], _SOURCE)

    assert len(moments) == 1
    m = moments[0]
    assert isinstance(m, Moment)
    assert m.source_file == _SOURCE
    assert m.score > 0.0
    assert m.start_sec == pytest.approx(5.0, abs=0.01)


def test_score_clusters_close_signals() -> None:
    """Signals within cluster_window_sec should merge into one Moment."""
    scorer = _scorer(cluster_window_sec=2.0, min_moment_score=0.0)
    result = _result(
        "audio",
        [
            _signal(5.0, 0.8),
            _signal(5.5, 0.9),  # within 2 s of 5.0
            _signal(10.0, 0.7),  # far apart
        ],
    )
    moments = scorer.score([result], _SOURCE)

    assert len(moments) == 2


def test_score_filters_below_min_score() -> None:
    scorer = _scorer(min_moment_score=0.99, cluster_window_sec=1.0)
    result = _result("audio", [_signal(5.0, confidence=0.1)])
    moments = scorer.score([result], _SOURCE)
    assert moments == []


# ---------------------------------------------------------------------------
# Max moments cap
# ---------------------------------------------------------------------------


def test_score_caps_to_max_moments_per_video() -> None:
    scorer = _scorer(
        max_moments_per_video=3,
        cluster_window_sec=0.1,
        min_moment_score=0.0,
    )
    # 10 signals well separated in time
    sigs = [_signal(float(i * 5), confidence=0.8) for i in range(10)]
    result = _result("audio", sigs)
    moments = scorer.score([result], _SOURCE)

    assert len(moments) <= 3


# ---------------------------------------------------------------------------
# Multi-detector scoring
# ---------------------------------------------------------------------------


def test_score_merges_audio_and_visual_signals() -> None:
    scorer = _scorer(cluster_window_sec=2.0, min_moment_score=0.0)
    audio_result = _result("audio", [_signal(5.0, 0.8, "audio")])
    visual_result = _result("visual", [_signal(5.2, 0.7, "visual", "motion_spike")])

    moments = scorer.score([audio_result, visual_result], _SOURCE)

    assert len(moments) == 1
    m = moments[0]
    assert "audio" in m.detector_breakdown
    assert "visual" in m.detector_breakdown


def test_score_sorts_by_score_descending() -> None:
    scorer = _scorer(cluster_window_sec=0.5, min_moment_score=0.0)
    low_sig = _signal(1.0, confidence=0.2)
    high_sig = _signal(10.0, confidence=0.95)
    result = _result("audio", [low_sig, high_sig])

    moments = scorer.score([result], _SOURCE)
    assert len(moments) == 2
    assert moments[0].score >= moments[1].score

"""Tests for the adapters package."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from gaming_highlight_gen.adapters import get_adapter
from gaming_highlight_gen.adapters.default_adapter import DefaultAdapter
from gaming_highlight_gen.adapters.valorant_adapter import ValorantAdapter
from gaming_highlight_gen.config.game_config import AudioDetectorConfig, GameConfig
from gaming_highlight_gen.models.moment import DetectorSignal


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CONFIG_DIR = Path(__file__).parent.parent / "game_configs"


def _game_config(game_id: str = "default") -> GameConfig:
    from gaming_highlight_gen.config.game_config import load_game_config
    return load_game_config(game_id, _CONFIG_DIR)


# ---------------------------------------------------------------------------
# get_adapter registry
# ---------------------------------------------------------------------------


def test_get_adapter_returns_valorant_adapter() -> None:
    cfg = _game_config("valorant")
    adapter = get_adapter(cfg)
    assert isinstance(adapter, ValorantAdapter)


def test_get_adapter_returns_default_adapter_for_r6() -> None:
    cfg = _game_config("r6")
    adapter = get_adapter(cfg)
    assert isinstance(adapter, DefaultAdapter)


def test_get_adapter_returns_default_for_unknown_game_id() -> None:
    cfg = GameConfig(game_id="unknown_game", display_name="Unknown")
    adapter = get_adapter(cfg)
    assert isinstance(adapter, DefaultAdapter)


# ---------------------------------------------------------------------------
# DefaultAdapter
# ---------------------------------------------------------------------------


def test_default_adapter_game_id() -> None:
    adapter = DefaultAdapter(_game_config("default"))
    assert adapter.game_id == "default"


def test_default_adapter_get_detectors_includes_audio_and_visual() -> None:
    from gaming_highlight_gen.detectors.audio_detector import AudioDetector
    from gaming_highlight_gen.detectors.visual_detector import VisualDetector

    adapter = DefaultAdapter(_game_config("default"))
    detectors = adapter.get_detectors()
    types = {d.detector_type for d in detectors}
    assert "audio" in types
    assert "visual" in types


def test_default_adapter_post_process_is_passthrough() -> None:
    adapter = DefaultAdapter(_game_config("default"))
    sigs = [
        DetectorSignal(timestamp_sec=1.0, confidence=0.9, detector_type="audio"),
        DetectorSignal(timestamp_sec=2.0, confidence=0.8, detector_type="visual"),
    ]
    result = adapter.post_process_signals(sigs)
    assert result == sigs


# ---------------------------------------------------------------------------
# ValorantAdapter
# ---------------------------------------------------------------------------


def test_valorant_adapter_game_id() -> None:
    adapter = ValorantAdapter(_game_config("valorant"))
    assert adapter.game_id == "valorant"


def test_valorant_adapter_removes_scene_cut_signals() -> None:
    adapter = ValorantAdapter(_game_config("valorant"))
    sigs = [
        DetectorSignal(timestamp_sec=1.0, confidence=0.9, detector_type="visual", event_type="scene_cut"),
        DetectorSignal(timestamp_sec=2.0, confidence=0.8, detector_type="audio", event_type="audio_spike"),
        DetectorSignal(timestamp_sec=3.0, confidence=0.7, detector_type="visual", event_type="motion_spike"),
    ]
    result = adapter.post_process_signals(sigs)
    assert all(s.event_type != "scene_cut" for s in result)
    assert len(result) == 2


def test_valorant_adapter_deduplicates_close_audio_signals() -> None:
    adapter = ValorantAdapter(_game_config("valorant"))
    # Two audio signals only 0.2 s apart (below 0.3 s gap threshold)
    sigs = [
        DetectorSignal(timestamp_sec=5.0, confidence=0.6, detector_type="audio", event_type="audio_spike"),
        DetectorSignal(timestamp_sec=5.2, confidence=0.9, detector_type="audio", event_type="audio_spike"),  # kept
        DetectorSignal(timestamp_sec=10.0, confidence=0.7, detector_type="audio", event_type="audio_spike"),
    ]
    result = adapter.post_process_signals(sigs)
    audio_sigs = [s for s in result if s.detector_type == "audio"]
    assert len(audio_sigs) == 2  # deduplicated pair + far signal
    # The kept near signal should be the higher-confidence one
    near_sig = next(s for s in audio_sigs if abs(s.timestamp_sec - 5.2) < 0.1)
    assert near_sig.confidence == pytest.approx(0.9)


def test_valorant_adapter_keeps_well_separated_audio_signals() -> None:
    adapter = ValorantAdapter(_game_config("valorant"))
    sigs = [
        DetectorSignal(timestamp_sec=1.0, confidence=0.8, detector_type="audio", event_type="audio_spike"),
        DetectorSignal(timestamp_sec=5.0, confidence=0.7, detector_type="audio", event_type="audio_spike"),
        DetectorSignal(timestamp_sec=9.0, confidence=0.9, detector_type="audio", event_type="audio_spike"),
    ]
    result = adapter.post_process_signals(sigs)
    audio_sigs = [s for s in result if s.detector_type == "audio"]
    assert len(audio_sigs) == 3

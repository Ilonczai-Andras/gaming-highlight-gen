"""Game-specific configuration loaded from YAML files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel


class MomentConfig(BaseModel):
    """Configuration for moment buffering and merging.

    Attributes:
        pre_buffer_sec: Seconds to include before a detected moment.
        post_buffer_sec: Seconds to include after a detected moment.
        min_gap_sec: Moments closer than this are merged into one clip.
        max_clip_duration_sec: Hard cap on a single clip length.
    """

    pre_buffer_sec: float = 3.0
    post_buffer_sec: float = 2.0
    min_gap_sec: float = 1.0
    max_clip_duration_sec: float = 15.0


class AudioDetectorConfig(BaseModel):
    """Configuration for the audio spike detector.

    Attributes:
        enabled: Whether this detector is active.
        spike_threshold_sigma: Threshold = mean + N*std of the RMS signal.
        min_peak_distance_sec: Minimum seconds between two detected peaks.
        max_signals_per_minute: Cap to prevent overly dense detections.
        weight: Weight assigned to this detector in the composite scorer.
    """

    enabled: bool = True
    spike_threshold_sigma: float = 2.0
    min_peak_distance_sec: float = 1.5
    max_signals_per_minute: int = 20
    weight: float = 0.5


class VisualDetectorConfig(BaseModel):
    """Configuration for the visual frame-diff detector.

    Attributes:
        enabled: Whether this detector is active.
        sample_every_n_frames: Process every N-th frame for performance.
        threshold_sigma: Adaptive threshold = mean + N*std of change scores.
        min_peak_distance_sec: Minimum seconds between two detected peaks.
        smooth_window_sec: Rolling average window for score smoothing.
        hard_cut_threshold: Change score above this is classified as a scene cut.
        filter_scene_cuts: When True, scene_cut signals are removed before scoring.
        weight: Weight assigned to this detector in the composite scorer.
    """

    enabled: bool = True
    sample_every_n_frames: int = 3
    threshold_sigma: float = 2.5
    min_peak_distance_sec: float = 1.0
    smooth_window_sec: float = 0.5
    hard_cut_threshold: float = 0.85
    filter_scene_cuts: bool = True
    weight: float = 0.4


class CompositeConfig(BaseModel):
    """Configuration for the composite moment scorer.

    Attributes:
        cluster_window_sec: Signals within this window are merged into one Moment.
        min_moment_score: Moments below this score are discarded.
        max_moments_per_video: Maximum number of moments returned per video.
        ml_weight: Weight for the ML detector (0.0 until Sprint 5).
    """

    cluster_window_sec: float = 2.0
    min_moment_score: float = 0.3
    max_moments_per_video: int = 15
    ml_weight: float = 0.0  # TODO(sprint-5): increase when ML detector is active


class GameConfig(BaseModel):
    """Game-specific configuration.

    Attributes:
        game_id: Unique game identifier (e.g. "valorant").
        display_name: Human-readable game name.
        moment: Moment buffer/merge settings.
        audio: Audio detector settings.
        visual: Visual detector settings.
        composite: Composite scorer settings.
        extra: Arbitrary extra fields (future use).
    """

    game_id: str
    display_name: str
    moment: MomentConfig = MomentConfig()
    audio: AudioDetectorConfig = AudioDetectorConfig()
    visual: VisualDetectorConfig = VisualDetectorConfig()
    composite: CompositeConfig = CompositeConfig()
    # TODO(sprint-3): color_preset, subtitle_style
    # TODO(sprint-5): detector_model_path, event_types
    extra: dict[str, Any] = {}


def load_game_config(game_id: str, config_dir: Path) -> GameConfig:
    """Load a game config from YAML, deep-merging it with ``default.yaml``.

    Args:
        game_id: Game identifier to load (e.g. "valorant").
        config_dir: Directory that contains ``default.yaml`` and
            optional ``{game_id}.yaml`` files.

    Returns:
        Fully merged :class:`GameConfig` instance.

    Raises:
        FileNotFoundError: If ``default.yaml`` or the requested game YAML
            is not found in *config_dir*.
    """
    default_path = config_dir / "default.yaml"
    if not default_path.exists():
        raise FileNotFoundError(f"Default config not found: {default_path}")

    with default_path.open("r", encoding="utf-8") as fh:
        default_data: dict[str, Any] = yaml.safe_load(fh) or {}

    if game_id == "default":
        return GameConfig(**default_data)

    game_path = config_dir / f"{game_id}.yaml"
    if not game_path.exists():
        raise FileNotFoundError(f"Game config not found: {game_path}")

    with game_path.open("r", encoding="utf-8") as fh:
        game_data: dict[str, Any] = yaml.safe_load(fh) or {}

    merged = _deep_merge(default_data, game_data)
    return GameConfig(**merged)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge *override* into *base*, returning a new dict.

    Nested dicts are merged recursively; all other values are replaced.
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result



def load_game_config(game_id: str, config_dir: Path) -> GameConfig:
    """Load a game config from YAML, deep-merging it with ``default.yaml``.

    Args:
        game_id: Game identifier to load (e.g. "valorant").
        config_dir: Directory that contains ``default.yaml`` and
            optional ``{game_id}.yaml`` files.

    Returns:
        Fully merged :class:`GameConfig` instance.

    Raises:
        FileNotFoundError: If ``default.yaml`` or the requested game YAML
            is not found in *config_dir*.
    """
    default_path = config_dir / "default.yaml"
    if not default_path.exists():
        raise FileNotFoundError(f"Default config not found: {default_path}")

    with default_path.open("r", encoding="utf-8") as fh:
        default_data: dict[str, Any] = yaml.safe_load(fh) or {}

    if game_id == "default":
        return GameConfig(**default_data)

    game_path = config_dir / f"{game_id}.yaml"
    if not game_path.exists():
        raise FileNotFoundError(f"Game config not found: {game_path}")

    with game_path.open("r", encoding="utf-8") as fh:
        game_data: dict[str, Any] = yaml.safe_load(fh) or {}

    merged = _deep_merge(default_data, game_data)
    return GameConfig(**merged)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge *override* into *base*, returning a new dict.

    Nested dicts are merged recursively; all other values are replaced.
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result

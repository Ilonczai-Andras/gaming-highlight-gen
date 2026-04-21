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


class GameConfig(BaseModel):
    """Game-specific configuration.

    Attributes:
        game_id: Unique game identifier (e.g. "valorant").
        display_name: Human-readable game name.
        moment: Moment buffer/merge settings.
        extra: Arbitrary extra fields (future use).
    """

    game_id: str
    display_name: str
    moment: MomentConfig = MomentConfig()
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

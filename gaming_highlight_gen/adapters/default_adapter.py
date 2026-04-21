"""Default game adapter — runs all enabled detectors with default settings."""

from __future__ import annotations

from pathlib import Path

from gaming_highlight_gen.adapters.base import BaseGameAdapter
from gaming_highlight_gen.config.game_config import GameConfig, load_game_config
from gaming_highlight_gen.detectors.audio_detector import AudioDetector
from gaming_highlight_gen.detectors.base import BaseDetector
from gaming_highlight_gen.detectors.ml_detector import MLDetector
from gaming_highlight_gen.detectors.visual_detector import VisualDetector
from gaming_highlight_gen.models.moment import DetectorSignal


class DefaultAdapter(BaseGameAdapter):
    """Default adapter that runs AudioDetector, VisualDetector, and MLDetector.

    Post-processing is a pass-through; no game-specific filtering is applied.
    """

    def __init__(self, game_config: GameConfig) -> None:
        self._game_config = game_config

    @property
    def game_id(self) -> str:
        return self._game_config.game_id

    def get_game_config(self) -> GameConfig:
        return self._game_config

    def get_detectors(self) -> list[BaseDetector]:
        detectors: list[BaseDetector] = []
        if self._game_config.audio.enabled:
            detectors.append(AudioDetector(self._game_config.audio))
        if self._game_config.visual.enabled:
            detectors.append(VisualDetector(self._game_config.visual))
        detectors.append(MLDetector())
        return detectors

    def post_process_signals(
        self, signals: list[DetectorSignal]
    ) -> list[DetectorSignal]:
        """Pass-through — no game-specific post-processing."""
        return signals

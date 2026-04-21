"""Abstract base adapter and shared types."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from gaming_highlight_gen.config.game_config import GameConfig
from gaming_highlight_gen.detectors.base import BaseDetector
from gaming_highlight_gen.models.moment import DetectorSignal


class BaseGameAdapter(ABC):
    """Abstract base class for game-specific adapter implementations.

    An adapter is responsible for:
    1. Providing the set of detectors to run for its game.
    2. Post-processing raw signals (e.g. filtering, deduplication).
    3. Supplying the game's :class:`GameConfig`.
    """

    @property
    @abstractmethod
    def game_id(self) -> str:
        """Unique identifier for this game (e.g. ``"valorant"``)."""

    @abstractmethod
    def get_game_config(self) -> GameConfig:
        """Return the :class:`GameConfig` for this game."""

    @abstractmethod
    def get_detectors(self) -> list[BaseDetector]:
        """Return ordered list of detectors to run for this game."""

    @abstractmethod
    def post_process_signals(
        self, signals: list[DetectorSignal]
    ) -> list[DetectorSignal]:
        """Apply game-specific signal filtering/deduplication.

        Args:
            signals: Raw signals collected from all detectors.

        Returns:
            Processed signal list.
        """

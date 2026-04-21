"""Adapter registry and factory for game-specific adapters."""

from __future__ import annotations

from pathlib import Path

from gaming_highlight_gen.adapters.base import BaseGameAdapter
from gaming_highlight_gen.adapters.default_adapter import DefaultAdapter
from gaming_highlight_gen.adapters.valorant_adapter import ValorantAdapter
from gaming_highlight_gen.config.game_config import GameConfig, load_game_config

__all__ = [
    "BaseGameAdapter",
    "DefaultAdapter",
    "ValorantAdapter",
    "get_adapter",
]

#: Map of game_id → adapter class.
#: Register new adapters here.
ADAPTER_REGISTRY: dict[str, type[BaseGameAdapter]] = {
    "valorant": ValorantAdapter,
}


def get_adapter(game_config: GameConfig) -> BaseGameAdapter:
    """Return the appropriate adapter for *game_config*.

    Uses :data:`ADAPTER_REGISTRY` to look up the adapter class.  Falls back to
    :class:`DefaultAdapter` when no specialised adapter is registered.

    Args:
        game_config: Loaded game configuration.

    Returns:
        Initialised :class:`BaseGameAdapter` instance.
    """
    adapter_cls = ADAPTER_REGISTRY.get(game_config.game_id, DefaultAdapter)
    return adapter_cls(game_config)

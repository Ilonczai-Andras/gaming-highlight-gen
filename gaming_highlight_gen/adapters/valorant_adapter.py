"""Valorant-specific game adapter."""

from __future__ import annotations

from gaming_highlight_gen.adapters.base import BaseGameAdapter
from gaming_highlight_gen.config.game_config import GameConfig
from gaming_highlight_gen.detectors.audio_detector import AudioDetector
from gaming_highlight_gen.detectors.base import BaseDetector
from gaming_highlight_gen.detectors.ml_detector import MLDetector
from gaming_highlight_gen.detectors.visual_detector import VisualDetector
from gaming_highlight_gen.models.moment import DetectorSignal


class ValorantAdapter(BaseGameAdapter):
    """Valorant-specific adapter.

    Differences from :class:`DefaultAdapter`:
    - Removes ``scene_cut`` signals (engine transitions are not highlights).
    - Deduplicates audio signals that arrive within 0.3 s of each other,
      keeping the one with the highest confidence.
    """

    _AUDIO_DEDUP_GAP_SEC: float = 0.3

    def __init__(self, game_config: GameConfig) -> None:
        self._game_config = game_config

    @property
    def game_id(self) -> str:
        return "valorant"

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
        """Filter scene cuts and deduplicate close audio signals."""
        # Remove scene-cut signals (already filtered by VisualDetectorConfig
        # if filter_scene_cuts=True, but handled here too for safety)
        filtered = [s for s in signals if s.event_type != "scene_cut"]

        # Deduplicate audio signals within _AUDIO_DEDUP_GAP_SEC
        audio_sigs = sorted(
            [s for s in filtered if s.detector_type == "audio"],
            key=lambda s: s.timestamp_sec,
        )
        non_audio = [s for s in filtered if s.detector_type != "audio"]

        deduped_audio: list[DetectorSignal] = []
        for sig in audio_sigs:
            if deduped_audio and (
                sig.timestamp_sec - deduped_audio[-1].timestamp_sec
                < self._AUDIO_DEDUP_GAP_SEC
            ):
                # Keep the higher-confidence signal
                if sig.confidence > deduped_audio[-1].confidence:
                    deduped_audio[-1] = sig
            else:
                deduped_audio.append(sig)

        return sorted(non_audio + deduped_audio, key=lambda s: s.timestamp_sec)

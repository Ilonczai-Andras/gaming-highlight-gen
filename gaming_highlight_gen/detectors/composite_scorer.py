"""Composite scorer: merges signals from multiple detectors into Moments."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import structlog

from gaming_highlight_gen.config.game_config import CompositeConfig
from gaming_highlight_gen.models.moment import DetectionResult, DetectorSignal, Moment

logger: structlog.BoundLogger = structlog.get_logger(__name__)


class CompositeScorer:
    """Combine signals from multiple :class:`DetectionResult` objects into moments.

    Algorithm:
    1. Collect all signals from all results.
    2. Sort by timestamp.
    3. Greedy clustering: start a new cluster when the gap since the last
       signal exceeds ``cluster_window_sec``.
    4. Score each cluster: weighted average of signal confidences, where
       weights come from each detector's configured weight.
    5. Discard clusters below ``min_moment_score``.
    6. Return at most ``max_moments_per_video`` moments, sorted by score.
    """

    def __init__(
        self,
        config: CompositeConfig,
        audio_weight: float = 0.5,
        visual_weight: float = 0.4,
        ml_weight: float = 0.0,
    ) -> None:
        self._config = config
        self._weights: dict[str, float] = {
            "audio": audio_weight,
            "visual": visual_weight,
            "ml": ml_weight,
        }

    def score(
        self,
        results: list[DetectionResult],
        source_file: Path,
    ) -> list[Moment]:
        """Produce scored :class:`Moment` objects from detector results.

        Args:
            results: One result per detector that ran.
            source_file: The video file that was analysed.

        Returns:
            Sorted list of :class:`Moment` objects (highest score first),
            capped at ``max_moments_per_video``.
        """
        all_signals: list[DetectorSignal] = []
        for result in results:
            all_signals.extend(result.signals)

        if not all_signals:
            logger.info("composite_scorer.no_signals", source=str(source_file))
            return []

        all_signals.sort(key=lambda s: s.timestamp_sec)

        clusters = self._cluster(all_signals)
        moments: list[Moment] = []

        for cluster in clusters:
            moment = self._cluster_to_moment(cluster, source_file)
            if moment.score >= self._config.min_moment_score:
                moments.append(moment)

        # Sort by score descending, cap to max
        moments.sort(key=lambda m: m.score, reverse=True)
        moments = moments[: self._config.max_moments_per_video]

        logger.info(
            "composite_scorer.complete",
            source=str(source_file),
            moments_found=len(moments),
        )
        return moments

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _cluster(self, signals: list[DetectorSignal]) -> list[list[DetectorSignal]]:
        """Greedy time-window clustering of sorted signals."""
        clusters: list[list[DetectorSignal]] = []
        current: list[DetectorSignal] = [signals[0]]

        for sig in signals[1:]:
            if sig.timestamp_sec - current[-1].timestamp_sec <= self._config.cluster_window_sec:
                current.append(sig)
            else:
                clusters.append(current)
                current = [sig]
        clusters.append(current)
        return clusters

    def _cluster_to_moment(
        self, cluster: list[DetectorSignal], source_file: Path
    ) -> Moment:
        """Convert a signal cluster into a single :class:`Moment`."""
        timestamps = [s.timestamp_sec for s in cluster]
        center_ts = float(np.mean(timestamps))

        # Weighted average confidence per detector type
        detector_scores: dict[str, list[float]] = {}
        for sig in cluster:
            detector_scores.setdefault(sig.detector_type, []).append(sig.confidence)

        breakdown: dict[str, float] = {}
        total_weight = 0.0
        weighted_sum = 0.0

        for det_type, confidences in detector_scores.items():
            avg_conf = float(np.mean(confidences))
            breakdown[det_type] = avg_conf
            w = self._weights.get(det_type, 0.1)
            weighted_sum += avg_conf * w
            total_weight += w

        composite_score = (weighted_sum / total_weight) if total_weight > 0 else 0.0
        composite_score = float(np.clip(composite_score, 0.0, 1.0))

        # Pick the most common event_type in this cluster
        event_counts: dict[str, int] = {}
        for sig in cluster:
            event_counts[sig.event_type] = event_counts.get(sig.event_type, 0) + 1
        dominant_event = max(event_counts, key=lambda k: event_counts[k])

        return Moment(
            start_sec=center_ts,
            end_sec=center_ts + 0.001,  # placeholder; buffers applied by pipeline/renderer
            score=composite_score,
            source_file=source_file,
            event_type=dominant_event,
            contributing_signals=list(cluster),
            detector_breakdown=breakdown,
        )

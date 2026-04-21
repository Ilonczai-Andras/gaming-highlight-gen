"""Core data models for the gaming highlight pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class DetectorSignal:
    """A single potential moment flagged by one detector.

    Attributes:
        timestamp_sec: Detection timestamp in seconds.
        confidence: Signal confidence in range [0.0, 1.0].
        detector_type: Source detector: ``"audio"``, ``"visual"``, or ``"ml"``.
        event_type: Event category (e.g. ``"kill"``, ``"explosion"``, ``"scene_cut"``).
        raw_value: Raw numeric value from the detector (e.g. RMS energy or frame diff score).
        metadata: Arbitrary extra data attached to this signal.
    """

    timestamp_sec: float
    confidence: float  # 0.0–1.0
    detector_type: str  # "audio" | "visual" | "ml"
    event_type: str = "generic"
    raw_value: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DetectionResult:
    """Full result of one detector run on a video.

    Attributes:
        detector_type: Identifier of the detector that produced this result.
        signals: All signals found during this run.
        processing_time_sec: Wall-clock time used for processing.
        video_duration_sec: Duration of the source video.
        config_snapshot: Copy of the relevant config at the time of detection.
    """

    detector_type: str
    signals: list[DetectorSignal]
    processing_time_sec: float
    video_duration_sec: float
    config_snapshot: dict[str, Any] = field(default_factory=dict)


@dataclass
class Moment:
    """A detected interesting moment in a gameplay video.

    Attributes:
        start_sec: Moment start time in seconds.
        end_sec: Moment end time in seconds.
        score: Importance score in range [0.0, 1.0].
        source_file: Path to the source video file.
        event_type: Event category (e.g. "kill", "spike_plant", "generic").
        metadata: Arbitrary extra data attached to this moment.
        contributing_signals: All detector signals that contributed to this moment.
        detector_breakdown: Per-detector confidence scores, e.g.
            ``{"audio": 0.8, "visual": 0.6, "ml": 0.0}``.
    """

    start_sec: float
    end_sec: float
    score: float  # 0.0–1.0 importance score
    source_file: Path
    event_type: str = "generic"  # e.g. "kill", "spike_plant", "generic"
    metadata: dict[str, Any] = field(default_factory=dict)
    contributing_signals: list[DetectorSignal] = field(default_factory=list)
    detector_breakdown: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate moment field invariants."""
        if self.start_sec < 0:
            raise ValueError(f"start_sec must be >= 0, got {self.start_sec}")
        if self.end_sec <= self.start_sec:
            raise ValueError(
                f"end_sec must be > start_sec, got {self.end_sec} <= {self.start_sec}"
            )
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"score must be in [0.0, 1.0], got {self.score}")


@dataclass
class ClipSegment:
    """A render-ready video segment derived from a Moment.

    Attributes:
        source_file: Path to the source video file.
        start_sec: Segment start time in seconds.
        end_sec: Segment end time in seconds.
        moment: The originating Moment, if applicable.
    """

    source_file: Path
    start_sec: float
    end_sec: float
    moment: Moment | None = None

    @property
    def duration_sec(self) -> float:
        """Return the duration of this segment in seconds."""
        return self.end_sec - self.start_sec

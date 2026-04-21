"""Core data models for the gaming highlight pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


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
    """

    start_sec: float
    end_sec: float
    score: float  # 0.0–1.0 importance score
    source_file: Path
    event_type: str = "generic"  # e.g. "kill", "spike_plant", "generic"
    metadata: dict[str, Any] = field(default_factory=dict)

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

"""Shared pytest fixtures for the gaming highlight generator test suite."""

from pathlib import Path

import pytest

from gaming_highlight_gen.config.game_config import GameConfig, MomentConfig
from gaming_highlight_gen.config.global_config import GlobalConfig
from gaming_highlight_gen.models.moment import ClipSegment, Moment


@pytest.fixture
def global_config(tmp_path: Path) -> GlobalConfig:
    """Return a GlobalConfig pointing to isolated temp directories."""
    return GlobalConfig(
        output_dir=tmp_path / "output",
        temp_dir=tmp_path / ".tmp",
    )


@pytest.fixture
def game_config() -> GameConfig:
    """Return a default GameConfig for tests."""
    return GameConfig(
        game_id="test",
        display_name="Test Game",
        moment=MomentConfig(
            pre_buffer_sec=3.0,
            post_buffer_sec=2.0,
            min_gap_sec=1.0,
            max_clip_duration_sec=15.0,
        ),
    )


@pytest.fixture
def sample_video_path(tmp_path: Path) -> Path:
    """Return a path to a (non-existent) fake video file.

    Most tests mock FFmpeg calls so the file doesn't need real content.
    """
    p = tmp_path / "gameplay.mp4"
    p.touch()
    return p


@pytest.fixture
def sample_moment(sample_video_path: Path) -> Moment:
    """Return a sample Moment instance."""
    return Moment(
        start_sec=10.0,
        end_sec=15.0,
        score=0.8,
        source_file=sample_video_path,
        event_type="kill",
    )


@pytest.fixture
def sample_clip_segment(sample_moment: Moment) -> ClipSegment:
    """Return a ClipSegment derived from sample_moment."""
    return ClipSegment(
        source_file=sample_moment.source_file,
        start_sec=sample_moment.start_sec,
        end_sec=sample_moment.end_sec,
        moment=sample_moment,
    )

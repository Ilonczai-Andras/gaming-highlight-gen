"""Tests for gaming_highlight_gen.core.pipeline (and Renderer._apply_buffers)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gaming_highlight_gen.config.game_config import GameConfig, MomentConfig
from gaming_highlight_gen.config.global_config import GlobalConfig
from gaming_highlight_gen.core.pipeline import Pipeline
from gaming_highlight_gen.core.renderer import RenderResult
from gaming_highlight_gen.models.moment import Moment


# ---------------------------------------------------------------------------
# Fixtures (local overrides for pipeline-specific config)
# ---------------------------------------------------------------------------


@pytest.fixture
def pipeline_config(tmp_path: Path) -> GlobalConfig:
    return GlobalConfig(
        output_dir=tmp_path / "output",
        temp_dir=tmp_path / ".tmp",
    )


@pytest.fixture
def pipeline_game_config() -> GameConfig:
    return GameConfig(
        game_id="test",
        display_name="Test Game",
        moment=MomentConfig(
            pre_buffer_sec=2.0,
            post_buffer_sec=1.0,
            min_gap_sec=0.5,
            max_clip_duration_sec=10.0,
        ),
    )


@pytest.fixture
def pipeline(pipeline_config: GlobalConfig, pipeline_game_config: GameConfig) -> Pipeline:
    return Pipeline(pipeline_config, pipeline_game_config)


# ---------------------------------------------------------------------------
# _generate_dummy_moments
# ---------------------------------------------------------------------------


class TestGenerateDummyMoments:
    def test_generates_requested_count(
        self, pipeline: Pipeline, tmp_path: Path
    ) -> None:
        """_generate_dummy_moments produces exactly *count* moments."""
        video = tmp_path / "video.mp4"
        video.touch()

        with patch.object(pipeline._ffmpeg, "get_duration", return_value=120.0):
            moments = pipeline._generate_dummy_moments(video, count=3)

        assert len(moments) == 3

    def test_default_count_is_three(self, pipeline: Pipeline, tmp_path: Path) -> None:
        """Default count argument produces 3 moments."""
        video = tmp_path / "video.mp4"
        video.touch()

        with patch.object(pipeline._ffmpeg, "get_duration", return_value=60.0):
            moments = pipeline._generate_dummy_moments(video)

        assert len(moments) == 3

    def test_moments_within_video_bounds(
        self, pipeline: Pipeline, tmp_path: Path
    ) -> None:
        """All generated moments fall within [0, video_duration]."""
        video = tmp_path / "video.mp4"
        video.touch()
        duration = 60.0

        with patch.object(pipeline._ffmpeg, "get_duration", return_value=duration):
            moments = pipeline._generate_dummy_moments(video, count=5)

        for m in moments:
            assert m.start_sec >= 0.0
            assert m.end_sec <= duration
            assert m.start_sec < m.end_sec

    def test_scores_are_valid(self, pipeline: Pipeline, tmp_path: Path) -> None:
        """Every moment has a score in [0.0, 1.0]."""
        video = tmp_path / "video.mp4"
        video.touch()

        with patch.object(pipeline._ffmpeg, "get_duration", return_value=90.0):
            moments = pipeline._generate_dummy_moments(video)

        for m in moments:
            assert 0.0 <= m.score <= 1.0

    def test_zero_duration_returns_empty(
        self, pipeline: Pipeline, tmp_path: Path
    ) -> None:
        """_generate_dummy_moments returns [] when video duration is 0."""
        video = tmp_path / "video.mp4"
        video.touch()

        with patch.object(pipeline._ffmpeg, "get_duration", return_value=0.0):
            moments = pipeline._generate_dummy_moments(video)

        assert moments == []

    def test_zero_count_returns_empty(self, pipeline: Pipeline, tmp_path: Path) -> None:
        """_generate_dummy_moments returns [] when count=0."""
        video = tmp_path / "video.mp4"
        video.touch()

        with patch.object(pipeline._ffmpeg, "get_duration", return_value=60.0):
            moments = pipeline._generate_dummy_moments(video, count=0)

        assert moments == []

    def test_moments_distributed_evenly(
        self, pipeline: Pipeline, tmp_path: Path
    ) -> None:
        """Moments are roughly evenly spaced across the video."""
        video = tmp_path / "video.mp4"
        video.touch()
        duration = 100.0

        with patch.object(pipeline._ffmpeg, "get_duration", return_value=duration):
            moments = pipeline._generate_dummy_moments(video, count=4)

        centers = [(m.start_sec + m.end_sec) / 2 for m in moments]
        # Expected centers: 20, 40, 60, 80
        expected = [20.0, 40.0, 60.0, 80.0]
        for actual, exp in zip(centers, expected, strict=True):
            assert actual == pytest.approx(exp, abs=1.0)


# ---------------------------------------------------------------------------
# Renderer._apply_buffers (tested via pipeline._renderer)
# ---------------------------------------------------------------------------


class TestApplyBuffers:
    def _make_moment(
        self, video: Path, start: float, end: float, score: float = 0.5
    ) -> Moment:
        return Moment(
            start_sec=start, end_sec=end, score=score, source_file=video
        )

    def test_applies_pre_and_post_buffer(
        self, pipeline: Pipeline, tmp_path: Path
    ) -> None:
        """Segments are expanded by pre_buffer_sec before and post_buffer_sec after."""
        video = tmp_path / "video.mp4"
        moment = self._make_moment(video, start=10.0, end=12.0)

        segments = pipeline._renderer._apply_buffers(
            [moment],
            video_duration=120.0,
            game_config=pipeline._game_config,
        )

        assert len(segments) == 1
        # pre_buffer=2.0 → start = 10 - 2 = 8
        assert segments[0].start_sec == pytest.approx(8.0)
        # post_buffer=1.0 → end = 12 + 1 = 13
        assert segments[0].end_sec == pytest.approx(13.0)

    def test_merges_overlapping_moments(
        self, pipeline: Pipeline, tmp_path: Path
    ) -> None:
        """Moments whose buffered intervals overlap are merged into one segment."""
        video = tmp_path / "video.mp4"
        moments = [
            self._make_moment(video, 10.0, 11.0, score=0.7),
            self._make_moment(video, 11.5, 12.5, score=0.9),
        ]

        segments = pipeline._renderer._apply_buffers(
            moments,
            video_duration=120.0,
            game_config=pipeline._game_config,
        )

        # After buffering:
        #  m1 → [8.0, 12.0], m2 → [9.5, 13.5]  → they overlap → merge to [8.0, 13.5]
        assert len(segments) == 1

    def test_keeps_high_score_moment_after_merge(
        self, pipeline: Pipeline, tmp_path: Path
    ) -> None:
        """After merging, the segment references the higher-scoring moment."""
        video = tmp_path / "video.mp4"
        moments = [
            self._make_moment(video, 10.0, 11.0, score=0.3),
            self._make_moment(video, 11.5, 12.5, score=0.9),
        ]

        segments = pipeline._renderer._apply_buffers(
            moments,
            video_duration=120.0,
            game_config=pipeline._game_config,
        )

        assert segments[0].moment is not None
        assert segments[0].moment.score == pytest.approx(0.9)

    def test_clamps_start_to_zero(self, pipeline: Pipeline, tmp_path: Path) -> None:
        """Segments are clamped so that start_sec >= 0."""
        video = tmp_path / "video.mp4"
        moment = self._make_moment(video, start=1.0, end=2.0)

        segments = pipeline._renderer._apply_buffers(
            [moment],
            video_duration=30.0,
            game_config=pipeline._game_config,
        )

        assert segments[0].start_sec >= 0.0

    def test_clamps_end_to_video_duration(
        self, pipeline: Pipeline, tmp_path: Path
    ) -> None:
        """Segments are clamped so that end_sec <= video_duration."""
        video = tmp_path / "video.mp4"
        moment = self._make_moment(video, start=28.0, end=29.0)

        segments = pipeline._renderer._apply_buffers(
            [moment],
            video_duration=30.0,
            game_config=pipeline._game_config,
        )

        assert segments[0].end_sec <= 30.0

    def test_empty_moments_returns_empty_list(
        self, pipeline: Pipeline, tmp_path: Path
    ) -> None:
        """_apply_buffers returns an empty list when given no moments."""
        segments = pipeline._renderer._apply_buffers(
            [],
            video_duration=120.0,
            game_config=pipeline._game_config,
        )
        assert segments == []

    def test_respects_max_clip_duration(
        self, pipeline: Pipeline, tmp_path: Path
    ) -> None:
        """A single clip is never longer than max_clip_duration_sec."""
        video = tmp_path / "video.mp4"
        # With pre_buffer=2, post_buffer=1, a 5s moment → 8s clip (within 10s cap)
        moment = self._make_moment(video, start=50.0, end=55.0)

        segments = pipeline._renderer._apply_buffers(
            [moment],
            video_duration=120.0,
            game_config=pipeline._game_config,
        )

        assert len(segments) == 1
        assert segments[0].duration_sec <= 10.0


# ---------------------------------------------------------------------------
# Pipeline.run – integration (mocked FFmpeg)
# ---------------------------------------------------------------------------


class TestPipelineRun:
    def _make_render_result(self, output: Path) -> RenderResult:
        return RenderResult(
            output_path=output,
            duration_sec=30.0,
            thumbnail_path=None,
            segments_count=3,
        )

    def test_run_success_with_dummy_moments(
        self,
        pipeline: Pipeline,
        tmp_path: Path,
    ) -> None:
        """Pipeline.run completes end-to-end with mocked FFmpeg calls."""
        video = tmp_path / "video.mp4"
        video.touch()
        output = tmp_path / "output.mp4"

        mock_result = self._make_render_result(output)

        with (
            patch.object(pipeline._ffmpeg, "get_duration", return_value=120.0),
            patch.object(pipeline._renderer, "render", return_value=mock_result),
        ):
            result = pipeline.run(input_files=[video], output_path=output)

        assert result.output_path == output
        assert result.moments_count == 3  # default count

    def test_run_accepts_explicit_moments(
        self,
        pipeline: Pipeline,
        tmp_path: Path,
    ) -> None:
        """Pipeline.run uses provided moments instead of generating dummy ones."""
        video = tmp_path / "video.mp4"
        video.touch()
        output = tmp_path / "output.mp4"

        explicit_moments = [
            Moment(
                start_sec=10.0,
                end_sec=11.0,
                score=0.8,
                source_file=video,
            )
        ]
        mock_result = self._make_render_result(output)

        with (
            patch.object(pipeline._ffmpeg, "get_duration", return_value=120.0),
            patch.object(pipeline._renderer, "render", return_value=mock_result),
        ):
            result = pipeline.run(
                input_files=[video], output_path=output, moments=explicit_moments
            )

        assert result.moments_count == 1

    def test_run_raises_for_missing_input_file(
        self, pipeline: Pipeline, tmp_path: Path
    ) -> None:
        """Pipeline.run raises FileNotFoundError when an input path doesn't exist."""
        with pytest.raises(FileNotFoundError):
            pipeline.run(
                input_files=[tmp_path / "nonexistent.mp4"],
                output_path=tmp_path / "out.mp4",
            )

    def test_run_raises_for_empty_input_list(
        self, pipeline: Pipeline, tmp_path: Path
    ) -> None:
        """Pipeline.run raises ValueError when input_files is empty."""
        with pytest.raises(ValueError, match="empty"):
            pipeline.run(
                input_files=[],
                output_path=tmp_path / "out.mp4",
            )

    def test_run_cleans_up_temp_dir(
        self, pipeline: Pipeline, tmp_path: Path
    ) -> None:
        """Pipeline.run removes the temp directory after completion."""
        video = tmp_path / "video.mp4"
        video.touch()
        output = tmp_path / "output.mp4"
        temp_dir = pipeline._global_config.temp_dir
        # Pre-create temp dir to simulate leftover files
        temp_dir.mkdir(parents=True, exist_ok=True)

        mock_result = self._make_render_result(output)

        with (
            patch.object(pipeline._ffmpeg, "get_duration", return_value=60.0),
            patch.object(pipeline._renderer, "render", return_value=mock_result),
        ):
            pipeline.run(input_files=[video], output_path=output)

        assert not temp_dir.exists()

    def test_run_raises_when_explicit_moments_produce_no_segments(
        self,
        pipeline: Pipeline,
        tmp_path: Path,
    ) -> None:
        """Pipeline.run raises ValueError when an explicit empty moments list is given."""
        video = tmp_path / "video.mp4"
        video.touch()

        with pytest.raises(ValueError, match="No segments"):
            pipeline.run(
                input_files=[video],
                output_path=tmp_path / "out.mp4",
                moments=[],  # empty → no segments built
            )

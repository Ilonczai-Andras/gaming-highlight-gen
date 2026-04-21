"""Tests for gaming_highlight_gen.core.renderer and gaming_highlight_gen.logging_setup."""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gaming_highlight_gen.config.game_config import GameConfig, MomentConfig
from gaming_highlight_gen.config.global_config import GlobalConfig
from gaming_highlight_gen.core.ffmpeg_wrapper import FFmpegError, FFmpegWrapper
from gaming_highlight_gen.core.renderer import RenderResult, Renderer
from gaming_highlight_gen.logging_setup import setup_logging
from gaming_highlight_gen.models.moment import ClipSegment, Moment


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def render_config(tmp_path: Path) -> GlobalConfig:
    return GlobalConfig(
        output_dir=tmp_path / "output",
        temp_dir=tmp_path / ".tmp",
    )


@pytest.fixture
def render_game_config() -> GameConfig:
    return GameConfig(
        game_id="test",
        display_name="Test",
        moment=MomentConfig(pre_buffer_sec=2.0, post_buffer_sec=1.0),
    )


@pytest.fixture
def ffmpeg_mock(render_config: GlobalConfig) -> FFmpegWrapper:
    return FFmpegWrapper(render_config)


@pytest.fixture
def renderer(ffmpeg_mock: FFmpegWrapper, render_config: GlobalConfig) -> Renderer:
    return Renderer(ffmpeg_mock, render_config)


def _make_segment(source: Path, start: float = 0.0, end: float = 5.0) -> ClipSegment:
    moment = Moment(start_sec=start, end_sec=end, score=0.5, source_file=source)
    return ClipSegment(source_file=source, start_sec=start, end_sec=end, moment=moment)


# ---------------------------------------------------------------------------
# Renderer.render
# ---------------------------------------------------------------------------


class TestRendererRender:
    def test_render_returns_render_result(
        self,
        renderer: Renderer,
        render_game_config: GameConfig,
        tmp_path: Path,
    ) -> None:
        """render() returns a RenderResult with correct segment count."""
        source = tmp_path / "video.mp4"
        source.touch()
        output = tmp_path / "output.mp4"

        segments = [_make_segment(source, 0.0, 5.0), _make_segment(source, 10.0, 15.0)]

        with (
            patch.object(renderer._ffmpeg, "cut_segment", return_value=MagicMock()),
            patch.object(renderer._ffmpeg, "concat_segments", return_value=output),
            patch.object(renderer._ffmpeg, "extract_thumbnail", return_value=tmp_path / "t.jpg"),
            patch.object(renderer._ffmpeg, "get_duration", return_value=10.0),
        ):
            result = renderer.render(segments, output, game_config=render_game_config)

        assert isinstance(result, RenderResult)
        assert result.segments_count == 2
        assert result.duration_sec == pytest.approx(10.0)

    def test_render_calls_cut_segment_per_item(
        self,
        renderer: Renderer,
        render_game_config: GameConfig,
        tmp_path: Path,
    ) -> None:
        """render() calls cut_segment once per ClipSegment."""
        source = tmp_path / "video.mp4"
        source.touch()
        output = tmp_path / "output.mp4"
        segments = [_make_segment(source, float(i), float(i + 5)) for i in range(4)]

        with (
            patch.object(renderer._ffmpeg, "cut_segment") as mock_cut,
            patch.object(renderer._ffmpeg, "concat_segments", return_value=output),
            patch.object(renderer._ffmpeg, "extract_thumbnail", return_value=tmp_path / "t.jpg"),
            patch.object(renderer._ffmpeg, "get_duration", return_value=20.0),
        ):
            mock_cut.return_value = tmp_path / "seg.mp4"
            renderer.render(segments, output, game_config=render_game_config)

        assert mock_cut.call_count == 4

    def test_render_raises_on_empty_segments(
        self,
        renderer: Renderer,
        render_game_config: GameConfig,
        tmp_path: Path,
    ) -> None:
        """render() raises ValueError when given an empty segment list."""
        with pytest.raises(ValueError, match="empty"):
            renderer.render([], tmp_path / "output.mp4", game_config=render_game_config)

    def test_render_invokes_progress_callback(
        self,
        renderer: Renderer,
        render_game_config: GameConfig,
        tmp_path: Path,
    ) -> None:
        """render() calls the progress callback after each segment is cut."""
        source = tmp_path / "video.mp4"
        source.touch()
        output = tmp_path / "output.mp4"
        segments = [_make_segment(source, 0.0, 5.0), _make_segment(source, 10.0, 15.0)]
        progress_calls: list[tuple[int, int]] = []

        with (
            patch.object(renderer._ffmpeg, "cut_segment", return_value=MagicMock()),
            patch.object(renderer._ffmpeg, "concat_segments", return_value=output),
            patch.object(renderer._ffmpeg, "extract_thumbnail", return_value=tmp_path / "t.jpg"),
            patch.object(renderer._ffmpeg, "get_duration", return_value=10.0),
        ):
            renderer.render(
                segments,
                output,
                game_config=render_game_config,
                progress_callback=lambda cur, tot: progress_calls.append((cur, tot)),
            )

        assert progress_calls == [(1, 2), (2, 2)]

    def test_render_handles_thumbnail_failure_gracefully(
        self,
        renderer: Renderer,
        render_game_config: GameConfig,
        tmp_path: Path,
    ) -> None:
        """render() sets thumbnail_path=None when extract_thumbnail raises."""
        source = tmp_path / "video.mp4"
        source.touch()
        output = tmp_path / "output.mp4"
        segments = [_make_segment(source)]

        with (
            patch.object(renderer._ffmpeg, "cut_segment", return_value=MagicMock()),
            patch.object(renderer._ffmpeg, "concat_segments", return_value=output),
            patch.object(
                renderer._ffmpeg,
                "extract_thumbnail",
                side_effect=FFmpegError("fail", 1, "err"),
            ),
            patch.object(renderer._ffmpeg, "get_duration", return_value=5.0),
        ):
            result = renderer.render(segments, output, game_config=render_game_config)

        assert result.thumbnail_path is None

    def test_render_fallback_duration_on_get_duration_failure(
        self,
        renderer: Renderer,
        render_game_config: GameConfig,
        tmp_path: Path,
    ) -> None:
        """render() sums segment durations when get_duration on output fails."""
        source = tmp_path / "video.mp4"
        source.touch()
        output = tmp_path / "output.mp4"
        segments = [_make_segment(source, 0.0, 5.0), _make_segment(source, 10.0, 15.0)]

        with (
            patch.object(renderer._ffmpeg, "cut_segment", return_value=MagicMock()),
            patch.object(renderer._ffmpeg, "concat_segments", return_value=output),
            patch.object(renderer._ffmpeg, "extract_thumbnail", return_value=tmp_path / "t.jpg"),
            patch.object(
                renderer._ffmpeg, "get_duration", side_effect=FFmpegError("fail", 1, "")
            ),
        ):
            result = renderer.render(segments, output, game_config=render_game_config)

        # Fallback: sum of segment durations = 5 + 5 = 10
        assert result.duration_sec == pytest.approx(10.0)

    def test_clip_segment_duration_property(self, tmp_path: Path) -> None:
        """ClipSegment.duration_sec returns the correct computed duration."""
        seg = _make_segment(tmp_path / "v.mp4", start=3.0, end=8.5)
        assert seg.duration_sec == pytest.approx(5.5)


# ---------------------------------------------------------------------------
# setup_logging
# ---------------------------------------------------------------------------


class TestSetupLogging:
    def test_json_format_configures_structlog(self) -> None:
        """setup_logging('INFO', 'json') completes without error and sets level."""
        setup_logging("INFO", "json")
        assert logging.getLogger().level == logging.INFO

    def test_console_format_configures_structlog(self) -> None:
        """setup_logging('DEBUG', 'console') sets DEBUG level."""
        setup_logging("DEBUG", "console")
        assert logging.getLogger().level == logging.DEBUG

    def test_invalid_level_falls_back_to_info(self) -> None:
        """setup_logging with an unrecognised level defaults to INFO."""
        setup_logging("NOTAREAL", "json")
        assert logging.getLogger().level == logging.INFO

    def test_logging_resets_level_on_second_call(self) -> None:
        """Calling setup_logging twice correctly updates the root logger level."""
        setup_logging("WARNING", "console")
        assert logging.getLogger().level == logging.WARNING
        setup_logging("DEBUG", "json")
        assert logging.getLogger().level == logging.DEBUG

"""Tests for gaming_highlight_gen.core.ffmpeg_wrapper."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gaming_highlight_gen.config.global_config import GlobalConfig
from gaming_highlight_gen.core.ffmpeg_wrapper import (
    FFmpegError,
    FFmpegWrapper,
    VideoInfo,
    _parse_fraction,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def config(tmp_path: Path) -> GlobalConfig:
    return GlobalConfig(temp_dir=tmp_path / ".tmp", output_dir=tmp_path / "output")


@pytest.fixture
def wrapper(config: GlobalConfig) -> FFmpegWrapper:
    return FFmpegWrapper(config)


def _make_probe_output(
    duration: float = 120.0,
    width: int = 1920,
    height: int = 1080,
    fps: str = "30/1",
    codec: str = "h264",
    audio_codec: str | None = "aac",
) -> str:
    """Build a minimal ffprobe JSON output string."""
    streams = [
        {
            "codec_type": "video",
            "codec_name": codec,
            "width": width,
            "height": height,
            "r_frame_rate": fps,
            "duration": str(duration),
        }
    ]
    if audio_codec:
        streams.append({"codec_type": "audio", "codec_name": audio_codec})
    return json.dumps({"streams": streams, "format": {"duration": str(duration)}})


# ---------------------------------------------------------------------------
# get_duration / get_video_info
# ---------------------------------------------------------------------------


class TestGetDuration:
    def test_returns_correct_duration(self, wrapper: FFmpegWrapper, tmp_path: Path) -> None:
        """get_duration returns the numeric duration from ffprobe output."""
        video = tmp_path / "video.mp4"
        video.touch()

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0, stdout=_make_probe_output(duration=120.0), stderr=""
            )
            duration = wrapper.get_duration(video)

        assert duration == pytest.approx(120.0)

    def test_raises_on_ffprobe_failure(self, wrapper: FFmpegWrapper, tmp_path: Path) -> None:
        """get_duration raises FFmpegError when ffprobe exits non-zero."""
        video = tmp_path / "video.mp4"
        video.touch()

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=1, stdout="", stderr="No such file"
            )
            with pytest.raises(FFmpegError) as exc_info:
                wrapper.get_duration(video)

        assert exc_info.value.returncode == 1
        assert "No such file" in exc_info.value.stderr

    def test_raises_when_no_video_stream(self, wrapper: FFmpegWrapper, tmp_path: Path) -> None:
        """get_duration raises FFmpegError when there is no video stream."""
        video = tmp_path / "audio_only.mp4"
        video.touch()
        probe_out = json.dumps({"streams": [], "format": {"duration": "60.0"}})

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout=probe_out, stderr="")
            with pytest.raises(FFmpegError):
                wrapper.get_duration(video)


class TestGetVideoInfo:
    def test_returns_video_info(self, wrapper: FFmpegWrapper, tmp_path: Path) -> None:
        """get_video_info populates all VideoInfo fields correctly."""
        video = tmp_path / "video.mp4"
        video.touch()

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout=_make_probe_output(
                    duration=60.0, width=1280, height=720, fps="60/1", codec="hevc"
                ),
                stderr="",
            )
            info = wrapper.get_video_info(video)

        assert isinstance(info, VideoInfo)
        assert info.duration == pytest.approx(60.0)
        assert info.width == 1280
        assert info.height == 720
        assert info.fps == pytest.approx(60.0)
        assert info.codec == "hevc"
        assert info.audio_codec == "aac"

    def test_audio_codec_is_none_when_missing(
        self, wrapper: FFmpegWrapper, tmp_path: Path
    ) -> None:
        """get_video_info sets audio_codec to None when there is no audio stream."""
        video = tmp_path / "video.mp4"
        video.touch()

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout=_make_probe_output(audio_codec=None),
                stderr="",
            )
            info = wrapper.get_video_info(video)

        assert info.audio_codec is None


# ---------------------------------------------------------------------------
# cut_segment
# ---------------------------------------------------------------------------


class TestCutSegment:
    def test_stream_copy_uses_copy_flag(
        self, wrapper: FFmpegWrapper, tmp_path: Path
    ) -> None:
        """cut_segment passes '-c copy' when re_encode=False (default)."""
        source = tmp_path / "source.mp4"
        output = tmp_path / "out.mp4"
        source.touch()

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            wrapper.cut_segment(source, output, start_sec=10.0, end_sec=20.0)

        args: list[str] = mock_run.call_args[0][0]
        assert "-c" in args
        assert "copy" in args
        # Duration equals end - start
        t_idx = args.index("-t")
        assert float(args[t_idx + 1]) == pytest.approx(10.0)

    def test_re_encode_uses_codec_flags(
        self, wrapper: FFmpegWrapper, tmp_path: Path
    ) -> None:
        """cut_segment passes codec flags when re_encode=True."""
        source = tmp_path / "source.mp4"
        output = tmp_path / "out.mp4"
        source.touch()

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            wrapper.cut_segment(
                source, output, start_sec=5.0, end_sec=15.0, re_encode=True
            )

        args: list[str] = mock_run.call_args[0][0]
        assert "-c:v" in args
        assert "libx264" in args

    def test_raises_ffmpeg_error_on_failure(
        self, wrapper: FFmpegWrapper, tmp_path: Path
    ) -> None:
        """cut_segment raises FFmpegError when ffmpeg exits non-zero."""
        source = tmp_path / "source.mp4"
        output = tmp_path / "out.mp4"
        source.touch()

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=1, stdout="", stderr="ffmpeg error"
            )
            with pytest.raises(FFmpegError) as exc_info:
                wrapper.cut_segment(source, output, start_sec=0.0, end_sec=5.0)

        assert exc_info.value.returncode == 1

    def test_returns_output_path(self, wrapper: FFmpegWrapper, tmp_path: Path) -> None:
        """cut_segment returns the output Path on success."""
        source = tmp_path / "source.mp4"
        output = tmp_path / "out.mp4"
        source.touch()

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            result = wrapper.cut_segment(source, output, start_sec=0.0, end_sec=5.0)

        assert result == output


# ---------------------------------------------------------------------------
# concat_segments
# ---------------------------------------------------------------------------


class TestConcatSegments:
    def test_uses_concat_demuxer(self, wrapper: FFmpegWrapper, tmp_path: Path) -> None:
        """concat_segments passes '-f concat' to ffmpeg."""
        segments = [tmp_path / f"seg_{i}.mp4" for i in range(3)]
        output = tmp_path / "output.mp4"

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            wrapper.concat_segments(segments, output)

        args: list[str] = mock_run.call_args[0][0]
        assert "-f" in args
        assert "concat" in args
        assert "-safe" in args

    def test_concat_list_contains_segment_paths(
        self, wrapper: FFmpegWrapper, tmp_path: Path
    ) -> None:
        """concat_segments writes each segment path into the concat list file."""
        segments = [tmp_path / f"seg_{i}.mp4" for i in range(2)]
        output = tmp_path / "output.mp4"

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            wrapper.concat_segments(segments, output)

        args: list[str] = mock_run.call_args[0][0]
        i_idx = args.index("-i")
        concat_list_path = Path(args[i_idx + 1])
        assert concat_list_path.exists()
        content = concat_list_path.read_text(encoding="utf-8")
        for seg in segments:
            assert str(seg.resolve()).replace("\\", "/") in content

    def test_raises_value_error_on_empty_list(
        self, wrapper: FFmpegWrapper, tmp_path: Path
    ) -> None:
        """concat_segments raises ValueError when the segment list is empty."""
        with pytest.raises(ValueError, match="empty"):
            wrapper.concat_segments([], tmp_path / "output.mp4")

    def test_raises_ffmpeg_error_on_failure(
        self, wrapper: FFmpegWrapper, tmp_path: Path
    ) -> None:
        """concat_segments raises FFmpegError when ffmpeg exits non-zero."""
        segments = [tmp_path / "seg.mp4"]
        output = tmp_path / "output.mp4"

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=1, stdout="", stderr="concat error"
            )
            with pytest.raises(FFmpegError):
                wrapper.concat_segments(segments, output)


# ---------------------------------------------------------------------------
# extract_thumbnail
# ---------------------------------------------------------------------------


class TestExtractThumbnail:
    def test_generates_thumbnail(self, wrapper: FFmpegWrapper, tmp_path: Path) -> None:
        """extract_thumbnail calls ffmpeg with -vframes 1."""
        source = tmp_path / "video.mp4"
        output = tmp_path / "thumb.jpg"
        source.touch()

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            result = wrapper.extract_thumbnail(source, output, at_sec=5.0)

        args: list[str] = mock_run.call_args[0][0]
        assert "-vframes" in args
        assert "1" in args
        assert result == output

    def test_raises_on_failure(self, wrapper: FFmpegWrapper, tmp_path: Path) -> None:
        """extract_thumbnail raises FFmpegError on ffmpeg failure."""
        source = tmp_path / "video.mp4"
        output = tmp_path / "thumb.jpg"
        source.touch()

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=1, stdout="", stderr="thumbnail error"
            )
            with pytest.raises(FFmpegError):
                wrapper.extract_thumbnail(source, output)


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------


class TestParseFraction:
    def test_integer_fraction(self) -> None:
        assert _parse_fraction("30/1") == pytest.approx(30.0)

    def test_fractional_fps(self) -> None:
        assert _parse_fraction("2997/100") == pytest.approx(29.97)

    def test_plain_float(self) -> None:
        assert _parse_fraction("25.0") == pytest.approx(25.0)

    def test_zero_denominator(self) -> None:
        assert _parse_fraction("1/0") == pytest.approx(0.0)

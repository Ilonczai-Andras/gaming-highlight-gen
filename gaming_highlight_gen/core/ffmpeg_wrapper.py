"""FFmpeg / FFprobe Python abstraction layer."""

from __future__ import annotations

import json
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog

from gaming_highlight_gen.config.global_config import GlobalConfig

log: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)


class FFmpegError(Exception):
    """Raised when an FFmpeg or FFprobe subprocess call fails.

    Attributes:
        returncode: Exit code returned by the subprocess.
        stderr: Captured stderr output from the subprocess.
    """

    def __init__(self, message: str, returncode: int, stderr: str) -> None:
        super().__init__(message)
        self.returncode = returncode
        self.stderr = stderr


@dataclass
class VideoInfo:
    """Metadata extracted from a video file via FFprobe.

    Attributes:
        duration: Total duration in seconds.
        width: Frame width in pixels.
        height: Frame height in pixels.
        fps: Frames per second.
        codec: Video codec name (e.g. "h264").
        audio_codec: Audio codec name, or ``None`` if no audio track.
    """

    duration: float
    width: int
    height: int
    fps: float
    codec: str
    audio_codec: str | None


class FFmpegWrapper:
    """Thin wrapper around the FFmpeg and FFprobe CLI tools.

    Every method captures stderr, logs structured timing data, and raises
    :class:`FFmpegError` on non-zero exit codes.
    """

    def __init__(self, config: GlobalConfig) -> None:
        """Initialise with global config.

        Args:
            config: Application-wide settings (binary paths, codec options …).
        """
        self._config = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_duration(self, video_path: Path) -> float:
        """Return the duration of a video file in seconds via FFprobe.

        Args:
            video_path: Path to the video file.

        Returns:
            Duration in seconds.

        Raises:
            FFmpegError: If FFprobe fails or returns a non-zero exit code.
        """
        return self.get_video_info(video_path).duration

    def get_video_info(self, video_path: Path) -> VideoInfo:
        """Return metadata for a video file via FFprobe.

        Args:
            video_path: Path to the video file.

        Returns:
            :class:`VideoInfo` with duration, resolution, fps, and codec info.

        Raises:
            FFmpegError: If FFprobe fails or the file has no video stream.
        """
        cmd = [
            self._config.ffprobe_binary,
            "-v", "quiet",
            "-print_format", "json",
            "-show_streams",
            "-show_format",
            str(video_path),
        ]

        log.info("ffprobe.start", path=str(video_path))
        start = time.monotonic()
        result = subprocess.run(cmd, capture_output=True, text=True)
        elapsed_ms = int((time.monotonic() - start) * 1000)

        if result.returncode != 0:
            log.error(
                "ffprobe.failed",
                path=str(video_path),
                returncode=result.returncode,
                elapsed_ms=elapsed_ms,
            )
            raise FFmpegError(
                f"ffprobe failed for {video_path}",
                result.returncode,
                result.stderr,
            )

        data: dict[str, Any] = json.loads(result.stdout)
        streams: list[dict[str, Any]] = data.get("streams", [])
        fmt: dict[str, Any] = data.get("format", {})

        video_stream = next(
            (s for s in streams if s.get("codec_type") == "video"), None
        )
        audio_stream = next(
            (s for s in streams if s.get("codec_type") == "audio"), None
        )

        if video_stream is None:
            raise FFmpegError(
                f"No video stream found in {video_path}",
                0,
                "",
            )

        fps = _parse_fraction(str(video_stream.get("r_frame_rate", "0/1")))
        duration = float(
            fmt.get("duration", video_stream.get("duration", 0))
        )

        log.info(
            "ffprobe.done",
            path=str(video_path),
            duration=duration,
            elapsed_ms=elapsed_ms,
        )
        return VideoInfo(
            duration=duration,
            width=int(video_stream.get("width", 0)),
            height=int(video_stream.get("height", 0)),
            fps=fps,
            codec=str(video_stream.get("codec_name", "unknown")),
            audio_codec=(
                str(audio_stream.get("codec_name")) if audio_stream else None
            ),
        )

    def cut_segment(
        self,
        source: Path,
        output: Path,
        start_sec: float,
        end_sec: float,
        *,
        re_encode: bool = False,
    ) -> Path:
        """Cut a segment from a source video file.

        Args:
            source: Source video file.
            output: Destination file path.
            start_sec: Start time in seconds.
            end_sec: End time in seconds.
            re_encode: If ``False`` (default), use stream copy (fast).
                If ``True``, re-encode with the configured codec.

        Returns:
            Path to the output file.

        Raises:
            FFmpegError: If FFmpeg returns a non-zero exit code.
        """
        output.parent.mkdir(parents=True, exist_ok=True)
        duration = end_sec - start_sec

        cmd = [
            self._config.ffmpeg_binary,
            "-y",
            "-ss", str(start_sec),
            "-i", str(source),
            "-t", str(duration),
        ]

        if re_encode:
            cmd += [
                "-c:v", self._config.output_codec,
                "-crf", str(self._config.output_crf),
                "-preset", self._config.output_preset,
                "-c:a", "aac",
            ]
        else:
            cmd += ["-c", "copy"]

        cmd.append(str(output))

        log.info(
            "cut_segment.start",
            source=str(source),
            output=str(output),
            start_sec=start_sec,
            end_sec=end_sec,
            re_encode=re_encode,
        )
        start = time.monotonic()
        result = subprocess.run(cmd, capture_output=True, text=True)
        elapsed_ms = int((time.monotonic() - start) * 1000)

        if result.returncode != 0:
            log.error(
                "cut_segment.failed",
                source=str(source),
                returncode=result.returncode,
                elapsed_ms=elapsed_ms,
            )
            raise FFmpegError(
                f"cut_segment failed for {source}",
                result.returncode,
                result.stderr,
            )

        log.info("cut_segment.done", output=str(output), elapsed_ms=elapsed_ms)
        return output

    def concat_segments(
        self,
        segments: list[Path],
        output: Path,
        *,
        re_encode: bool = True,
    ) -> Path:
        """Concatenate video segments using the FFmpeg concat demuxer.

        Args:
            segments: Ordered list of segment file paths to concatenate.
            output: Destination file path.
            re_encode: If ``True`` (default), re-encode the concatenated output.

        Returns:
            Path to the output file.

        Raises:
            ValueError: If *segments* is empty.
            FFmpegError: If FFmpeg returns a non-zero exit code.
        """
        if not segments:
            raise ValueError("segments list cannot be empty")

        output.parent.mkdir(parents=True, exist_ok=True)

        list_file = self._config.temp_dir / str(uuid.uuid4()) / "concat_list.txt"
        list_file.parent.mkdir(parents=True, exist_ok=True)

        with list_file.open("w", encoding="utf-8") as fh:
            for seg in segments:
                escaped = str(seg.resolve()).replace("\\", "/").replace("'", "\\'")
                fh.write(f"file '{escaped}'\n")

        cmd = [
            self._config.ffmpeg_binary,
            "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(list_file),
        ]

        if re_encode:
            cmd += [
                "-c:v", self._config.output_codec,
                "-crf", str(self._config.output_crf),
                "-preset", self._config.output_preset,
                "-c:a", "aac",
            ]
        else:
            cmd += ["-c", "copy"]

        cmd.append(str(output))

        log.info(
            "concat_segments.start",
            count=len(segments),
            output=str(output),
        )
        start = time.monotonic()
        result = subprocess.run(cmd, capture_output=True, text=True)
        elapsed_ms = int((time.monotonic() - start) * 1000)

        if result.returncode != 0:
            log.error(
                "concat_segments.failed",
                returncode=result.returncode,
                elapsed_ms=elapsed_ms,
            )
            raise FFmpegError(
                "concat_segments failed",
                result.returncode,
                result.stderr,
            )

        log.info(
            "concat_segments.done",
            output=str(output),
            elapsed_ms=elapsed_ms,
        )
        return output

    def extract_thumbnail(
        self,
        source: Path,
        output: Path,
        at_sec: float = 0.0,
    ) -> Path:
        """Generate a JPEG thumbnail from the source video at a given timestamp.

        Args:
            source: Source video file.
            output: Destination JPEG file path.
            at_sec: Timestamp in seconds (default ``0.0``).

        Returns:
            Path to the generated thumbnail file.

        Raises:
            FFmpegError: If FFmpeg returns a non-zero exit code.
        """
        output.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            self._config.ffmpeg_binary,
            "-y",
            "-ss", str(at_sec),
            "-i", str(source),
            "-vframes", "1",
            "-q:v", "2",
            str(output),
        ]

        log.info(
            "extract_thumbnail.start",
            source=str(source),
            at_sec=at_sec,
            output=str(output),
        )
        start = time.monotonic()
        result = subprocess.run(cmd, capture_output=True, text=True)
        elapsed_ms = int((time.monotonic() - start) * 1000)

        if result.returncode != 0:
            log.error(
                "extract_thumbnail.failed",
                source=str(source),
                returncode=result.returncode,
                elapsed_ms=elapsed_ms,
            )
            raise FFmpegError(
                f"extract_thumbnail failed for {source}",
                result.returncode,
                result.stderr,
            )

        log.info(
            "extract_thumbnail.done",
            output=str(output),
            elapsed_ms=elapsed_ms,
        )
        return output

    def extract_audio(self, source: Path, output: Path) -> Path:
        """Extract the audio track from *source* as a WAV file.

        Args:
            source: Source video file.
            output: Destination WAV file path.

        Returns:
            Path to the generated WAV file.

        Raises:
            FFmpegError: If FFmpeg returns a non-zero exit code.
        """
        output.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            self._config.ffmpeg_binary,
            "-y",
            "-i", str(source),
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "44100",
            "-ac", "1",
            str(output),
        ]

        log.info("extract_audio.start", source=str(source), output=str(output))
        start = time.monotonic()
        result = subprocess.run(cmd, capture_output=True, text=True)
        elapsed_ms = int((time.monotonic() - start) * 1000)

        if result.returncode != 0:
            log.error(
                "extract_audio.failed",
                source=str(source),
                returncode=result.returncode,
                elapsed_ms=elapsed_ms,
            )
            raise FFmpegError(
                f"extract_audio failed for {source}",
                result.returncode,
                result.stderr,
            )

        log.info("extract_audio.done", output=str(output), elapsed_ms=elapsed_ms)
        return output


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _parse_fraction(fraction: str) -> float:
    """Parse a fraction string such as ``"30/1"`` or ``"2997/100"`` to float."""
    parts = fraction.split("/")
    if len(parts) == 2:
        num, den = float(parts[0]), float(parts[1])
        return num / den if den != 0 else 0.0
    return float(fraction)

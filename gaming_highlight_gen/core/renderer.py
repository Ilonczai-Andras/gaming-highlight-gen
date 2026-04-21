"""Timeline assembly and highlight video rendering."""

from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import structlog

from gaming_highlight_gen.config.game_config import GameConfig
from gaming_highlight_gen.config.global_config import GlobalConfig
from gaming_highlight_gen.core.ffmpeg_wrapper import FFmpegError, FFmpegWrapper
from gaming_highlight_gen.models.moment import ClipSegment, Moment

log: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)


@dataclass
class RenderResult:
    """Result of a completed render operation.

    Attributes:
        output_path: Path to the rendered highlight video.
        duration_sec: Duration of the output video in seconds.
        thumbnail_path: Path to the generated JPEG thumbnail, or ``None``.
        segments_count: Number of clip segments included in the render.
    """

    output_path: Path
    duration_sec: float
    thumbnail_path: Path | None
    segments_count: int


class Renderer:
    """Assembles clip segments and renders them into a single highlight video."""

    def __init__(self, ffmpeg: FFmpegWrapper, config: GlobalConfig) -> None:
        """Initialise with an FFmpeg wrapper and global config.

        Args:
            ffmpeg: Configured :class:`FFmpegWrapper` instance.
            config: Application-wide settings.
        """
        self._ffmpeg = ffmpeg
        self._config = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render(
        self,
        segments: list[ClipSegment],
        output_path: Path,
        *,
        game_config: GameConfig,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> RenderResult:
        """Render clip segments into a single output video.

        Steps:
        1. Cut each segment into a temp file (stream copy).
        2. Concatenate all temp files (re-encode).
        3. Generate a JPEG thumbnail from the first segment.
        4. Return a :class:`RenderResult`.

        Args:
            segments: Ordered list of clip segments to render.
            output_path: Destination path for the highlight video.
            game_config: Game-specific configuration (unused in Sprint 1,
                reserved for Sprint 3 colour grading).
            progress_callback: Optional ``callback(current, total)`` invoked
                after each segment is cut.

        Returns:
            :class:`RenderResult` with output metadata.

        Raises:
            ValueError: If *segments* is empty.
            FFmpegError: If any FFmpeg call fails.
        """
        if not segments:
            raise ValueError("segments list cannot be empty")

        render_id = str(uuid.uuid4())
        temp_dir = self._config.temp_dir / render_id
        temp_dir.mkdir(parents=True, exist_ok=True)

        log.info(
            "renderer.start",
            segments_count=len(segments),
            output=str(output_path),
            render_id=render_id,
            game=game_config.game_id,
        )

        temp_files: list[Path] = []
        for i, segment in enumerate(segments):
            temp_file = temp_dir / f"segment_{i:04d}.{self._config.output_format}"
            self._ffmpeg.cut_segment(
                source=segment.source_file,
                output=temp_file,
                start_sec=segment.start_sec,
                end_sec=segment.end_sec,
                re_encode=False,
            )
            temp_files.append(temp_file)

            if progress_callback is not None:
                progress_callback(i + 1, len(segments))

        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._ffmpeg.concat_segments(temp_files, output_path, re_encode=True)

        thumbnail_path: Path | None = None
        try:
            thumbnail_path = output_path.with_suffix(".jpg")
            self._ffmpeg.extract_thumbnail(
                source=segments[0].source_file,
                output=thumbnail_path,
                at_sec=segments[0].start_sec,
            )
        except (FFmpegError, OSError, ValueError) as exc:
            log.warning("renderer.thumbnail_failed", error=str(exc))
            thumbnail_path = None

        try:
            duration = self._ffmpeg.get_duration(output_path)
        except (FFmpegError, OSError):
            duration = sum(s.duration_sec for s in segments)

        log.info(
            "renderer.done",
            output=str(output_path),
            duration_sec=duration,
            segments_count=len(segments),
            render_id=render_id,
        )
        return RenderResult(
            output_path=output_path,
            duration_sec=duration,
            thumbnail_path=thumbnail_path,
            segments_count=len(segments),
        )

    def _apply_buffers(
        self,
        moments: list[Moment],
        video_duration: float,
        game_config: GameConfig,
    ) -> list[ClipSegment]:
        """Apply pre/post buffers to moments and merge overlapping segments.

        Each moment is expanded by ``pre_buffer_sec`` before and
        ``post_buffer_sec`` after its timestamp. Resulting intervals that
        are closer than ``min_gap_sec`` are merged into a single segment.
        All intervals are clamped to ``[0, video_duration]`` and capped at
        ``max_clip_duration_sec``.

        Args:
            moments: Detected moments to process.
            video_duration: Total source video duration in seconds.
            game_config: Game-specific buffer and merge settings.

        Returns:
            Ordered list of :class:`ClipSegment` instances ready for rendering.
        """
        if not moments:
            return []

        cfg = game_config.moment

        # Expand each moment with pre/post buffers, clamp to video bounds.
        buffered: list[tuple[float, float, Moment]] = []
        for moment in moments:
            raw_start = moment.start_sec - cfg.pre_buffer_sec
            raw_end = moment.end_sec + cfg.post_buffer_sec
            start = max(0.0, raw_start)
            end = min(video_duration, raw_end)
            # Hard cap on single clip duration.
            if end - start > cfg.max_clip_duration_sec:
                end = start + cfg.max_clip_duration_sec
            buffered.append((start, end, moment))

        # Sort by start time before merging.
        buffered.sort(key=lambda x: x[0])

        # Merge segments whose gap is <= min_gap_sec.
        merged: list[tuple[float, float, Moment]] = []
        cur_start, cur_end, cur_moment = buffered[0]

        for start, end, moment in buffered[1:]:
            if start - cur_end <= cfg.min_gap_sec:
                # Merge: extend the current window, keep higher-score moment.
                cur_end = max(cur_end, end)
                if moment.score > cur_moment.score:
                    cur_moment = moment
            else:
                merged.append((cur_start, cur_end, cur_moment))
                cur_start, cur_end, cur_moment = start, end, moment

        merged.append((cur_start, cur_end, cur_moment))

        return [
            ClipSegment(
                source_file=moment.source_file,
                start_sec=seg_start,
                end_sec=seg_end,
                moment=moment,
            )
            for seg_start, seg_end, moment in merged
        ]

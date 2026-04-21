"""Pipeline orchestrator for the highlight generation workflow."""

from __future__ import annotations

import shutil
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import structlog

from gaming_highlight_gen.config.game_config import GameConfig
from gaming_highlight_gen.config.global_config import GlobalConfig
from gaming_highlight_gen.core.ffmpeg_wrapper import FFmpegWrapper
from gaming_highlight_gen.core.renderer import RenderResult, Renderer
from gaming_highlight_gen.models.moment import Moment

log: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)


@dataclass
class PipelineResult:
    """Result of a completed pipeline run.

    Attributes:
        output_path: Path to the rendered highlight video.
        render_result: Detailed result from the renderer.
        moments_count: Total number of moments used.
        input_files: Source video files that were processed.
    """

    output_path: Path
    render_result: RenderResult
    moments_count: int
    input_files: list[Path] = field(default_factory=list)


class Pipeline:
    """Orchestrates the full highlight generation pipeline.

    Sprint 1 uses dummy, evenly-distributed moments as a placeholder for real
    AI-based detection (Sprint 2).
    """

    def __init__(
        self,
        global_config: GlobalConfig,
        game_config: GameConfig,
    ) -> None:
        """Initialise with global and game-specific configuration.

        Args:
            global_config: Application-wide settings (paths, codecs …).
            game_config: Game-specific buffer and merge settings.
        """
        self._global_config = global_config
        self._game_config = game_config
        self._ffmpeg = FFmpegWrapper(global_config)
        self._renderer = Renderer(self._ffmpeg, global_config)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        input_files: list[Path],
        output_path: Path,
        moments: list[Moment] | None = None,
    ) -> PipelineResult:
        """Execute the full pipeline from input files to a highlight video.

        Steps:
        1. Validate inputs (files exist and are accessible).
        2. Generate dummy moments when *moments* is ``None`` (Sprint 1).
        3. Apply buffers and merge overlapping moments into clip segments.
        4. Render all segments to *output_path*.
        5. Clean up temp files.
        6. Return :class:`PipelineResult`.

        Args:
            input_files: Source video file paths to process.
            output_path: Destination path for the highlight video.
            moments: Pre-detected moments. Pass ``None`` to use the Sprint 1
                dummy generator.

        Returns:
            :class:`PipelineResult` with output metadata.

        Raises:
            FileNotFoundError: If any input file does not exist.
            ValueError: If *input_files* is empty or no segments can be built.
        """
        run_id = str(uuid.uuid4())
        log.info("pipeline.start", run_id=run_id, input_count=len(input_files))

        # Step 1 – validate
        self._validate_inputs(input_files)

        # Step 2 – gather moments
        all_moments: list[Moment] = list(moments) if moments is not None else []
        if moments is None:
            for video_path in input_files:
                all_moments.extend(self._generate_dummy_moments(video_path))

        log.info("pipeline.moments_ready", count=len(all_moments), run_id=run_id)

        # Step 3 – build clip segments per source file
        all_segments = []
        for video_path in input_files:
            file_moments = [m for m in all_moments if m.source_file == video_path]
            if not file_moments:
                continue
            video_duration = self._ffmpeg.get_duration(video_path)
            segments = self._renderer._apply_buffers(
                file_moments,
                video_duration,
                self._game_config,
            )
            all_segments.extend(segments)

        if not all_segments:
            raise ValueError("No segments generated from moments")

        # Step 4 – render
        render_result = self._renderer.render(
            all_segments,
            output_path,
            game_config=self._game_config,
        )

        # Step 5 – cleanup
        self._cleanup_temp()

        log.info(
            "pipeline.done",
            run_id=run_id,
            output=str(output_path),
            moments_count=len(all_moments),
        )
        return PipelineResult(
            output_path=output_path,
            render_result=render_result,
            moments_count=len(all_moments),
            input_files=input_files,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _generate_dummy_moments(
        self,
        video_path: Path,
        count: int = 3,
    ) -> list[Moment]:
        """Generate evenly-distributed dummy moments as a Sprint 1 placeholder.

        Moments are placed at equal intervals across the video duration, each
        lasting 1 second.  This method will be replaced by a real AI-based
        detector in Sprint 2.

        Args:
            video_path: Source video file.
            count: Number of dummy moments to generate.

        Returns:
            List of :class:`Moment` instances with ``event_type="generic"``.
        """
        # TODO(sprint-2): Replace with real AI-based moment detector.
        duration = self._ffmpeg.get_duration(video_path)

        if duration <= 0 or count <= 0:
            return []

        interval = duration / (count + 1)
        moments: list[Moment] = []

        for i in range(count):
            center = interval * (i + 1)
            start = max(0.0, center - 0.5)
            end = min(duration, center + 0.5)

            moments.append(
                Moment(
                    start_sec=start,
                    end_sec=end,
                    score=0.5,
                    source_file=video_path,
                    event_type="generic",
                    metadata={"dummy": True, "index": i},
                )
            )

        log.info(
            "pipeline.dummy_moments_generated",
            video=str(video_path),
            count=len(moments),
        )
        return moments

    def _validate_inputs(self, input_files: list[Path]) -> None:
        """Validate that all input files exist and are regular files.

        Args:
            input_files: Paths to validate.

        Raises:
            ValueError: If *input_files* is empty.
            FileNotFoundError: If any path does not exist.
            ValueError: If any path is not a regular file.
        """
        if not input_files:
            raise ValueError("input_files cannot be empty")

        for path in input_files:
            if not path.exists():
                raise FileNotFoundError(f"Input file not found: {path}")
            if not path.is_file():
                raise ValueError(f"Input path is not a file: {path}")

    def _cleanup_temp(self) -> None:
        """Remove the temp directory and all its contents."""
        temp_dir = self._global_config.temp_dir
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
            log.info("pipeline.cleanup_done", temp_dir=str(temp_dir))

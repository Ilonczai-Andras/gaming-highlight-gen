"""Command-line interface for the gaming highlight generator."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.table import Table

from gaming_highlight_gen.config.game_config import load_game_config
from gaming_highlight_gen.config.global_config import GlobalConfig
from gaming_highlight_gen.core.pipeline import Pipeline
from gaming_highlight_gen.logging_setup import setup_logging

app = typer.Typer(
    name="highlight-gen",
    help="AI-based gaming highlight generator.",
    add_completion=False,
)
console = Console()

# TODO(sprint-4): Use importlib.resources for installed-package compatibility.
_DEFAULT_CONFIG_DIR = Path(__file__).parent.parent / "game_configs"


def _get_config_dir() -> Path:
    """Return the game configs directory, falling back to cwd/game_configs."""
    if _DEFAULT_CONFIG_DIR.exists():
        return _DEFAULT_CONFIG_DIR
    cwd_path = Path.cwd() / "game_configs"
    return cwd_path if cwd_path.exists() else _DEFAULT_CONFIG_DIR


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@app.command("run")
def run_command(
    input_files: list[Path] = typer.Option(
        ...,
        "--input",
        "-i",
        help="Input video file(s). Repeat the flag for multiple files.",
    ),
    game: str = typer.Option(
        "default",
        "--game",
        "-g",
        help="Game ID (e.g. valorant, cs2).",
    ),
    output: Path = typer.Option(
        Path("output/highlight.mp4"),
        "--output",
        "-o",
        help="Output video file path.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose (DEBUG) logging.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="List dummy moments without rendering.",
    ),
) -> None:
    """Generate a gaming highlight video from one or more input files."""
    global_config = GlobalConfig()
    log_level = "DEBUG" if verbose else global_config.log_level
    setup_logging(log_level, global_config.log_format)

    for p in input_files:
        if not p.exists():
            console.print(f"[red]File not found:[/red] {p}")
            raise typer.Exit(code=1)

    try:
        game_config = load_game_config(game, _get_config_dir())
    except FileNotFoundError as exc:
        console.print(f"[red]Config error:[/red] {exc}")
        raise typer.Exit(code=1)

    pipeline = Pipeline(global_config, game_config)

    if dry_run:
        console.print(Panel(f"[bold]Dry run[/bold] – game: [cyan]{game}[/cyan]"))
        for video_path in input_files:
            moments = pipeline._generate_dummy_moments(video_path)
            table = Table(title=f"Dummy moments: {video_path.name}")
            table.add_column("Index", style="cyan", no_wrap=True)
            table.add_column("Start (s)", style="green")
            table.add_column("End (s)", style="green")
            table.add_column("Score", style="magenta")
            table.add_column("Event Type", style="yellow")
            for i, moment in enumerate(moments):
                table.add_row(
                    str(i),
                    f"{moment.start_sec:.2f}",
                    f"{moment.end_sec:.2f}",
                    f"{moment.score:.2f}",
                    moment.event_type,
                )
            console.print(table)
        return

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Analyzing…", total=3)
        try:
            progress.update(task, description="Analyzing…", advance=1)
            result = pipeline.run(input_files=list(input_files), output_path=output)
            progress.update(task, description="Rendering…", advance=1)
            progress.update(task, description="Done!", advance=1)
        except FileNotFoundError as exc:
            console.print(f"[red]File not found:[/red] {exc}")
            raise typer.Exit(code=1)
        except ValueError as exc:
            console.print(f"[red]Error:[/red] {exc}")
            raise typer.Exit(code=1)

    out_size_mb = result.render_result.output_path.stat().st_size / (1024 * 1024)
    console.print(
        Panel(
            f"[bold green]Highlight generated![/bold green]\n"
            f"File:     [cyan]{result.render_result.output_path}[/cyan]\n"
            f"Size:     {out_size_mb:.1f} MB\n"
            f"Duration: {result.render_result.duration_sec:.1f}s\n"
            f"Segments: {result.render_result.segments_count}",
            title="Result",
        )
    )


@app.command("info")
def info_command(
    input_file: Path = typer.Option(
        ...,
        "--input",
        "-i",
        help="Input video file.",
    ),
) -> None:
    """Display metadata for a video file."""
    global_config = GlobalConfig()
    setup_logging(global_config.log_level, global_config.log_format)

    from gaming_highlight_gen.core.ffmpeg_wrapper import FFmpegWrapper

    ffmpeg = FFmpegWrapper(global_config)
    try:
        info = ffmpeg.get_video_info(input_file)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1)

    table = Table(title=f"Video Info: {input_file.name}")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Duration", f"{info.duration:.2f}s")
    table.add_row("Resolution", f"{info.width}x{info.height}")
    table.add_row("FPS", f"{info.fps:.2f}")
    table.add_row("Video Codec", info.codec)
    table.add_row("Audio Codec", info.audio_codec or "N/A")
    console.print(table)


# ---------------------------------------------------------------------------
# Config sub-commands
# ---------------------------------------------------------------------------

config_app = typer.Typer(help="Configuration management commands.")
app.add_typer(config_app, name="config")


@config_app.command("validate")
def config_validate_command(
    game: str = typer.Option(
        "default",
        "--game",
        "-g",
        help="Game ID to validate.",
    ),
) -> None:
    """Validate a game configuration file and print its resolved values."""
    try:
        game_config = load_game_config(game, _get_config_dir())
        console.print(
            f"[green]Config valid:[/green] {game_config.game_id}"
            f" ({game_config.display_name})"
        )
        console.print(game_config.model_dump_json(indent=2))
    except FileNotFoundError as exc:
        console.print(f"[red]Config not found:[/red] {exc}")
        raise typer.Exit(code=1)
    except ValueError as exc:
        console.print(f"[red]Config error:[/red] {exc}")
        raise typer.Exit(code=1)


@app.command("detect")
def detect_command(
    input_files: list[Path] = typer.Option(
        ...,
        "--input",
        "-i",
        help="Input video file(s). Repeat the flag for multiple files.",
    ),
    game: str = typer.Option(
        "default",
        "--game",
        "-g",
        help="Game ID (e.g. valorant, r6).",
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Write JSON results to this file instead of stdout.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose (DEBUG) logging.",
    ),
) -> None:
    """Run moment detection and print results as JSON.

    Runs the detector pipeline on each input file and outputs all discovered
    moments as a JSON array.  No video is rendered.
    """
    import json

    global_config = GlobalConfig()
    log_level = "DEBUG" if verbose else global_config.log_level
    setup_logging(log_level, global_config.log_format)

    for p in input_files:
        if not p.exists():
            console.print(f"[red]File not found:[/red] {p}")
            raise typer.Exit(code=1)

    try:
        game_config = load_game_config(game, _get_config_dir())
    except FileNotFoundError as exc:
        console.print(f"[red]Config error:[/red] {exc}")
        raise typer.Exit(code=1)

    pipeline = Pipeline(global_config, game_config)

    try:
        moments = pipeline.detect_only(list(input_files))
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]Detection error:[/red] {exc}")
        raise typer.Exit(code=1)

    payload = [
        {
            "source_file": str(m.source_file),
            "start_sec": m.start_sec,
            "end_sec": m.end_sec,
            "score": m.score,
            "event_type": m.event_type,
            "detector_breakdown": m.detector_breakdown,
        }
        for m in moments
    ]

    json_output = json.dumps(payload, indent=2)

    if output is not None:
        output.write_text(json_output, encoding="utf-8")
        console.print(f"[green]Moments written to:[/green] {output}  ({len(moments)} moments)")
    else:
        typer.echo(json_output)


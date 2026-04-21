# Gaming Highlight Generator

Modular, AI-based gaming highlight generator. Automatically cuts the best
moments from raw gameplay videos into a single highlight clip.

> **Sprint 1** – core pipeline with dummy moment detection.
> Real AI detection lands in Sprint 2.

---

## Quickstart

### Prerequisites

* Python 3.12+
* [Poetry](https://python-poetry.org/) ≥ 1.8
* FFmpeg / FFprobe available on `PATH`

### Installation

```bash
git clone <repo-url>
cd gaming-highlight-gen

poetry install
poetry shell
```

### Basic usage

```bash
# Generate a highlight from a single video (dummy moments, Sprint 1)
highlight-gen run --input gameplay.mp4 --game r6

# Multiple input files
highlight-gen run --input clip1.mp4 --input clip2.mp4 --game cs2 --output out.mp4

# Preview detected moments without rendering
highlight-gen run --dry-run --input gameplay.mp4

# Verbose (DEBUG) logging
highlight-gen run --input gameplay.mp4 --verbose

# Show video metadata
highlight-gen info --input gameplay.mp4

# Validate a game config
highlight-gen config validate --game r6
```

---

## Docker usage

```bash
# Build the image
docker build -t highlight-gen .

# Place input files under ./input/, then run:
docker-compose run highlight-gen run --input /app/input/gameplay.mp4 --game r6

# Output appears in ./output/
```

Or pass extra environment variables directly:

```bash
docker run --rm \
  -v "$PWD/input:/app/input" \
  -v "$PWD/output:/app/output" \
  -e HLG_LOG_FORMAT=console \
  highlight-gen run --input /app/input/gameplay.mp4 --game r6
```

---

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `HLG_FFMPEG_BINARY` | `ffmpeg` | Path to the FFmpeg binary |
| `HLG_FFPROBE_BINARY` | `ffprobe` | Path to the FFprobe binary |
| `HLG_HW_ACCELERATION` | `none` | HW accel profile: `none` / `nvenc` / `vaapi` |
| `HLG_OUTPUT_DIR` | `output` | Directory for rendered output files |
| `HLG_TEMP_DIR` | `.tmp` | Temporary working directory |
| `HLG_OUTPUT_FORMAT` | `mp4` | Container format |
| `HLG_OUTPUT_CODEC` | `libx264` | Video codec |
| `HLG_OUTPUT_CRF` | `23` | CRF quality (lower = better quality / larger file) |
| `HLG_OUTPUT_PRESET` | `fast` | FFmpeg encoding speed preset |
| `HLG_LOG_LEVEL` | `INFO` | Log level: `DEBUG` / `INFO` / `WARNING` / `ERROR` |
| `HLG_LOG_FORMAT` | `json` | Log renderer: `json` (production) / `console` (development) |

Create a `.env` file in the project root to override defaults locally:

```env
HLG_LOG_FORMAT=console
HLG_OUTPUT_DIR=/path/to/output
HLG_OUTPUT_CRF=18
```

---

## Game configs

Per-game settings live in `game_configs/`. Add a new `{game_id}.yaml` that
overrides only the fields that differ from `default.yaml`:

```yaml
# game_configs/cs2.yaml
game_id: cs2
display_name: "Counter-Strike 2"
moment:
  pre_buffer_sec: 3.5
```

At runtime `load_game_config("cs2", ...)` deep-merges `cs2.yaml` over
`default.yaml`.

---

## Development

```bash
# Run full test suite with coverage report
pytest

# Lint
ruff check .

# Auto-format
ruff format .

# Type-check
mypy gaming_highlight_gen/

# Install pre-commit hooks (runs ruff on every commit)
pre-commit install
```

---

## Project structure

```
gaming_highlight_gen/
├── cli.py                  # Typer CLI (highlight-gen)
├── logging_setup.py        # structlog configuration
├── config/
│   ├── global_config.py    # Pydantic Settings (env + .env)
│   └── game_config.py      # YAML game configs + deep merge
├── core/
│   ├── pipeline.py         # Orchestrator
│   ├── ffmpeg_wrapper.py   # subprocess-based FFmpeg abstraction
│   └── renderer.py         # Clip assembly & render
└── models/
    └── moment.py           # Moment / ClipSegment dataclasses
```

---

## Sprint roadmap

| Sprint | Feature |
|---|---|
| **1** (current) | Core pipeline, dummy moment detection, CLI, Docker, CI |
| 2 | AI-based moment detection (audio / video analysis) |
| 3 | Subtitles via Whisper STT + colour grading |
| 4 | MCP Server / GitHub Copilot agent integration |
| 5 | Per-game ML models + extended YAML config |
| 6 | GPU hardware acceleration (NVENC / VAAPI) |

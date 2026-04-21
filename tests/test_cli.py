"""Tests for the gaming_highlight_gen CLI (gaming_highlight_gen.cli)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from gaming_highlight_gen.cli import app

runner = CliRunner()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _probe_json(duration: float = 60.0) -> str:
    return json.dumps(
        {
            "streams": [
                {
                    "codec_type": "video",
                    "codec_name": "h264",
                    "width": 1920,
                    "height": 1080,
                    "r_frame_rate": "30/1",
                    "duration": str(duration),
                }
            ],
            "format": {"duration": str(duration)},
        }
    )


# ---------------------------------------------------------------------------
# config validate
# ---------------------------------------------------------------------------


class TestConfigValidate:
    def test_valid_default_game_exits_zero(self) -> None:
        """config validate --game default succeeds for the built-in default config."""
        result = runner.invoke(app, ["config", "validate", "--game", "default"])
        assert result.exit_code == 0
        assert "default" in result.output.lower()

    def test_valid_r6_game_exits_zero(self) -> None:
        """config validate --game r6 resolves the merged r6 config."""
        result = runner.invoke(app, ["config", "validate", "--game", "r6"])
        assert result.exit_code == 0
        assert "r6" in result.output.lower()

    def test_unknown_game_exits_one(self) -> None:
        """config validate --game unknown exits 1 and prints an error."""
        result = runner.invoke(app, ["config", "validate", "--game", "unknowngame99"])
        assert result.exit_code == 1
        assert "not found" in result.output.lower() or "error" in result.output.lower()


# ---------------------------------------------------------------------------
# info
# ---------------------------------------------------------------------------


class TestInfoCommand:
    def test_info_prints_video_metadata(self, tmp_path: Path) -> None:
        """info --input <file> prints duration, resolution, and codec."""
        video = tmp_path / "video.mp4"
        video.touch()

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0, stdout=_probe_json(90.0), stderr=""
            )
            result = runner.invoke(app, ["info", "--input", str(video)])

        assert result.exit_code == 0
        assert "90.00" in result.output
        assert "h264" in result.output

    def test_info_exits_one_on_ffprobe_failure(self, tmp_path: Path) -> None:
        """info --input <file> exits 1 when FFprobe returns an error."""
        video = tmp_path / "video.mp4"
        video.touch()

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=1, stdout="", stderr="no such file"
            )
            result = runner.invoke(app, ["info", "--input", str(video)])

        assert result.exit_code == 1


# ---------------------------------------------------------------------------
# run --dry-run
# ---------------------------------------------------------------------------


class TestRunDryRun:
    def test_dry_run_lists_moments(self, tmp_path: Path) -> None:
        """run --dry-run --input <file> prints a moments table and exits 0."""
        video = tmp_path / "video.mp4"
        video.touch()

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0, stdout=_probe_json(120.0), stderr=""
            )
            result = runner.invoke(
                app,
                ["run", "--dry-run", "--input", str(video), "--game", "default"],
            )

        assert result.exit_code == 0
        assert "generic" in result.output.lower()

    def test_dry_run_missing_input_exits_one(self, tmp_path: Path) -> None:
        """run --dry-run exits 1 when the input file does not exist."""
        result = runner.invoke(
            app,
            ["run", "--dry-run", "--input", str(tmp_path / "nonexistent.mp4")],
        )
        assert result.exit_code == 1

    def test_dry_run_unknown_game_exits_one(self, tmp_path: Path) -> None:
        """run --dry-run exits 1 when the game config is missing."""
        video = tmp_path / "video.mp4"
        video.touch()
        result = runner.invoke(
            app,
            ["run", "--dry-run", "--input", str(video), "--game", "unknowngame99"],
        )
        assert result.exit_code == 1

"""Tests for gaming_highlight_gen.config (GlobalConfig and GameConfig)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from gaming_highlight_gen.config.game_config import (
    GameConfig,
    MomentConfig,
    _deep_merge,
    load_game_config,
)
from gaming_highlight_gen.config.global_config import GlobalConfig
from gaming_highlight_gen.models.moment import Moment


# ---------------------------------------------------------------------------
# GlobalConfig
# ---------------------------------------------------------------------------


class TestGlobalConfig:
    def test_default_values(self) -> None:
        """GlobalConfig uses expected built-in defaults."""
        config = GlobalConfig()
        assert config.ffmpeg_binary == "ffmpeg"
        assert config.ffprobe_binary == "ffprobe"
        assert config.output_format == "mp4"
        assert config.output_codec == "libx264"
        assert config.output_crf == 23
        assert config.output_preset == "fast"
        assert config.log_level == "INFO"
        assert config.log_format == "json"

    def test_env_prefix_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """GlobalConfig reads HLG_* environment variables."""
        monkeypatch.setenv("HLG_FFMPEG_BINARY", "/usr/local/bin/ffmpeg")
        monkeypatch.setenv("HLG_OUTPUT_CRF", "18")

        config = GlobalConfig()

        assert config.ffmpeg_binary == "/usr/local/bin/ffmpeg"
        assert config.output_crf == 18

    def test_env_file_loading(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """GlobalConfig loads values from an explicitly provided .env file."""
        env_file = tmp_path / "test.env"
        env_file.write_text("HLG_LOG_LEVEL=DEBUG\nHLG_OUTPUT_PRESET=slow\n")

        # pydantic-settings v2 accepts _env_file as a constructor kwarg.
        config = GlobalConfig(_env_file=str(env_file))  # type: ignore[call-arg]

        assert config.log_level == "DEBUG"
        assert config.output_preset == "slow"

    def test_output_dir_is_path(self) -> None:
        """output_dir is coerced to a Path object."""
        config = GlobalConfig()
        assert isinstance(config.output_dir, Path)


# ---------------------------------------------------------------------------
# GameConfig / load_game_config
# ---------------------------------------------------------------------------


def _write_yaml(path: Path, data: dict) -> None:  # type: ignore[type-arg]
    path.write_text(yaml.dump(data), encoding="utf-8")


def _default_yaml_data() -> dict:  # type: ignore[type-arg]
    return {
        "game_id": "default",
        "display_name": "Default",
        "moment": {
            "pre_buffer_sec": 3.0,
            "post_buffer_sec": 2.0,
            "min_gap_sec": 1.0,
            "max_clip_duration_sec": 15.0,
        },
    }


class TestLoadGameConfig:
    def test_loads_default_config(self, tmp_path: Path) -> None:
        """load_game_config returns a valid GameConfig for game_id='default'."""
        _write_yaml(tmp_path / "default.yaml", _default_yaml_data())

        config = load_game_config("default", tmp_path)

        assert config.game_id == "default"
        assert config.display_name == "Default"
        assert config.moment.pre_buffer_sec == pytest.approx(3.0)
        assert config.moment.post_buffer_sec == pytest.approx(2.0)

    def test_merges_game_specific_yaml_over_default(self, tmp_path: Path) -> None:
        """Game-specific YAML values override their default counterparts."""
        _write_yaml(tmp_path / "default.yaml", _default_yaml_data())
        _write_yaml(
            tmp_path / "valorant.yaml",
            {
                "game_id": "valorant",
                "display_name": "Valorant",
                "moment": {"pre_buffer_sec": 4.0, "post_buffer_sec": 3.0},
            },
        )

        config = load_game_config("valorant", tmp_path)

        assert config.game_id == "valorant"
        assert config.moment.pre_buffer_sec == pytest.approx(4.0)
        assert config.moment.post_buffer_sec == pytest.approx(3.0)
        # Inherited from default.yaml
        assert config.moment.min_gap_sec == pytest.approx(1.0)
        assert config.moment.max_clip_duration_sec == pytest.approx(15.0)

    def test_missing_default_raises_file_not_found(self, tmp_path: Path) -> None:
        """load_game_config raises FileNotFoundError when default.yaml is absent."""
        with pytest.raises(FileNotFoundError, match="default"):
            load_game_config("valorant", tmp_path)

    def test_missing_game_config_raises_file_not_found(self, tmp_path: Path) -> None:
        """load_game_config raises FileNotFoundError for an unknown game_id."""
        _write_yaml(tmp_path / "default.yaml", _default_yaml_data())

        with pytest.raises(FileNotFoundError, match="nonexistent"):
            load_game_config("nonexistent", tmp_path)

    def test_game_config_model_is_valid(self, tmp_path: Path) -> None:
        """Loaded GameConfig is a proper Pydantic model instance."""
        _write_yaml(tmp_path / "default.yaml", _default_yaml_data())

        config = load_game_config("default", tmp_path)

        assert isinstance(config, GameConfig)
        assert isinstance(config.moment, MomentConfig)


# ---------------------------------------------------------------------------
# _deep_merge helper
# ---------------------------------------------------------------------------


class TestDeepMerge:
    def test_override_scalar_value(self) -> None:
        base = {"a": 1, "b": 2}
        override = {"b": 99}
        assert _deep_merge(base, override) == {"a": 1, "b": 99}

    def test_nested_dict_is_merged_recursively(self) -> None:
        base = {"moment": {"pre": 3.0, "post": 2.0}}
        override = {"moment": {"pre": 4.0}}
        result = _deep_merge(base, override)
        assert result["moment"]["pre"] == pytest.approx(4.0)
        assert result["moment"]["post"] == pytest.approx(2.0)

    def test_new_keys_are_added(self) -> None:
        base = {"a": 1}
        override = {"b": 2}
        assert _deep_merge(base, override) == {"a": 1, "b": 2}

    def test_base_is_not_mutated(self) -> None:
        base = {"x": {"y": 1}}
        override = {"x": {"y": 2}}
        _deep_merge(base, override)
        assert base["x"]["y"] == 1


# ---------------------------------------------------------------------------
# Moment validation
# ---------------------------------------------------------------------------


class TestMomentValidation:
    def test_valid_moment(self, tmp_path: Path) -> None:
        """A correctly constructed Moment raises no errors."""
        m = Moment(
            start_sec=5.0,
            end_sec=10.0,
            score=0.75,
            source_file=tmp_path / "v.mp4",
        )
        assert m.start_sec == pytest.approx(5.0)
        assert m.end_sec == pytest.approx(10.0)
        assert m.score == pytest.approx(0.75)

    def test_negative_start_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="start_sec"):
            Moment(
                start_sec=-1.0,
                end_sec=5.0,
                score=0.5,
                source_file=tmp_path / "v.mp4",
            )

    def test_end_before_start_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="end_sec"):
            Moment(
                start_sec=10.0,
                end_sec=9.0,
                score=0.5,
                source_file=tmp_path / "v.mp4",
            )

    def test_score_out_of_range_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="score"):
            Moment(
                start_sec=0.0,
                end_sec=5.0,
                score=1.5,
                source_file=tmp_path / "v.mp4",
            )

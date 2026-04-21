"""Global application configuration via Pydantic Settings."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class GlobalConfig(BaseSettings):
    """Global configuration for the highlight generator.

    All fields can be overridden via environment variables with the
    ``HLG_`` prefix (e.g. ``HLG_OUTPUT_CRF=18``) or via a ``.env`` file.
    """

    # FFmpeg
    ffmpeg_binary: str = "ffmpeg"
    ffprobe_binary: str = "ffprobe"
    hw_acceleration: str = "none"  # "none" | "nvenc" | "vaapi"

    # Pipeline defaults
    output_dir: Path = Path("output")
    temp_dir: Path = Path(".tmp")
    output_format: str = "mp4"
    output_codec: str = "libx264"
    output_crf: int = 23
    output_preset: str = "fast"

    # Logging
    log_level: str = "INFO"
    log_format: str = "json"  # "json" | "console"

    model_config = SettingsConfigDict(
        env_prefix="HLG_",
        env_file=".env",
        env_file_encoding="utf-8",
    )

# Test Fixtures

This directory contains helpers for generating synthetic test fixtures.

## `generate_fixtures.py`

Two factory functions:

| Function | Description |
|---|---|
| `generate_sine_audio_with_spikes(path, ...)` | 16-bit mono WAV with quiet 440 Hz background and loud amplitude spikes at configurable timestamps. |
| `generate_test_video_with_motion(path, ...)` | Small MP4 (default 64×64 @ 30 fps) with near-static frames and sudden white-flash motion events at configurable timestamps. |

Both functions accept a `path` argument so callers can place fixtures in `tmp_path` (pytest's built-in temp directory fixture).

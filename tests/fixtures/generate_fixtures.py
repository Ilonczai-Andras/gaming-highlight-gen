"""Utilities for generating synthetic test fixtures."""

from __future__ import annotations

import struct
import wave
from pathlib import Path


def generate_sine_audio_with_spikes(
    path: Path,
    duration_sec: float = 10.0,
    sample_rate: int = 44100,
    spike_times: list[float] | None = None,
) -> Path:
    """Write a WAV file with a quiet background and loud spikes.

    Args:
        path: Destination WAV file path.
        duration_sec: Total audio length in seconds.
        sample_rate: Sample rate in Hz.
        spike_times: Timestamps (in seconds) for 0.1-second loud spikes.
            Defaults to spikes at 2.0 s, 5.0 s and 8.0 s.

    Returns:
        Path to the written WAV file.
    """
    if spike_times is None:
        spike_times = [2.0, 5.0, 8.0]

    n_samples = int(duration_sec * sample_rate)
    samples = [0] * n_samples

    # Background: low-amplitude sine at 440 Hz
    import math

    for i in range(n_samples):
        t = i / sample_rate
        samples[i] = int(1000 * math.sin(2 * math.pi * 440 * t))

    # Spikes: 0.1 s of 20 000-amplitude noise
    spike_width = int(0.1 * sample_rate)
    for spike_t in spike_times:
        spike_start = int(spike_t * sample_rate)
        for j in range(spike_width):
            idx = spike_start + j
            if idx < n_samples:
                import random

                samples[idx] = random.randint(18000, 20000)

    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        data = struct.pack(f"<{n_samples}h", *samples)
        wf.writeframes(data)

    return path


def generate_test_video_with_motion(
    path: Path,
    duration_sec: float = 10.0,
    fps: int = 30,
    width: int = 64,
    height: int = 64,
    motion_times: list[float] | None = None,
) -> Path:
    """Write a minimal MP4 with simulated motion spikes using OpenCV.

    Args:
        path: Destination MP4 file path.
        duration_sec: Total video length in seconds.
        fps: Frame rate.
        width: Frame width in pixels.
        height: Frame height in pixels.
        motion_times: Timestamps (in seconds) for sharp visual changes.
            Defaults to changes at 2.0 s, 5.0 s and 8.0 s.

    Returns:
        Path to the written video file.
    """
    import cv2
    import numpy as np

    if motion_times is None:
        motion_times = [2.0, 5.0, 8.0]

    motion_set = set(int(t * fps) for t in motion_times)
    total_frames = int(duration_sec * fps)

    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))

    rng = np.random.default_rng(seed=42)
    base = rng.integers(80, 120, (height, width, 3), dtype=np.uint8)

    for frame_idx in range(total_frames):
        if frame_idx in motion_set:
            # Sudden bright flash
            frame = np.full((height, width, 3), 255, dtype=np.uint8)
        else:
            noise = rng.integers(-5, 5, (height, width, 3), dtype=np.int16)
            frame = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        writer.write(frame)

    writer.release()
    return path

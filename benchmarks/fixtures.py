"""Synthetic test inputs for the benchmark harness.

These are intentionally deterministic (seeded) so benchmark numbers are
comparable across runs and machines.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image


def make_synthetic_image(
    out: Path,
    width: int = 4000,
    height: int = 3000,
    seed: int = 42,
) -> Path:
    """Generate a deterministic noisy RGB PNG.

    If `out` is a directory, file is saved as `synthetic_{width}x{height}.png`
    inside it. If `out` looks like a file path, save there directly.
    """
    if out.is_dir():
        out = out / f"synthetic_{width}x{height}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    # Smooth-ish noise: random + low-frequency gradient → exercises Sobel
    base = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    yy = np.linspace(0, 255, height, dtype=np.uint8)[:, None, None]
    base = ((base.astype(np.uint16) + yy) // 2).astype(np.uint8)
    Image.fromarray(base, mode="RGB").save(out)
    return out


def make_synthetic_video(
    out_dir: Path,
    width: int = 1280,
    height: int = 720,
    duration_s: int = 10,
    fps: int = 24,
    seed: int = 1,
) -> Path:
    """Generate a deterministic MP4 by piping numpy frames through ffmpeg.

    Frames are noise with a moving brightness band so motion is visible.
    """
    if shutil.which("ffmpeg") is None:
        raise FileNotFoundError("ffmpeg required for synthetic video fixture")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"synthetic_{width}x{height}_{duration_s}s_{fps}fps.mp4"

    n_frames = duration_s * fps
    rng = np.random.default_rng(seed)

    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-pixel_format", "rgb24",
        "-video_size", f"{width}x{height}",
        "-framerate", str(fps),
        "-i", "-",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        str(out_path),
    ]

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
    assert proc.stdin is not None
    try:
        for i in range(n_frames):
            frame = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
            # Moving brightness band
            band_y = int((i / n_frames) * height)
            frame[max(band_y - 20, 0):band_y + 20] = np.clip(
                frame[max(band_y - 20, 0):band_y + 20].astype(np.int16) + 60, 0, 255
            ).astype(np.uint8)
            try:
                proc.stdin.write(frame.tobytes())
            except BrokenPipeError:
                break
    finally:
        proc.stdin.close()
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"ffmpeg returned {rc} while encoding synthetic video")
    return out_path

"""Reference scenarios for benchmarks.

S1/S2/S3 match the targets agreed in the spec
(`docs/superpowers/specs/2026-05-29-perf-profile-and-easy-wins-design.md`).
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ImageScenario:
    id: str
    image_width: int
    image_height: int
    out_width: int          # ASCII char-grid width
    font_size: int
    layers: tuple[str, ...] = ("brightness", "sobel_x", "sobel_y", "diagonal")
    threshold: float = 0.15
    jitter: int = 1


@dataclass(frozen=True)
class VideoScenario:
    id: str
    video_width: int
    video_height: int
    duration_s: int
    source_fps: int
    out_fps: int
    out_width: int
    font_size: int
    layers: tuple[str, ...] = ("brightness", "sobel_x", "sobel_y", "diagonal")
    threshold: float = 0.15
    jitter: int = 1


S1 = ImageScenario(id="s1", image_width=4000, image_height=3000, out_width=140, font_size=12)
S2 = ImageScenario(id="s2", image_width=4000, image_height=3000, out_width=100, font_size=12)
S3 = VideoScenario(
    id="s3", video_width=1280, video_height=720, duration_s=10,
    source_fps=24, out_fps=10, out_width=100, font_size=12,
)

ALL_IMAGE: tuple[ImageScenario, ...] = (S1, S2)
ALL_VIDEO: tuple[VideoScenario, ...] = (S3,)

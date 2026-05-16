from __future__ import annotations

import shutil
import subprocess
import tempfile
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from PIL import Image

from ascii_combinator.bg_mode import BgMode, SoftBgConfig
from ascii_combinator.compositor import Compositor
from ascii_combinator.layers.brightness import BrightnessLayer
from ascii_combinator.layers.diagonal import DiagonalLayer
from ascii_combinator.layers.sobel_x import SobelXLayer
from ascii_combinator.layers.sobel_y import SobelYLayer
from ascii_combinator.profiles.monochrome import MonochromeProfile
from ascii_combinator.renderer import Renderer

_LAYER_REGISTRY = {
    "brightness": BrightnessLayer,
    "sobel_x": SobelXLayer,
    "sobel_y": SobelYLayer,
    "diagonal": DiagonalLayer,
}

_PROFILE_REGISTRY = {
    "monochrome": MonochromeProfile,
}


@dataclass
class VideoConfig:
    width: Optional[int]
    profile_name: str
    layer_names: list
    jitter: int
    threshold: float
    font_size: int
    bg_mode: BgMode
    soft_cfg: Optional[SoftBgConfig]


class FrameProcessor:
    def process(self, frame_in: Path, frame_out: Path, config: VideoConfig) -> Path:
        image = Image.open(frame_in)
        font_size = config.font_size
        cell_w = max(int(font_size * 0.6), 1)
        cell_h = max(font_size, 1)

        num_cols = config.width if config.width is not None else max(image.width // cell_w, 10)
        num_rows = max(1, int(num_cols * cell_w / cell_h * image.height / image.width))

        layers = [_LAYER_REGISTRY[n](threshold=config.threshold) for n in config.layer_names]
        charmap_list = [layer.process(image, num_rows, num_cols) for layer in layers]
        charmap = Compositor().composite(
            charmap_list, mask=None, bg_mode=config.bg_mode, soft_cfg=config.soft_cfg
        )
        profile = _PROFILE_REGISTRY[config.profile_name]()
        result = Renderer().render(charmap, profile, font_size=font_size, jitter=config.jitter)
        result.save(frame_out)
        return frame_out


class FrameExtractor:
    def extract(self, video_path: Path, tmp_dir: Path, fps, frame_step) -> list:
        raise NotImplementedError


class VideoAssembler:
    def assemble_mp4(self, frames_dir: Path, output: Path, fps: float) -> None:
        raise NotImplementedError

    def assemble_gif(self, mp4_path: Path, output: Path, fps: float) -> None:
        raise NotImplementedError


def _worker(args):
    frame_in, frame_out, config = args
    return FrameProcessor().process(frame_in, frame_out, config)


class VideoProcessor:
    def process(self, video_path, output, config, fps, frame_step, workers, preview, make_gif, gif_fps):
        raise NotImplementedError

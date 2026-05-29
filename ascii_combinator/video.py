import shutil
import subprocess
import tempfile
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from tqdm import tqdm
from pathlib import Path

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
    width: int | None
    profile_name: str
    layer_names: list[str]
    jitter: int
    threshold: float
    font_size: int
    bg_mode: BgMode
    soft_cfg: SoftBgConfig | None


class FrameProcessor:
    def process(self, frame_in: Path, frame_out: Path, config: VideoConfig) -> Path:
        image = Image.open(frame_in)
        font_size = config.font_size
        cell_w = max(int(font_size * 0.6), 1)
        cell_h = max(font_size, 1)

        num_cols = config.width if config.width is not None else max(image.width // cell_w, 10)
        num_rows = max(1, int(num_cols * cell_w / cell_h * image.height / image.width))

        from ascii_combinator.layers.base import LayerInputs
        layer_inputs = LayerInputs.from_image(image)
        layers = [_LAYER_REGISTRY[n](threshold=config.threshold) for n in config.layer_names]
        charmap_list = [
            layer.process(image, num_rows, num_cols, inputs=layer_inputs)
            for layer in layers
        ]
        charmap = Compositor().composite(
            charmap_list, mask=None, bg_mode=config.bg_mode, soft_cfg=config.soft_cfg
        )
        profile = _PROFILE_REGISTRY[config.profile_name]()
        result = Renderer().render(charmap, profile, font_size=font_size, jitter=config.jitter)
        result.save(frame_out)
        return frame_out


class FrameExtractor:
    def extract(
        self,
        video_path: Path,
        tmp_dir: Path,
        fps: float | None,
        frame_step: int | None,
    ) -> list[Path]:
        if shutil.which("ffmpeg") is None:
            raise FileNotFoundError(
                "ffmpeg not found. Install with: apt install ffmpeg"
            )

        output_pattern = str(tmp_dir / "frame_%06d.png")

        if frame_step is not None:
            vf = f"select='not(mod(n\\,{frame_step}))',setpts=N/FRAME_RATE/TB"
            cmd = [
                "ffmpeg", "-i", str(video_path),
                "-vf", vf, "-vsync", "vfr",
                output_pattern,
            ]
        else:
            cmd = [
                "ffmpeg", "-i", str(video_path),
                "-vf", f"fps={fps}",
                output_pattern,
            ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {result.stderr}")

        frames = sorted(tmp_dir.glob("frame_*.png"))
        if not frames:
            raise RuntimeError(f"No frames extracted from {video_path}")
        return frames


class VideoAssembler:
    def assemble_mp4(self, frames_dir: Path, output: Path, fps: float) -> None:
        pattern = str(frames_dir / "frame_%06d.png")
        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", pattern,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            str(output),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed assembling MP4: {result.stderr}")

    def assemble_gif(self, mp4_path: Path, output: Path, fps: float) -> None:
        with tempfile.TemporaryDirectory() as tmp_palette_dir:
            palette = Path(tmp_palette_dir) / "_palette.png"
            r1 = subprocess.run(
                ["ffmpeg", "-y", "-i", str(mp4_path),
                 "-vf", f"fps={fps},palettegen", str(palette)],
                capture_output=True, text=True,
            )
            if r1.returncode != 0:
                raise RuntimeError(f"ffmpeg palettegen failed: {r1.stderr}")

            r2 = subprocess.run(
                ["ffmpeg", "-y", "-i", str(mp4_path), "-i", str(palette),
                 "-vf", f"fps={fps},paletteuse", str(output)],
                capture_output=True, text=True,
            )
            if r2.returncode != 0:
                raise RuntimeError(f"ffmpeg GIF assembly failed: {r2.stderr}")


def _worker(args):
    frame_in, frame_out, config = args
    return FrameProcessor().process(frame_in, frame_out, config)


class VideoProcessor:
    def process(
        self,
        video_path: Path,
        output: Path,
        config: VideoConfig,
        fps: float,
        frame_step: int | None,
        workers: int,
        preview: bool,
        make_gif: bool,
        gif_fps: float,
    ) -> None:
        extractor = FrameExtractor()
        assembler = VideoAssembler()

        with tempfile.TemporaryDirectory() as tmp_in_s, \
             tempfile.TemporaryDirectory() as tmp_out_s:
            tmp_in = Path(tmp_in_s)
            tmp_out = Path(tmp_out_s)

            frames_in = extractor.extract(video_path, tmp_in, fps, frame_step)

            if preview:
                preview_out = output.parent / (output.stem + "_preview.png")
                FrameProcessor().process(frames_in[0], preview_out, config)
                print(f"Preview saved: {preview_out}")
                return

            frames_out = [tmp_out / f.name for f in frames_in]
            job_args = list(zip(frames_in, frames_out, [config] * len(frames_in)))

            with ProcessPoolExecutor(max_workers=workers) as executor:
                list(tqdm(
                    executor.map(_worker, job_args),
                    total=len(job_args),
                    desc="Rendering frames",
                ))

            assembler.assemble_mp4(tmp_out, output, fps)
            print(f"Saved: {output}")

            if make_gif:
                gif_output = output.with_suffix(".gif")
                assembler.assemble_gif(output, gif_output, gif_fps)
                print(f"Saved: {gif_output}")

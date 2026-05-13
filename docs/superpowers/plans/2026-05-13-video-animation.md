# Video Animation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `video` subcommand to ASCII Combinator that converts a video file into an ASCII-art animation (MP4 + optional GIF) using ffmpeg for frame extraction/assembly and parallel ASCII rendering per frame.

**Architecture:** A new `ascii_combinator/video.py` houses four classes: `VideoConfig` (picklable dataclass), `FrameExtractor` (ffmpeg → PNG frames), `FrameProcessor` (existing ASCII pipeline on one frame), `VideoAssembler` (PNG frames → MP4 + GIF via ffmpeg). `VideoProcessor` orchestrates them with `ProcessPoolExecutor` + `tqdm`. `cli.py` is refactored to use argparse subparsers while remaining backwards-compatible.

**Tech Stack:** Python 3.12, Pillow, NumPy, tqdm, system ffmpeg, concurrent.futures (stdlib), tempfile (stdlib), pytest

---

## File Map

| Path | Action | Responsibility |
|------|--------|----------------|
| `ascii_combinator/video.py` | Create | VideoConfig, FrameExtractor, FrameProcessor, VideoAssembler, VideoProcessor |
| `ascii_combinator/cli.py` | Modify | Add `video` subcommand; refactor flat main() into subparser-based structure |
| `tests/test_video.py` | Create | All video pipeline tests |
| `tests/test_cli.py` | Modify | Add backwards-compat test for image subcommand |
| `requirements.txt` | Modify | Add `tqdm>=4.65.0` |

---

### Task 1: VideoConfig dataclass and FrameProcessor

**Files:**
- Create: `ascii_combinator/video.py`
- Create: `tests/test_video.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_video.py`:

```python
import sys
import subprocess as _subprocess
import numpy as np
import pytest
from pathlib import Path
from PIL import Image
from unittest.mock import MagicMock, patch
from ascii_combinator.video import (
    FrameExtractor, FrameProcessor, VideoAssembler, VideoConfig, VideoProcessor,
)
from ascii_combinator.bg_mode import BgMode


def _make_png(path: Path, h: int = 32, w: int = 32) -> Path:
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[h // 2, :, :] = 120
    arr[:, w // 2, :] = 80
    Image.fromarray(arr).save(path)
    return path


def _default_config() -> VideoConfig:
    return VideoConfig(
        width=20,
        profile_name="monochrome",
        layer_names=["brightness"],
        jitter=0,
        threshold=0.15,
        font_size=8,
        bg_mode=BgMode.KEEP,
        soft_cfg=None,
    )


def test_frame_processor_produces_output(tmp_path):
    """FrameProcessor converts a real PNG into an ASCII PNG."""
    frame_in = _make_png(tmp_path / "frame_000001.png")
    frame_out = tmp_path / "frame_000001_ascii.png"

    result = FrameProcessor().process(frame_in, frame_out, _default_config())

    assert result == frame_out
    assert frame_out.exists()
    img = Image.open(frame_out)
    assert img.width > 0 and img.height > 0


def test_frame_processor_returns_frame_out(tmp_path):
    """process() return value is always frame_out."""
    frame_in = _make_png(tmp_path / "in.png")
    frame_out = tmp_path / "out.png"
    assert FrameProcessor().process(frame_in, frame_out, _default_config()) is frame_out


def test_video_config_is_picklable():
    """VideoConfig must survive pickle round-trip (needed for ProcessPoolExecutor)."""
    import pickle
    cfg = _default_config()
    assert pickle.loads(pickle.dumps(cfg)) == cfg
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd "/home/danil/Рабочий стол/ASCII_combinator" && python3 -m pytest tests/test_video.py -v
```

Expected: `ImportError: cannot import name 'FrameProcessor'`

- [ ] **Step 3: Implement VideoConfig and FrameProcessor**

Create `ascii_combinator/video.py`:

```python
from __future__ import annotations

import shutil
import subprocess
import tempfile
import os
from dataclasses import dataclass
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

        layers = [_LAYER_REGISTRY[n](threshold=config.threshold) for n in config.layer_names]
        charmap_list = [layer.process(image, num_rows, num_cols) for layer in layers]
        charmap = Compositor().composite(
            charmap_list, mask=None, bg_mode=config.bg_mode, soft_cfg=config.soft_cfg
        )
        profile = _PROFILE_REGISTRY[config.profile_name]()
        result = Renderer().render(charmap, profile, font_size=font_size, jitter=config.jitter)
        result.save(frame_out)
        return frame_out
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd "/home/danil/Рабочий стол/ASCII_combinator" && python3 -m pytest tests/test_video.py -v
```

Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
cd "/home/danil/Рабочий стол/ASCII_combinator" && git add ascii_combinator/video.py tests/test_video.py && git commit -m "feat: add VideoConfig dataclass and FrameProcessor"
```

---

### Task 2: FrameExtractor

**Files:**
- Modify: `ascii_combinator/video.py`
- Modify: `tests/test_video.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_video.py`:

```python
def _stub_ffmpeg_run(returncode: int = 0):
    return MagicMock(returncode=returncode, stderr="fake error")


def test_frame_extractor_fps_filter(tmp_path):
    """`fps` mode passes -vf fps=N to ffmpeg."""
    (tmp_path / "frame_000001.png").touch()  # fake extracted frame

    with patch("ascii_combinator.video.shutil.which", return_value="/usr/bin/ffmpeg"), \
         patch("ascii_combinator.video.subprocess.run", return_value=_stub_ffmpeg_run()) as mock_run:
        FrameExtractor().extract(Path("input.mp4"), tmp_path, fps=10.0, frame_step=None)

    args = mock_run.call_args[0][0]
    assert "-vf" in args
    vf_value = args[args.index("-vf") + 1]
    assert "fps=10.0" in vf_value


def test_frame_extractor_step_filter(tmp_path):
    """`frame_step` mode uses select filter, not fps."""
    (tmp_path / "frame_000001.png").touch()

    with patch("ascii_combinator.video.shutil.which", return_value="/usr/bin/ffmpeg"), \
         patch("ascii_combinator.video.subprocess.run", return_value=_stub_ffmpeg_run()) as mock_run:
        FrameExtractor().extract(Path("input.mp4"), tmp_path, fps=10.0, frame_step=3)

    args = mock_run.call_args[0][0]
    assert "-vf" in args
    vf_value = args[args.index("-vf") + 1]
    assert "select" in vf_value
    assert "3" in vf_value
    assert "fps=" not in vf_value


def test_frame_extractor_no_ffmpeg(tmp_path):
    """Missing ffmpeg binary raises FileNotFoundError."""
    with patch("ascii_combinator.video.shutil.which", return_value=None):
        with pytest.raises(FileNotFoundError, match="ffmpeg not found"):
            FrameExtractor().extract(Path("input.mp4"), tmp_path, fps=10.0, frame_step=None)


def test_frame_extractor_ffmpeg_failure(tmp_path):
    """ffmpeg non-zero exit raises RuntimeError with stderr."""
    with patch("ascii_combinator.video.shutil.which", return_value="/usr/bin/ffmpeg"), \
         patch("ascii_combinator.video.subprocess.run", return_value=_stub_ffmpeg_run(returncode=1)):
        with pytest.raises(RuntimeError, match="ffmpeg failed"):
            FrameExtractor().extract(Path("input.mp4"), tmp_path, fps=10.0, frame_step=None)


def test_frame_extractor_no_frames(tmp_path):
    """Empty output directory raises RuntimeError."""
    with patch("ascii_combinator.video.shutil.which", return_value="/usr/bin/ffmpeg"), \
         patch("ascii_combinator.video.subprocess.run", return_value=_stub_ffmpeg_run()):
        with pytest.raises(RuntimeError, match="No frames extracted"):
            FrameExtractor().extract(Path("input.mp4"), tmp_path, fps=10.0, frame_step=None)


def test_frame_extractor_returns_sorted_paths(tmp_path):
    """Returns sorted list of extracted frame paths."""
    for i in [3, 1, 2]:
        (tmp_path / f"frame_{i:06d}.png").touch()

    with patch("ascii_combinator.video.shutil.which", return_value="/usr/bin/ffmpeg"), \
         patch("ascii_combinator.video.subprocess.run", return_value=_stub_ffmpeg_run()):
        frames = FrameExtractor().extract(Path("input.mp4"), tmp_path, fps=5.0, frame_step=None)

    assert [f.name for f in frames] == [
        "frame_000001.png", "frame_000002.png", "frame_000003.png"
    ]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd "/home/danil/Рабочий стол/ASCII_combinator" && python3 -m pytest tests/test_video.py::test_frame_extractor_fps_filter -v
```

Expected: `ImportError: cannot import name 'FrameExtractor'`

- [ ] **Step 3: Implement FrameExtractor**

Add to `ascii_combinator/video.py` after the `FrameProcessor` class:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd "/home/danil/Рабочий стол/ASCII_combinator" && python3 -m pytest tests/test_video.py -v
```

Expected: 9 passed

- [ ] **Step 5: Commit**

```bash
cd "/home/danil/Рабочий стол/ASCII_combinator" && git add ascii_combinator/video.py tests/test_video.py && git commit -m "feat: add FrameExtractor with fps and frame_step support"
```

---

### Task 3: VideoAssembler

**Files:**
- Modify: `ascii_combinator/video.py`
- Modify: `tests/test_video.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_video.py`:

```python
def test_assembler_mp4_calls_ffmpeg(tmp_path):
    """assemble_mp4 invokes ffmpeg with libx264 and yuv420p."""
    with patch("ascii_combinator.video.subprocess.run", return_value=_stub_ffmpeg_run()) as mock_run:
        VideoAssembler().assemble_mp4(tmp_path, tmp_path / "out.mp4", fps=10.0)

    args = mock_run.call_args[0][0]
    assert "ffmpeg" in args
    assert "-c:v" in args
    assert "libx264" in args[args.index("-c:v") + 1]
    assert "-pix_fmt" in args
    assert "yuv420p" in args[args.index("-pix_fmt") + 1]
    assert str(tmp_path / "out.mp4") in args


def test_assembler_mp4_raises_on_failure(tmp_path):
    """assemble_mp4 raises RuntimeError when ffmpeg fails."""
    with patch("ascii_combinator.video.subprocess.run", return_value=_stub_ffmpeg_run(returncode=1)):
        with pytest.raises(RuntimeError, match="ffmpeg failed assembling MP4"):
            VideoAssembler().assemble_mp4(tmp_path, tmp_path / "out.mp4", fps=10.0)


def test_assembler_gif_calls_ffmpeg_twice(tmp_path):
    """assemble_gif calls ffmpeg exactly twice (palettegen + paletteuse)."""
    with patch("ascii_combinator.video.subprocess.run", return_value=_stub_ffmpeg_run()) as mock_run:
        VideoAssembler().assemble_gif(tmp_path / "in.mp4", tmp_path / "out.gif", fps=10.0)

    assert mock_run.call_count == 2
    first_args = mock_run.call_args_list[0][0][0]
    second_args = mock_run.call_args_list[1][0][0]
    assert "palettegen" in " ".join(first_args)
    assert "paletteuse" in " ".join(second_args)
    assert str(tmp_path / "out.gif") in second_args


def test_assembler_gif_raises_on_palettegen_failure(tmp_path):
    """assemble_gif raises RuntimeError if palettegen step fails."""
    with patch("ascii_combinator.video.subprocess.run", return_value=_stub_ffmpeg_run(returncode=1)):
        with pytest.raises(RuntimeError, match="palettegen failed"):
            VideoAssembler().assemble_gif(tmp_path / "in.mp4", tmp_path / "out.gif", fps=10.0)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd "/home/danil/Рабочий стол/ASCII_combinator" && python3 -m pytest tests/test_video.py::test_assembler_mp4_calls_ffmpeg -v
```

Expected: `ImportError: cannot import name 'VideoAssembler'`

- [ ] **Step 3: Implement VideoAssembler**

Add to `ascii_combinator/video.py` after `FrameExtractor`:

```python
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
        palette = mp4_path.parent / "_palette.png"
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd "/home/danil/Рабочий стол/ASCII_combinator" && python3 -m pytest tests/test_video.py -v
```

Expected: 13 passed

- [ ] **Step 5: Commit**

```bash
cd "/home/danil/Рабочий стол/ASCII_combinator" && git add ascii_combinator/video.py tests/test_video.py && git commit -m "feat: add VideoAssembler for MP4 and GIF output"
```

---

### Task 4: VideoProcessor orchestrator

**Files:**
- Modify: `ascii_combinator/video.py`
- Modify: `tests/test_video.py`
- Modify: `requirements.txt`

- [ ] **Step 1: Add tqdm to requirements.txt**

Add `tqdm>=4.65.0` to `requirements.txt`:

```
Pillow>=10.0.0
numpy>=1.24.0
scipy>=1.10.0
pytest>=7.4.0
tqdm>=4.65.0
```

Install it:

```bash
pip install tqdm
```

- [ ] **Step 2: Write failing tests**

Append to `tests/test_video.py`:

```python
def test_video_processor_preview_mode(tmp_path):
    """Preview mode processes only the first frame and saves a PNG."""
    frames_in = [_make_png(tmp_path / f"frame_{i:06d}.png") for i in range(1, 4)]
    output = tmp_path / "out.mp4"

    processed = []

    def fake_process(frame_in, frame_out, config):
        processed.append(frame_in)
        Image.new("RGB", (10, 10)).save(frame_out)
        return frame_out

    with patch.object(FrameProcessor, "process", side_effect=fake_process), \
         patch.object(FrameExtractor, "extract", return_value=frames_in), \
         patch.object(VideoAssembler, "assemble_mp4"), \
         patch.object(VideoAssembler, "assemble_gif"):
        VideoProcessor().process(
            video_path=Path("in.mp4"),
            output=output,
            config=_default_config(),
            fps=10.0,
            frame_step=None,
            workers=1,
            preview=True,
            make_gif=False,
            gif_fps=10.0,
        )

    assert len(processed) == 1
    assert processed[0] == frames_in[0]
    preview_png = tmp_path / "out_preview.png"
    assert preview_png.exists()


def test_video_processor_calls_assembler(tmp_path):
    """Normal mode calls assemble_mp4 after processing all frames."""
    frames_in = [_make_png(tmp_path / f"frame_{i:06d}.png") for i in range(1, 3)]
    output = tmp_path / "out.mp4"

    def fake_process(frame_in, frame_out, config):
        Image.new("RGB", (10, 10)).save(frame_out)
        return frame_out

    with patch.object(FrameProcessor, "process", side_effect=fake_process), \
         patch.object(FrameExtractor, "extract", return_value=frames_in), \
         patch.object(VideoAssembler, "assemble_mp4") as mock_mp4, \
         patch.object(VideoAssembler, "assemble_gif") as mock_gif:
        VideoProcessor().process(
            video_path=Path("in.mp4"),
            output=output,
            config=_default_config(),
            fps=10.0,
            frame_step=None,
            workers=1,
            preview=False,
            make_gif=False,
            gif_fps=10.0,
        )

    mock_mp4.assert_called_once()
    mock_gif.assert_not_called()


def test_video_processor_gif_flag(tmp_path):
    """make_gif=True calls assemble_gif with gif_fps=5.0."""
    frames_in = [_make_png(tmp_path / f"frame_{i:06d}.png") for i in range(1, 3)]
    output = tmp_path / "out.mp4"

    def fake_process(frame_in, frame_out, config):
        Image.new("RGB", (10, 10)).save(frame_out)
        return frame_out

    with patch.object(FrameProcessor, "process", side_effect=fake_process), \
         patch.object(FrameExtractor, "extract", return_value=frames_in), \
         patch.object(VideoAssembler, "assemble_mp4"), \
         patch.object(VideoAssembler, "assemble_gif") as mock_gif:
        VideoProcessor().process(
            video_path=Path("in.mp4"),
            output=output,
            config=_default_config(),
            fps=10.0,
            frame_step=None,
            workers=1,
            preview=False,
            make_gif=True,
            gif_fps=5.0,
        )

    mock_gif.assert_called_once()
    # assemble_gif(mp4_path, gif_output, gif_fps) — third positional arg is gif_fps
    assert mock_gif.call_args[0][2] == 5.0
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
cd "/home/danil/Рабочий стол/ASCII_combinator" && python3 -m pytest tests/test_video.py::test_video_processor_preview_mode -v
```

Expected: `ImportError: cannot import name 'VideoProcessor'`

- [ ] **Step 4: Implement VideoProcessor**

Add to `ascii_combinator/video.py` after `VideoAssembler`. First add the module-level worker function (must be at top level for pickling):

Add this import at the top of the file alongside existing imports:
```python
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
```

Add this module-level function right before the `VideoProcessor` class definition:

```python
def _worker(args: tuple[Path, Path, VideoConfig]) -> Path:
    frame_in, frame_out, config = args
    return FrameProcessor().process(frame_in, frame_out, config)
```

Then add the `VideoProcessor` class:

```python
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
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd "/home/danil/Рабочий стол/ASCII_combinator" && python3 -m pytest tests/test_video.py -v
```

Expected: 16 passed

- [ ] **Step 6: Commit**

```bash
cd "/home/danil/Рабочий стол/ASCII_combinator" && git add ascii_combinator/video.py tests/test_video.py requirements.txt && git commit -m "feat: add VideoProcessor orchestrator with parallel processing and tqdm"
```

---

### Task 5: CLI video subcommand + refactor

**Files:**
- Modify: `ascii_combinator/cli.py`
- Modify: `tests/test_video.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_video.py`:

```python
def _make_test_video(path: Path) -> Path:
    """Create a tiny 1-second test video using ffmpeg."""
    img_path = path.parent / "_tmp_frame.png"
    arr = np.zeros((32, 32, 3), dtype=np.uint8)
    arr[16, :, :] = 100
    Image.fromarray(arr).save(img_path)
    _subprocess.run(
        ["ffmpeg", "-y", "-loop", "1", "-i", str(img_path),
         "-t", "0.3", "-r", "5", "-c:v", "libx264", "-pix_fmt", "yuv420p",
         str(path)],
        capture_output=True, check=True,
    )
    img_path.unlink(missing_ok=True)
    return path


def test_video_cli_preview(tmp_path):
    """`video --preview` renders first frame as PNG and exits 0."""
    video = _make_test_video(tmp_path / "test.mp4")
    output = tmp_path / "out.mp4"

    result = _subprocess.run(
        [sys.executable, "-m", "ascii_combinator", "video", str(video),
         "-o", str(output), "--width", "20", "--preview"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    preview = tmp_path / "out_preview.png"
    assert preview.exists()


def test_video_cli_invalid_bg_mode(tmp_path):
    """`--bg-mode remove` is rejected in video mode."""
    video = _make_test_video(tmp_path / "test.mp4")

    result = _subprocess.run(
        [sys.executable, "-m", "ascii_combinator", "video", str(video),
         "--width", "20", "--bg-mode", "remove"],
        capture_output=True, text=True,
    )
    assert result.returncode != 0


def test_video_cli_fps_and_step_both_warn(tmp_path):
    """When both --fps and --frame-step are given, frame-step wins and a warning is printed to stderr."""
    video = _make_test_video(tmp_path / "test.mp4")
    output = tmp_path / "out.mp4"

    result = _subprocess.run(
        [sys.executable, "-m", "ascii_combinator", "video", str(video),
         "-o", str(output), "--width", "20", "--fps", "5", "--frame-step", "2", "--preview"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "frame-step" in result.stderr.lower()
```

Append to `tests/test_cli.py`:

```python
def test_cli_image_subcommand_explicit(tmp_path):
    """Explicit `image` subcommand works like the bare invocation."""
    input_img = tmp_path / "test.jpg"
    output_img = tmp_path / "out.png"
    _make_test_image(input_img)

    result = subprocess.run(
        [sys.executable, "-m", "ascii_combinator", "image", str(input_img),
         "-o", str(output_img), "--width", "20"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert output_img.exists()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd "/home/danil/Рабочий стол/ASCII_combinator" && python3 -m pytest tests/test_video.py::test_video_cli_preview tests/test_cli.py::test_cli_image_subcommand_explicit -v
```

Expected: tests fail — `video` subcommand not recognized yet.

- [ ] **Step 3: Replace full content of `ascii_combinator/cli.py`**

```python
import argparse
import os
import sys
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
from ascii_combinator.video import VideoConfig, VideoProcessor

LAYER_REGISTRY = {
    "brightness": BrightnessLayer,
    "sobel_x": SobelXLayer,
    "sobel_y": SobelYLayer,
    "diagonal": DiagonalLayer,
}

PROFILE_REGISTRY = {
    "monochrome": MonochromeProfile,
}


def _opacity_type(v: str) -> float:
    try:
        f = float(v)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid float value: '{v}'")
    if not 0.0 <= f <= 1.0:
        raise argparse.ArgumentTypeError("--bg-opacity must be between 0.0 and 1.0")
    return f


def _add_shared_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("input", type=Path, help="Input file path")
    parser.add_argument("-o", "--output", type=Path, default=None)
    parser.add_argument("--width", type=int, default=None, help="Output width in characters")
    parser.add_argument("--profile", default="monochrome", choices=list(PROFILE_REGISTRY))
    parser.add_argument("--layers", default=",".join(LAYER_REGISTRY.keys()))
    parser.add_argument("--jitter", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0.15)
    parser.add_argument("--font-size", type=int, default=12)
    parser.add_argument("--bg-mode", default="keep", choices=["keep", "remove", "soft"],
                        dest="bg_mode")
    parser.add_argument("--bg-opacity", type=_opacity_type, default=0.25, dest="bg_opacity")
    parser.add_argument("--bg-chars", type=str, default=".,", dest="bg_chars")


def _add_video_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--fps", type=float, default=10.0,
                        help="Frames per second to extract (and output fps)")
    parser.add_argument("--frame-step", type=int, default=None, dest="frame_step",
                        help="Extract every N-th frame (overrides --fps for extraction)")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1,
                        help="Parallel worker processes")
    parser.add_argument("--preview", action="store_true",
                        help="Render first frame only as a preview PNG")
    parser.add_argument("--gif", action="store_true",
                        help="Also produce a GIF alongside the MP4")
    parser.add_argument("--gif-fps", type=float, default=10.0, dest="gif_fps",
                        help="FPS for GIF output")


def _parse_shared(args: argparse.Namespace, parser: argparse.ArgumentParser):
    layer_names = [n.strip() for n in args.layers.split(",") if n.strip()]
    unknown = [n for n in layer_names if n not in LAYER_REGISTRY]
    if unknown:
        parser.error(f"unknown layer(s): {', '.join(unknown)}. Choose from: {list(LAYER_REGISTRY)}")
    if not layer_names:
        parser.error("--layers cannot be empty")

    bg_mode = BgMode(args.bg_mode)
    if bg_mode == BgMode.SOFT and not args.bg_chars:
        parser.error("--bg-chars cannot be empty when --bg-mode is soft")

    soft_cfg = SoftBgConfig(opacity=args.bg_opacity, chars=args.bg_chars) if bg_mode == BgMode.SOFT else None
    return layer_names, bg_mode, soft_cfg


def _run_image(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    layer_names, bg_mode, soft_cfg = _parse_shared(args, parser)

    image = Image.open(args.input)
    font_size = args.font_size
    cell_w = max(int(font_size * 0.6), 1)
    cell_h = max(font_size, 1)

    num_cols = args.width if args.width is not None else max(image.width // cell_w, 10)
    num_rows = max(1, int(num_cols * cell_w / cell_h * image.height / image.width))

    if bg_mode != BgMode.KEEP:
        try:
            from ascii_combinator.segmentation import Segmenter
            mask = Segmenter().segment(image, num_rows, num_cols)
        except ImportError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)
    else:
        mask = None

    layers = [LAYER_REGISTRY[n](threshold=args.threshold) for n in layer_names]
    charmap_list = [layer.process(image, num_rows, num_cols) for layer in layers]
    charmap = Compositor().composite(charmap_list, mask=mask, bg_mode=bg_mode, soft_cfg=soft_cfg)

    profile = PROFILE_REGISTRY[args.profile]()
    result = Renderer().render(charmap, profile, font_size=font_size, jitter=args.jitter)

    output = args.output or args.input.with_name(args.input.stem + "_ascii.png")
    result.save(output)
    print(f"Saved: {output}")


def _run_video(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if BgMode(args.bg_mode) == BgMode.REMOVE:
        parser.error("--bg-mode remove is not supported in video mode (too slow)")

    layer_names, bg_mode, soft_cfg = _parse_shared(args, parser)

    if args.frame_step is not None and args.fps != 10.0:
        print(
            "Warning: both --frame-step and --fps specified; "
            "--frame-step takes priority for extraction, --fps used for output.",
            file=sys.stderr,
        )

    config = VideoConfig(
        width=args.width,
        profile_name=args.profile,
        layer_names=layer_names,
        jitter=args.jitter,
        threshold=args.threshold,
        font_size=args.font_size,
        bg_mode=bg_mode,
        soft_cfg=soft_cfg,
    )

    output = args.output or args.input.with_name(args.input.stem + "_ascii.mp4")

    VideoProcessor().process(
        video_path=args.input,
        output=output,
        config=config,
        fps=args.fps,
        frame_step=args.frame_step,
        workers=args.workers,
        preview=args.preview,
        make_gif=args.gif,
        gif_fps=args.gif_fps,
    )


def main() -> None:
    # Backwards compatibility: if first arg is not a known subcommand, treat as `image`
    if len(sys.argv) > 1 and sys.argv[1] not in ("video", "image", "-h", "--help"):
        sys.argv.insert(1, "image")

    parser = argparse.ArgumentParser(description="ASCII Combinator — multilayer image/video to ASCII")
    subparsers = parser.add_subparsers(dest="command", required=True)

    image_parser = subparsers.add_parser("image", help="Convert a single image to ASCII PNG")
    _add_shared_args(image_parser)

    video_parser = subparsers.add_parser("video", help="Convert a video to ASCII animation")
    _add_shared_args(video_parser)
    # Override --bg-mode choices for video (remove not allowed)
    for action in video_parser._actions:
        if hasattr(action, "dest") and action.dest == "bg_mode":
            action.choices = ["keep", "soft"]
            break
    _add_video_args(video_parser)

    args = parser.parse_args()
    if args.command == "image":
        _run_image(args, parser)
    else:
        _run_video(args, video_parser)
```

- [ ] **Step 4: Run all tests**

```bash
cd "/home/danil/Рабочий стол/ASCII_combinator" && python3 -m pytest tests/ -v
```

Expected: all tests pass (42 existing + new video and CLI tests).

- [ ] **Step 5: Commit**

```bash
cd "/home/danil/Рабочий стол/ASCII_combinator" && git add ascii_combinator/cli.py tests/test_video.py tests/test_cli.py && git commit -m "feat: add video subcommand with parallel frame processing, preview, and GIF support"
```

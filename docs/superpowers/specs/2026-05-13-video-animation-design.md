# ASCII Combinator — Video Animation Design Spec
**Date:** 2026-05-13

## Overview

Add a `video` subcommand to ASCII Combinator that converts a video file into an ASCII-art animation. The pipeline extracts frames via ffmpeg, processes each frame through the existing ASCII pipeline in parallel, then reassembles the frames back into MP4 (and optionally GIF) using ffmpeg.

---

## Goals

- Convert any ffmpeg-readable video (MP4, AVI, MOV, WebM) to ASCII-art animation
- Output MP4 by default; optionally also produce a GIF
- Let the user control frame density with `--fps` or `--frame-step`
- Preview mode: render only the first frame to a PNG for settings validation
- Parallel frame processing with a tqdm progress bar
- No new heavy dependencies — only `tqdm` and system `ffmpeg`

## Non-Goals

- Real-time preview / streaming output
- Per-frame `--bg-mode remove/soft` (segmentation on every frame is too slow for v1)
- Audio passthrough
- Batch processing of multiple videos

---

## Architecture

```
video.py
  ├── FrameExtractor   — subprocess ffmpeg → temp PNG frames
  ├── FrameProcessor   — runs existing layer/compositor/renderer pipeline on one frame
  ├── VideoAssembler   — subprocess ffmpeg → MP4 + optional GIF from PNG frames
  └── VideoProcessor   — orchestrates the above three; exposes process()
```

### New files

| Path | Action | Responsibility |
|------|--------|----------------|
| `ascii_combinator/video.py` | Create | FrameExtractor, FrameProcessor, VideoAssembler, VideoProcessor |
| `ascii_combinator/cli.py` | Modify | Add `video` subcommand; refactor shared args into helper |
| `tests/test_video.py` | Create | Unit + integration tests for video pipeline |

### Existing files — unchanged

`layers/`, `compositor.py`, `renderer.py`, `bg_mode.py`, `segmentation.py` — not modified.

---

## CLI

```
python -m ascii_combinator video INPUT [-o OUTPUT]
    [--fps FLOAT] [--frame-step INT]
    [--workers INT]
    [--preview]
    [--gif] [--gif-fps FLOAT]
    [--width INT] [--profile PROFILE]
    [--layers LAYERS] [--jitter INT] [--threshold FLOAT] [--font-size INT]
    [--bg-mode {keep,soft}] [--bg-opacity FLOAT] [--bg-chars STR]
```

### Flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `INPUT` | Path | required | Input video file |
| `-o OUTPUT` | Path | `<stem>_ascii.mp4` | Output MP4 path |
| `--fps` | float | 10.0 | Frames per second to extract from source video |
| `--frame-step` | int | None | Extract every N-th frame (takes priority over `--fps`) |
| `--workers` | int | `os.cpu_count()` | Parallel worker processes |
| `--preview` | flag | False | Render first frame only → `<stem>_preview.png` |
| `--gif` | flag | False | Also produce a GIF alongside the MP4 |
| `--gif-fps` | float | 10.0 | FPS for the GIF output |
| Shared image flags | — | same defaults | `--width`, `--profile`, `--layers`, `--jitter`, `--threshold`, `--font-size`, `--bg-mode`, `--bg-opacity`, `--bg-chars` |

**Constraint:** `--bg-mode remove` is not allowed in video mode (too slow). Only `keep` and `soft` are accepted.

---

## Data Flow

```
input.mp4
  → FrameExtractor.extract()
      ffmpeg -i input.mp4 -vf fps=10 /tmp/<uuid>/frame_%06d.png
      returns: list[Path]  (sorted frame paths)

  → VideoProcessor loops frames:
      if --preview: process frame[0] only → save PNG, exit
      else: ProcessPoolExecutor(workers) maps FrameProcessor.process over all frames
            tqdm progress bar on the iterator

  → FrameProcessor.process(frame_path, config) -> Path
      Image.open(frame_path)
      → layers → compositor → renderer → save as PNG in output temp dir
      returns: output frame path

  → VideoAssembler.assemble(frame_paths, output, fps)
      ffmpeg -framerate FPS -i frame_%06d.png -c:v libx264 -pix_fmt yuv420p output.mp4
      if --gif:
          ffmpeg -i output.mp4 -vf "fps=GIF_FPS,scale=WIDTH:-1:flags=lanczos" output.gif

  → cleanup temp dirs
```

---

## FrameExtractor

```python
class FrameExtractor:
    def extract(self, video_path: Path, tmp_dir: Path,
                fps: float | None, frame_step: int | None) -> list[Path]:
        """Run ffmpeg to extract frames into tmp_dir. Returns sorted frame paths."""
```

- If `frame_step` is set: use `-vf select='not(mod(n,STEP))'` filter
- If only `fps` is set: use `-vf fps=FPS`
- Raises `RuntimeError` if ffmpeg exits non-zero (includes stderr in message)
- Raises `FileNotFoundError` if ffmpeg binary not found

---

## FrameProcessor

```python
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
        """Process one frame through the ASCII pipeline. Returns frame_out."""
```

- Stateless: safe to call from worker processes
- `VideoConfig` is a plain dataclass — picklable for multiprocessing

---

## VideoAssembler

```python
class VideoAssembler:
    def assemble_mp4(self, frames_dir: Path, output: Path, fps: float) -> None:
        """Assemble sorted PNGs from frames_dir into an MP4."""

    def assemble_gif(self, mp4_path: Path, output: Path, fps: float) -> None:
        """Convert MP4 to GIF via ffmpeg palette trick."""
```

- GIF assembly uses two-pass ffmpeg palette trick for best quality:
  1. `ffmpeg -i input.mp4 -vf "fps=FPS,palettegen" palette.png`
  2. `ffmpeg -i input.mp4 -i palette.png -vf "fps=FPS,paletteuse" output.gif`

---

## VideoProcessor (orchestrator)

```python
class VideoProcessor:
    def process(self, video_path: Path, output: Path, config: VideoConfig,
                fps: float, frame_step: int | None,
                workers: int, preview: bool,
                make_gif: bool, gif_fps: float) -> None:
```

- Creates two temp dirs: `frames_in/` and `frames_out/`
- Cleans up both dirs via `finally` block even on error
- Preview mode: processes `frames[0]` only, saves to `<output.stem>_preview.png`
- Normal mode: `ProcessPoolExecutor(max_workers=workers)` + `tqdm`

---

## CLI Refactor

`cli.py` currently has one flat `main()`. Refactor to:

```python
def main():
    parser = argparse.ArgumentParser(...)
    subparsers = parser.add_subparsers(dest="command")

    # image subcommand (existing behaviour, now explicit)
    image_parser = subparsers.add_parser("image", ...)
    _add_shared_args(image_parser)

    # video subcommand (new)
    video_parser = subparsers.add_parser("video", ...)
    _add_shared_args(video_parser)
    _add_video_args(video_parser)

    # backwards compatibility: no subcommand → treat as image
    args = parser.parse_args()
    if args.command is None or args.command == "image":
        _run_image(args)
    elif args.command == "video":
        _run_video(args)
```

Backwards compatibility: calling `python -m ascii_combinator input.jpg` (no subcommand) continues to work exactly as before.

---

## Error Handling

| Scenario | Behaviour |
|----------|-----------|
| ffmpeg not installed | `RuntimeError: ffmpeg not found. Install with: apt install ffmpeg` |
| ffmpeg exits non-zero | `RuntimeError: ffmpeg failed: <stderr output>` |
| No frames extracted | `RuntimeError: No frames extracted from <path>` |
| Worker process crash | Exception propagated from `ProcessPoolExecutor`, temp dirs cleaned up |
| `--bg-mode remove` in video | `parser.error(...)` — not supported in video mode |
| `--frame-step` and `--fps` both set | `--frame-step` wins, warning printed |

---

## Testing

| Test | File | Description |
|------|------|-------------|
| `test_frame_extractor_calls_ffmpeg` | `test_video.py` | Mock subprocess; verify ffmpeg args |
| `test_frame_extractor_fps_filter` | `test_video.py` | `--fps` → `-vf fps=N` in args |
| `test_frame_extractor_step_filter` | `test_video.py` | `--frame-step` → select filter |
| `test_frame_extractor_no_ffmpeg` | `test_video.py` | ffmpeg missing → FileNotFoundError |
| `test_frame_processor_produces_output` | `test_video.py` | Real PNG in → ASCII PNG out |
| `test_assembler_mp4_calls_ffmpeg` | `test_video.py` | Mock subprocess; verify ffmpeg args |
| `test_assembler_gif_two_pass` | `test_video.py` | GIF path calls ffmpeg twice |
| `test_video_cli_preview` | `test_video.py` | `video --preview` → PNG exists, returncode=0 |
| `test_video_cli_invalid_bg_mode` | `test_video.py` | `--bg-mode remove` → returncode!=0 |
| `test_cli_image_backwards_compat` | `test_cli.py` | Existing image invocation still works |

---

## Dependencies

| Package | Status | Purpose |
|---------|--------|---------|
| `ffmpeg` | system binary (already present) | Frame extraction + video assembly |
| `tqdm` | new pip dependency | Progress bar |
| `concurrent.futures` | stdlib | ProcessPoolExecutor |
| `tempfile` | stdlib | Temp directories |

Add `tqdm` to `requirements.txt` / `pyproject.toml`.

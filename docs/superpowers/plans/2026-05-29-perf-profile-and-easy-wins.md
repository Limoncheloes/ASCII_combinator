# Performance: Profiling Harness + Easy Wins — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reproducible benchmark harness for the ASCII pipeline, commit a baseline, then apply 2 (possibly 3) inline optimizations that the baseline confirms are worth it — all without changing public APIs.

**Architecture:** New self-contained `benchmarks/` package generates synthetic inputs in `/tmp`, runs the existing image and video pipelines under stage-timed wrappers, writes a JSON report and a console table. After baseline is committed, apply `lru_cache` to font loading and introduce a shared `LayerInputs` dataclass so 4 layers reuse one grayscale conversion (and Sobel-X/Y gradients). Re-run bench; if layer Python loops still dominate, vectorize them.

**Tech Stack:** Python 3.11+, numpy, scipy, Pillow, ffmpeg (already required), `functools.lru_cache`, `time.perf_counter`, `cProfile` (stdlib).

**Spec:** `docs/superpowers/specs/2026-05-29-perf-profile-and-easy-wins-design.md`

---

## File map

**New:**
- `benchmarks/__init__.py`
- `benchmarks/fixtures.py` — synthetic image + video generators
- `benchmarks/instrument.py` — stage timer
- `benchmarks/scenarios.py` — S1/S2/S3 dataclasses
- `benchmarks/run.py` — runner CLI + JSON + console table + visual smoke save
- `benchmarks/baseline.json` — generated, always reflects latest
- `benchmarks/baseline_initial.json` — generated once at Task 7, committed as historical anchor
- `benchmarks/visual_smoke/s1_baseline.png` — committed once at Task 7
- `benchmarks/visual_smoke/s1_after.png` — committed at Task 12
- `tests/test_instrument.py`
- `tests/test_layer_inputs.py`

**Modified:**
- `ascii_combinator/renderer.py` — `@lru_cache` on `_find_font`
- `ascii_combinator/layers/base.py` — add `LayerInputs` dataclass; extend `Layer.process` signature with optional `inputs=`
- `ascii_combinator/layers/brightness.py` — accept and use `inputs`
- `ascii_combinator/layers/sobel_x.py` — accept and use `inputs`
- `ascii_combinator/layers/sobel_y.py` — accept and use `inputs`
- `ascii_combinator/layers/diagonal.py` — accept and use `inputs`
- `ascii_combinator/cli.py:_run_image` — construct one `LayerInputs`, pass through
- `ascii_combinator/video.py:FrameProcessor.process` — same
- `web_ui.py:_convert_image` — same

---

## Task 1: Benchmark package skeleton + synthetic fixtures

**Files:**
- Create: `benchmarks/__init__.py`
- Create: `benchmarks/fixtures.py`
- Create: `tests/test_fixtures.py`

- [ ] **Step 1: Create empty package init**

```bash
mkdir -p benchmarks
```

Create `benchmarks/__init__.py`:

```python
```

(empty file)

- [ ] **Step 2: Write the failing test**

Create `tests/test_fixtures.py`:

```python
from pathlib import Path
from PIL import Image

from benchmarks.fixtures import make_synthetic_image, make_synthetic_video


def test_synthetic_image_has_requested_size(tmp_path: Path):
    path = make_synthetic_image(tmp_path, width=400, height=300, seed=7)
    assert path.exists()
    with Image.open(path) as im:
        assert im.size == (400, 300)
        assert im.mode == "RGB"


def test_synthetic_image_is_deterministic(tmp_path: Path):
    a = make_synthetic_image(tmp_path / "a.png", width=64, height=64, seed=42)
    b = make_synthetic_image(tmp_path / "b.png", width=64, height=64, seed=42)
    assert a.read_bytes() == b.read_bytes()


def test_synthetic_video_has_requested_duration(tmp_path: Path):
    path = make_synthetic_video(
        tmp_path, width=320, height=240, duration_s=2, fps=10, seed=1
    )
    assert path.exists()
    assert path.suffix == ".mp4"
    assert path.stat().st_size > 0
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_fixtures.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'benchmarks.fixtures'`

- [ ] **Step 4: Implement `benchmarks/fixtures.py`**

```python
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
            proc.stdin.write(frame.tobytes())
        proc.stdin.close()
    except BrokenPipeError:
        pass
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"ffmpeg returned {rc} while encoding synthetic video")
    return out_path
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_fixtures.py -v`
Expected: 3 PASS

- [ ] **Step 6: Commit**

```bash
git add benchmarks/__init__.py benchmarks/fixtures.py tests/test_fixtures.py
git commit -m "feat(bench): synthetic image and video fixtures for benchmarks"
```

---

## Task 2: Stage instrumentation

**Files:**
- Create: `benchmarks/instrument.py`
- Create: `tests/test_instrument.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_instrument.py`:

```python
import time

from benchmarks.instrument import StageRegistry, stage


def test_stage_records_elapsed_time():
    reg = StageRegistry()
    with stage("foo", registry=reg):
        time.sleep(0.01)
    samples = reg.samples_for("foo")
    assert len(samples) == 1
    assert samples[0] >= 0.01


def test_multiple_samples_accumulate():
    reg = StageRegistry()
    for _ in range(3):
        with stage("bar", registry=reg):
            pass
    assert len(reg.samples_for("bar")) == 3


def test_summary_returns_median_and_share():
    reg = StageRegistry()
    with stage("a", registry=reg):
        time.sleep(0.02)
    with stage("b", registry=reg):
        time.sleep(0.01)
    summary = reg.summary()
    assert "a" in summary and "b" in summary
    assert summary["a"]["median_s"] > summary["b"]["median_s"]
    assert 0.0 < summary["a"]["share_pct"] <= 100.0
    assert 0.0 < summary["b"]["share_pct"] <= 100.0
    total_share = summary["a"]["share_pct"] + summary["b"]["share_pct"]
    assert abs(total_share - 100.0) < 0.01


def test_unknown_stage_returns_empty_list():
    reg = StageRegistry()
    assert reg.samples_for("nope") == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_instrument.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'benchmarks.instrument'`

- [ ] **Step 3: Implement `benchmarks/instrument.py`**

```python
"""Stage timing helper.

The bench harness wraps each pipeline stage with `stage(name)` and reads
totals/medians from a `StageRegistry`. Source modules in `ascii_combinator/`
are NOT instrumented — wrapping happens in `benchmarks/run.py`.
"""
from __future__ import annotations

import statistics
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass
class StageRegistry:
    _samples: dict[str, list[float]] = field(
        default_factory=lambda: defaultdict(list)
    )

    def record(self, name: str, elapsed_s: float) -> None:
        self._samples[name].append(elapsed_s)

    def samples_for(self, name: str) -> list[float]:
        return list(self._samples.get(name, []))

    def summary(self) -> dict[str, dict[str, float]]:
        """Return per-stage median seconds and share-% of total median time."""
        medians = {
            name: statistics.median(samples)
            for name, samples in self._samples.items()
            if samples
        }
        total = sum(medians.values()) or 1.0
        return {
            name: {
                "median_s": round(med, 6),
                "share_pct": round(100.0 * med / total, 2),
                "count": len(self._samples[name]),
            }
            for name, med in medians.items()
        }


_DEFAULT_REGISTRY = StageRegistry()


@contextmanager
def stage(name: str, registry: StageRegistry | None = None):
    """Time the wrapped block and append the elapsed seconds to the registry."""
    reg = registry if registry is not None else _DEFAULT_REGISTRY
    t0 = time.perf_counter()
    try:
        yield
    finally:
        reg.record(name, time.perf_counter() - t0)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_instrument.py -v`
Expected: 4 PASS

- [ ] **Step 5: Commit**

```bash
git add benchmarks/instrument.py tests/test_instrument.py
git commit -m "feat(bench): stage timing context manager and registry"
```

---

## Task 3: Scenarios definition

**Files:**
- Create: `benchmarks/scenarios.py`

- [ ] **Step 1: Implement scenarios**

Create `benchmarks/scenarios.py`:

```python
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
```

- [ ] **Step 2: Smoke import check**

Run: `python -c "from benchmarks.scenarios import S1, S2, S3; print(S1, S2, S3)"`
Expected: prints three dataclasses, exit 0.

- [ ] **Step 3: Commit**

```bash
git add benchmarks/scenarios.py
git commit -m "feat(bench): define reference scenarios S1/S2/S3"
```

---

## Task 4: Bench runner — image pipeline timing

**Files:**
- Create: `benchmarks/run.py` (image-only part for now)

- [ ] **Step 1: Implement the image-pipeline runner**

Create `benchmarks/run.py`:

```python
"""Benchmark runner.

Usage:
    python -m benchmarks.run --scenarios s1,s2,s3 --repeats 3 \
        [--write-baseline] [--write-initial]

Outputs:
- prints per-scenario tables to stdout
- writes `benchmarks/baseline.json` if `--write-baseline`
- writes `benchmarks/baseline_initial.json` if `--write-initial`
- saves visual smoke PNG for S1 to `benchmarks/visual_smoke/s1_<tag>.png`
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

from PIL import Image

from ascii_combinator.bg_mode import BgMode
from ascii_combinator.compositor import Compositor
from ascii_combinator.layers.brightness import BrightnessLayer
from ascii_combinator.layers.diagonal import DiagonalLayer
from ascii_combinator.layers.sobel_x import SobelXLayer
from ascii_combinator.layers.sobel_y import SobelYLayer
from ascii_combinator.profiles.monochrome import MonochromeProfile
from ascii_combinator.renderer import Renderer

from benchmarks.fixtures import make_synthetic_image, make_synthetic_video
from benchmarks.instrument import StageRegistry, stage
from benchmarks.scenarios import ALL_IMAGE, ALL_VIDEO, ImageScenario, VideoScenario, S1, S2, S3

LAYER_REGISTRY = {
    "brightness": BrightnessLayer,
    "sobel_x": SobelXLayer,
    "sobel_y": SobelYLayer,
    "diagonal": DiagonalLayer,
}

REPO_ROOT = Path(__file__).resolve().parent.parent
VISUAL_SMOKE_DIR = REPO_ROOT / "benchmarks" / "visual_smoke"


def _grid_dims(image: Image.Image, out_width: int, font_size: int) -> tuple[int, int]:
    cell_w = max(int(font_size * 0.6), 1)
    cell_h = max(font_size, 1)
    num_cols = out_width
    num_rows = max(1, int(num_cols * cell_w / cell_h * image.height / image.width))
    return num_rows, num_cols


def run_image_scenario(
    scen: ImageScenario,
    tmpdir: Path,
    repeats: int,
    save_visual_as: Path | None = None,
) -> tuple[StageRegistry, float]:
    reg = StageRegistry()
    img_path = make_synthetic_image(tmpdir, width=scen.image_width, height=scen.image_height)

    last_result: Image.Image | None = None

    for _ in range(repeats):
        with stage(f"{scen.id}.total", registry=reg):
            with stage(f"{scen.id}.image.load", registry=reg):
                image = Image.open(img_path).copy()  # decode now, not lazily

            num_rows, num_cols = _grid_dims(image, scen.out_width, scen.font_size)
            layers = [LAYER_REGISTRY[n](threshold=scen.threshold) for n in scen.layers]
            charmap_list = []
            for layer in layers:
                with stage(f"{scen.id}.layer.{layer.id}.process", registry=reg):
                    charmap_list.append(layer.process(image, num_rows, num_cols))

            with stage(f"{scen.id}.compositor.composite", registry=reg):
                charmap = Compositor().composite(
                    charmap_list, mask=None, bg_mode=BgMode.KEEP, soft_cfg=None
                )

            with stage(f"{scen.id}.renderer.render", registry=reg):
                last_result = Renderer().render(
                    charmap, MonochromeProfile(),
                    font_size=scen.font_size, jitter=scen.jitter,
                )

    if save_visual_as is not None and last_result is not None:
        save_visual_as.parent.mkdir(parents=True, exist_ok=True)
        last_result.save(save_visual_as)

    import statistics
    total_median = statistics.median(reg.samples_for(f"{scen.id}.total"))
    return reg, total_median


def _placeholder_video_runner_will_come_in_task_5() -> None:
    """See Task 5 for the implementation. Calling this before Task 5 is a bug."""
    raise NotImplementedError("video runner not implemented yet — see Task 5")


def run_video_scenario(scen: VideoScenario, tmpdir: Path, repeats: int):
    return _placeholder_video_runner_will_come_in_task_5()


# --- CLI scaffold (extended in Task 6) ---

def _print_table(scenario_id: str, summary: dict, total_median: float) -> None:
    print(f"\n=== Scenario {scenario_id} — total median: {total_median:.3f} s ===")
    print(f"{'stage':<48} {'median_s':>10} {'share_%':>8} {'n':>4}")
    rows = sorted(summary.items(), key=lambda kv: -kv[1]["share_pct"])
    for name, info in rows:
        # Skip the .total row in the per-stage table — it's already in the header.
        if name.endswith(".total"):
            continue
        print(f"{name:<48} {info['median_s']:>10.4f} {info['share_pct']:>7.2f}% {info['count']:>4}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="benchmarks.run")
    parser.add_argument("--scenarios", default="s1,s2,s3",
                        help="Comma-separated scenario ids")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--write-baseline", action="store_true",
                        help="Write benchmarks/baseline.json")
    parser.add_argument("--write-initial", action="store_true",
                        help="Write benchmarks/baseline_initial.json (use once)")
    parser.add_argument("--visual-tag", default="current",
                        help="Suffix for visual smoke PNG (e.g. 'baseline', 'after')")
    parser.add_argument("--profile", action="store_true",
                        help="Also run S2 under cProfile and print top-30 functions")
    args = parser.parse_args(argv)

    requested = {s.strip() for s in args.scenarios.split(",") if s.strip()}
    by_id = {s.id: s for s in (*ALL_IMAGE, *ALL_VIDEO)}
    unknown = requested - by_id.keys()
    if unknown:
        print(f"Unknown scenarios: {sorted(unknown)}", file=sys.stderr)
        return 2

    report = {
        "host": {
            "cpu_count": os.cpu_count(),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
        "git_sha": _git_sha(),
        "scenarios": {},
    }

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        for sid in requested:
            scen = by_id[sid]
            if isinstance(scen, ImageScenario):
                visual_path = (
                    VISUAL_SMOKE_DIR / f"{sid}_{args.visual_tag}.png"
                    if sid == "s1" else None
                )
                reg, total_med = run_image_scenario(
                    scen, tmpdir, args.repeats, save_visual_as=visual_path,
                )
                summary = reg.summary()
                report["scenarios"][sid] = {
                    "total_median_s": round(total_med, 6),
                    "stages": summary,
                }
                _print_table(sid, summary, total_med)
            else:
                reg, total_med = run_video_scenario(scen, tmpdir, args.repeats)
                summary = reg.summary()
                report["scenarios"][sid] = {
                    "total_median_s": round(total_med, 6),
                    "stages": summary,
                }
                _print_table(sid, summary, total_med)

    out_dir = REPO_ROOT / "benchmarks"
    if args.write_baseline:
        (out_dir / "baseline.json").write_text(json.dumps(report, indent=2))
        print(f"\nWrote {out_dir / 'baseline.json'}")
    if args.write_initial:
        (out_dir / "baseline_initial.json").write_text(json.dumps(report, indent=2))
        print(f"Wrote {out_dir / 'baseline_initial.json'}")

    if args.profile:
        import cProfile
        import pstats
        with tempfile.TemporaryDirectory() as tmp:
            print("\n=== cProfile dump for S2 (top 30 by cumulative time) ===")
            pr = cProfile.Profile()
            pr.enable()
            run_image_scenario(S2, Path(tmp), repeats=1)
            pr.disable()
            pstats.Stats(pr).sort_stats("cumulative").print_stats(30)

    return 0


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Smoke-run with S2 only (faster than S1)**

Run: `python -m benchmarks.run --scenarios s2 --repeats 1`
Expected: prints a per-stage table for `s2`, exits 0. Numbers will vary by machine; the point is the harness runs.

- [ ] **Step 3: Commit**

```bash
git add benchmarks/run.py
git commit -m "feat(bench): runner for image scenarios with stage timing"
```

---

## Task 5: Bench runner — video pipeline timing

**Files:**
- Modify: `benchmarks/run.py` — replace `_placeholder_video_runner_will_come_in_task_5` body and `run_video_scenario`

- [ ] **Step 1: Replace placeholder with real implementation**

In `benchmarks/run.py`, delete the `_placeholder_video_runner_will_come_in_task_5` function and replace the `run_video_scenario` function with:

```python
def run_video_scenario(
    scen: VideoScenario,
    tmpdir: Path,
    repeats: int,
) -> tuple[StageRegistry, float]:
    """Time per-frame stages by running the video flow synchronously (no parallelism).

    Parallel video processing (`VideoProcessor`) hides per-stage timing because frames
    run in subprocesses. Here we deliberately use the synchronous path: extract frames
    via ffmpeg, then process each frame inline with stage() around each call.
    """
    import statistics
    import shutil
    import subprocess as sp

    reg = StageRegistry()
    video_path = make_synthetic_video(
        tmpdir,
        width=scen.video_width,
        height=scen.video_height,
        duration_s=scen.duration_s,
        fps=scen.source_fps,
    )

    for _ in range(repeats):
        with stage(f"{scen.id}.total", registry=reg):
            frames_dir = tmpdir / f"frames_{scen.id}"
            if frames_dir.exists():
                shutil.rmtree(frames_dir)
            frames_dir.mkdir(parents=True)

            with stage(f"{scen.id}.video.ffmpeg.extract", registry=reg):
                cmd = [
                    "ffmpeg", "-y", "-i", str(video_path),
                    "-vf", f"fps={scen.out_fps}",
                    str(frames_dir / "frame_%06d.png"),
                ]
                r = sp.run(cmd, capture_output=True, text=True)
                if r.returncode != 0:
                    raise RuntimeError(f"ffmpeg extract failed: {r.stderr}")

            frames_in = sorted(frames_dir.glob("frame_*.png"))
            out_dir = tmpdir / f"out_{scen.id}"
            if out_dir.exists():
                shutil.rmtree(out_dir)
            out_dir.mkdir()

            layers = [LAYER_REGISTRY[n](threshold=scen.threshold) for n in scen.layers]

            for frame_in in frames_in:
                image = Image.open(frame_in)
                num_rows, num_cols = _grid_dims(image, scen.out_width, scen.font_size)
                charmap_list = []
                for layer in layers:
                    with stage(f"{scen.id}.frame.layer.{layer.id}", registry=reg):
                        charmap_list.append(layer.process(image, num_rows, num_cols))
                with stage(f"{scen.id}.frame.compositor", registry=reg):
                    charmap = Compositor().composite(
                        charmap_list, mask=None, bg_mode=BgMode.KEEP, soft_cfg=None,
                    )
                with stage(f"{scen.id}.frame.renderer", registry=reg):
                    rendered = Renderer().render(
                        charmap, MonochromeProfile(),
                        font_size=scen.font_size, jitter=scen.jitter,
                    )
                rendered.save(out_dir / frame_in.name)

            with stage(f"{scen.id}.video.ffmpeg.assemble_mp4", registry=reg):
                out_mp4 = tmpdir / f"{scen.id}_result.mp4"
                cmd = [
                    "ffmpeg", "-y",
                    "-framerate", str(scen.out_fps),
                    "-i", str(out_dir / "frame_%06d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    str(out_mp4),
                ]
                r = sp.run(cmd, capture_output=True, text=True)
                if r.returncode != 0:
                    raise RuntimeError(f"ffmpeg assemble failed: {r.stderr}")

    total_median = statistics.median(reg.samples_for(f"{scen.id}.total"))
    return reg, total_median
```

- [ ] **Step 2: Smoke-run with S3 only, repeats=1**

Run: `python -m benchmarks.run --scenarios s3 --repeats 1`
Expected: prints a per-stage table for `s3` including `frame.*` and `video.ffmpeg.*` rows. Should complete in well under 5 minutes.

- [ ] **Step 3: Commit**

```bash
git add benchmarks/run.py
git commit -m "feat(bench): runner for video scenario S3 with per-frame stage timing"
```

---

## Task 6: First baseline — record and commit

**Files:**
- Create: `benchmarks/baseline_initial.json` (generated)
- Create: `benchmarks/baseline.json` (generated, identical at this point)
- Create: `benchmarks/visual_smoke/s1_baseline.png` (generated)
- Create: `benchmarks/visual_smoke/.gitkeep` (only if `visual_smoke/` was empty; otherwise skip)

- [ ] **Step 1: Run the full benchmark and write both JSONs**

Run:

```bash
python -m benchmarks.run --scenarios s1,s2,s3 --repeats 3 \
    --write-baseline --write-initial --visual-tag baseline
```

Expected: prints three tables, writes `benchmarks/baseline.json`, `benchmarks/baseline_initial.json`, and `benchmarks/visual_smoke/s1_baseline.png`.

- [ ] **Step 2: Inspect output**

Open `benchmarks/baseline.json` and confirm:
- All three scenarios present
- Each has a `total_median_s` and a `stages` map with non-empty entries
- Stages sum (approximately) to the total

Note the **top-3 stages by share_pct** for each scenario — this drives the decision about Win 3 in Task 12.

- [ ] **Step 3: Commit**

```bash
git add benchmarks/baseline.json benchmarks/baseline_initial.json benchmarks/visual_smoke/s1_baseline.png
git commit -m "bench: initial baseline numbers and visual smoke PNG"
```

---

## Task 7: Win 1 — LRU-cache the font loader

**Files:**
- Modify: `ascii_combinator/renderer.py`
- Create: `tests/test_renderer_font_cache.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_renderer_font_cache.py`:

```python
from ascii_combinator.renderer import _find_font


def test_find_font_returns_same_object_for_same_size():
    a = _find_font(12)
    b = _find_font(12)
    assert a is b, "LRU cache should return identical font object for same size"


def test_find_font_returns_different_object_for_different_size():
    a = _find_font(12)
    b = _find_font(18)
    assert a is not b
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_renderer_font_cache.py -v`
Expected: FAIL — first test fails because each call returns a fresh font object.

- [ ] **Step 3: Add `@lru_cache` to `_find_font`**

In `ascii_combinator/renderer.py`, change the top of the file from:

```python
import random
from PIL import Image, ImageDraw, ImageFont
from ascii_combinator.types import CharMap
from ascii_combinator.profiles.base import ColorProfile


def _find_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeMono.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except (IOError, OSError):
            continue
    return ImageFont.load_default()
```

to:

```python
import random
from functools import lru_cache
from PIL import Image, ImageDraw, ImageFont
from ascii_combinator.types import CharMap
from ascii_combinator.profiles.base import ColorProfile


@lru_cache(maxsize=16)
def _find_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeMono.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except (IOError, OSError):
            continue
    return ImageFont.load_default()
```

- [ ] **Step 4: Run new test + full suite**

Run: `pytest tests/test_renderer_font_cache.py -v && pytest`
Expected: new tests PASS; existing 21+ tests still PASS.

- [ ] **Step 5: Commit**

```bash
git add ascii_combinator/renderer.py tests/test_renderer_font_cache.py
git commit -m "perf(renderer): lru-cache font lookups to avoid per-render filesystem probing"
```

---

## Task 8: Win 2a — `LayerInputs` dataclass

**Files:**
- Modify: `ascii_combinator/layers/base.py`
- Create: `tests/test_layer_inputs.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_layer_inputs.py`:

```python
import numpy as np
from PIL import Image

from ascii_combinator.layers.base import LayerInputs


def _img(w: int = 16, h: int = 16) -> Image.Image:
    arr = np.arange(w * h, dtype=np.uint8).reshape(h, w) % 256
    return Image.fromarray(arr, mode="L").convert("RGB")


def test_layer_inputs_gray_matches_manual_conversion():
    img = _img()
    inputs = LayerInputs.from_image(img)
    expected = np.array(img.convert("L"), dtype=float) / 255.0
    np.testing.assert_allclose(inputs.gray, expected)


def test_layer_inputs_gray_is_cached():
    img = _img()
    inputs = LayerInputs.from_image(img)
    a = inputs.gray
    b = inputs.gray
    assert a is b, "gray should be computed once and reused"


def test_layer_inputs_sobel_x_and_y_match_scipy():
    from scipy.ndimage import sobel
    img = _img()
    inputs = LayerInputs.from_image(img)
    np.testing.assert_allclose(inputs.sobel_x, sobel(inputs.gray, axis=1))
    np.testing.assert_allclose(inputs.sobel_y, sobel(inputs.gray, axis=0))


def test_layer_inputs_sobel_cached():
    img = _img()
    inputs = LayerInputs.from_image(img)
    assert inputs.sobel_x is inputs.sobel_x
    assert inputs.sobel_y is inputs.sobel_y
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_layer_inputs.py -v`
Expected: FAIL — `ImportError: cannot import name 'LayerInputs'`.

- [ ] **Step 3: Add `LayerInputs` to `ascii_combinator/layers/base.py`**

Replace the contents of `ascii_combinator/layers/base.py` with:

```python
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from PIL import Image
from scipy.ndimage import sobel

from ascii_combinator.types import CharMap


@dataclass
class LayerInputs:
    """Precomputed numpy arrays shared between layers for a single image.

    Constructing one `LayerInputs` per image (instead of recomputing inside
    every layer) eliminates duplicate grayscale conversions and Sobel passes
    when multiple layers run together.
    """
    image: Image.Image
    _gray: Optional[np.ndarray] = field(default=None, repr=False)
    _sobel_x: Optional[np.ndarray] = field(default=None, repr=False)
    _sobel_y: Optional[np.ndarray] = field(default=None, repr=False)

    @classmethod
    def from_image(cls, image: Image.Image) -> "LayerInputs":
        return cls(image=image)

    @property
    def gray(self) -> np.ndarray:
        if self._gray is None:
            self._gray = np.array(self.image.convert("L"), dtype=float) / 255.0
        return self._gray

    @property
    def sobel_x(self) -> np.ndarray:
        if self._sobel_x is None:
            self._sobel_x = sobel(self.gray, axis=1)
        return self._sobel_x

    @property
    def sobel_y(self) -> np.ndarray:
        if self._sobel_y is None:
            self._sobel_y = sobel(self.gray, axis=0)
        return self._sobel_y


class Layer(ABC):
    id: str = "base"

    def __init__(self, threshold: float = 0.15):
        self.threshold = threshold

    @abstractmethod
    def process(
        self,
        image: Image.Image,
        num_rows: int,
        num_cols: int,
        inputs: LayerInputs | None = None,
    ) -> CharMap:
        ...


def _to_cell_grid(arr, num_rows: int, num_cols: int):
    """Downsample 2D numpy array to (num_rows, num_cols) by block averaging."""
    h, w = arr.shape
    if h < num_rows or w < num_cols:
        raise ValueError(f"Image ({h}x{w}) is smaller than requested grid ({num_rows}x{num_cols})")
    arr = arr[:h - h % num_rows, :w - w % num_cols]
    return arr.reshape(num_rows, h // num_rows, num_cols, w // num_cols).mean(axis=(1, 3))
```

- [ ] **Step 4: Run new test + full suite**

Run: `pytest tests/test_layer_inputs.py tests/test_layers.py -v`
Expected: new tests PASS. `tests/test_layers.py` will FAIL on the four layer classes because the abstract `process` signature now has a new optional param but their concrete signatures don't match. This is OK — Task 9 fixes that.

Wait — Python abstract methods don't enforce signature compatibility, only name presence. The existing layers' `process(self, image, num_rows, num_cols)` still satisfies the ABC, and calling them with 3 args still works. Re-run to confirm:

Run: `pytest tests/test_layers.py -v`
Expected: PASS — existing layer tests still pass because they call layers with the original 3-arg signature.

- [ ] **Step 5: Commit**

```bash
git add ascii_combinator/layers/base.py tests/test_layer_inputs.py
git commit -m "feat(layers): add LayerInputs dataclass with lazy grayscale and Sobel caches"
```

---

## Task 9: Win 2b — Plumb `inputs` through `BrightnessLayer`

**Files:**
- Modify: `ascii_combinator/layers/brightness.py`
- Modify: `tests/test_layer_inputs.py`

- [ ] **Step 1: Extend the test file**

Append to `tests/test_layer_inputs.py`:

```python
from ascii_combinator.layers.brightness import BrightnessLayer


def test_brightness_layer_with_and_without_inputs_match():
    img = _img(32, 32)
    layer = BrightnessLayer()
    without = layer.process(img, num_rows=4, num_cols=4)
    inputs = LayerInputs.from_image(img)
    with_ = layer.process(img, num_rows=4, num_cols=4, inputs=inputs)
    assert without == with_


def test_brightness_layer_does_not_recompute_when_given_inputs():
    img = _img(32, 32)
    inputs = LayerInputs.from_image(img)
    # Force `gray` to be computed
    _ = inputs.gray
    cached_id = id(inputs._gray)
    BrightnessLayer().process(img, num_rows=4, num_cols=4, inputs=inputs)
    assert id(inputs._gray) == cached_id, "Layer must reuse the cached gray, not recompute"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_layer_inputs.py -v`
Expected: the two new tests FAIL because `BrightnessLayer.process` doesn't accept `inputs=`.

- [ ] **Step 3: Update `BrightnessLayer`**

Replace `ascii_combinator/layers/brightness.py` with:

```python
import numpy as np
from PIL import Image
from ascii_combinator.types import CharCell, CharMap
from ascii_combinator.layers.base import Layer, LayerInputs, _to_cell_grid

DENSITY = " .,:;+*?%S#@"  # index 0 = white/light, index 11 = black/dark


class BrightnessLayer(Layer):
    id = "brightness"

    def process(
        self,
        image: Image.Image,
        num_rows: int,
        num_cols: int,
        inputs: LayerInputs | None = None,
    ) -> CharMap:
        gray = inputs.gray if inputs is not None else (
            np.array(image.convert("L"), dtype=float) / 255.0
        )
        grid = _to_cell_grid(gray, num_rows, num_cols)
        result: CharMap = [[[] for _ in range(num_cols)] for _ in range(num_rows)]
        for r in range(num_rows):
            for c in range(num_cols):
                intensity = float(grid[r, c])
                idx = int((1.0 - intensity) * (len(DENSITY) - 1))
                char = DENSITY[idx]
                if char != " ":
                    result[r][c].append(CharCell(
                        char=char,
                        intensity=1.0 - intensity,
                        layer_id=self.id,
                    ))
        return result
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_layer_inputs.py tests/test_layers.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add ascii_combinator/layers/brightness.py tests/test_layer_inputs.py
git commit -m "perf(brightness): accept LayerInputs to reuse shared grayscale"
```

---

## Task 10: Win 2b — Plumb `inputs` through `SobelXLayer`

**Files:**
- Modify: `ascii_combinator/layers/sobel_x.py`
- Modify: `tests/test_layer_inputs.py`

- [ ] **Step 1: Extend test file**

Append to `tests/test_layer_inputs.py`:

```python
from ascii_combinator.layers.sobel_x import SobelXLayer


def test_sobel_x_layer_with_and_without_inputs_match():
    img = _img(32, 32)
    layer = SobelXLayer(threshold=0.05)
    without = layer.process(img, num_rows=4, num_cols=4)
    inputs = LayerInputs.from_image(img)
    with_ = layer.process(img, num_rows=4, num_cols=4, inputs=inputs)
    assert without == with_


def test_sobel_x_layer_reuses_sobel_x_array():
    img = _img(32, 32)
    inputs = LayerInputs.from_image(img)
    _ = inputs.sobel_x
    cached_id = id(inputs._sobel_x)
    SobelXLayer(threshold=0.05).process(img, num_rows=4, num_cols=4, inputs=inputs)
    assert id(inputs._sobel_x) == cached_id
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_layer_inputs.py -v`
Expected: the two new tests FAIL.

- [ ] **Step 3: Update `SobelXLayer`**

Replace `ascii_combinator/layers/sobel_x.py` with:

```python
import numpy as np
from scipy.ndimage import sobel
from PIL import Image
from ascii_combinator.types import CharCell, CharMap
from ascii_combinator.layers.base import Layer, LayerInputs, _to_cell_grid


class SobelXLayer(Layer):
    id = "sobel_x"

    def process(
        self,
        image: Image.Image,
        num_rows: int,
        num_cols: int,
        inputs: LayerInputs | None = None,
    ) -> CharMap:
        if inputs is not None:
            sx = inputs.sobel_x
        else:
            gray = np.array(image.convert("L"), dtype=float) / 255.0
            sx = sobel(gray, axis=1)
        edges = np.abs(sx)
        norm = edges.max()
        if norm > 0:
            edges = edges / norm
        grid = _to_cell_grid(edges, num_rows, num_cols)
        result: CharMap = [[[] for _ in range(num_cols)] for _ in range(num_rows)]
        for r in range(num_rows):
            for c in range(num_cols):
                intensity = float(grid[r, c])
                if intensity > self.threshold:
                    result[r][c].append(CharCell(char="|", intensity=intensity, layer_id=self.id))
        return result
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_layer_inputs.py tests/test_layers.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add ascii_combinator/layers/sobel_x.py tests/test_layer_inputs.py
git commit -m "perf(sobel_x): accept LayerInputs to reuse shared Sobel-X gradient"
```

---

## Task 11: Win 2b — Plumb `inputs` through `SobelYLayer`

**Files:**
- Modify: `ascii_combinator/layers/sobel_y.py`
- Modify: `tests/test_layer_inputs.py`

- [ ] **Step 1: Extend test file**

Append to `tests/test_layer_inputs.py`:

```python
from ascii_combinator.layers.sobel_y import SobelYLayer


def test_sobel_y_layer_with_and_without_inputs_match():
    img = _img(32, 32)
    layer = SobelYLayer(threshold=0.05)
    without = layer.process(img, num_rows=4, num_cols=4)
    inputs = LayerInputs.from_image(img)
    with_ = layer.process(img, num_rows=4, num_cols=4, inputs=inputs)
    assert without == with_


def test_sobel_y_layer_reuses_sobel_y_array():
    img = _img(32, 32)
    inputs = LayerInputs.from_image(img)
    _ = inputs.sobel_y
    cached_id = id(inputs._sobel_y)
    SobelYLayer(threshold=0.05).process(img, num_rows=4, num_cols=4, inputs=inputs)
    assert id(inputs._sobel_y) == cached_id
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_layer_inputs.py -v`
Expected: the two new tests FAIL.

- [ ] **Step 3: Update `SobelYLayer`**

Replace `ascii_combinator/layers/sobel_y.py` with:

```python
import numpy as np
from scipy.ndimage import sobel
from PIL import Image
from ascii_combinator.types import CharCell, CharMap
from ascii_combinator.layers.base import Layer, LayerInputs, _to_cell_grid


class SobelYLayer(Layer):
    id = "sobel_y"

    def process(
        self,
        image: Image.Image,
        num_rows: int,
        num_cols: int,
        inputs: LayerInputs | None = None,
    ) -> CharMap:
        if inputs is not None:
            sy = inputs.sobel_y
        else:
            gray = np.array(image.convert("L"), dtype=float) / 255.0
            sy = sobel(gray, axis=0)
        edges = np.abs(sy)
        norm = edges.max()
        if norm > 0:
            edges = edges / norm
        grid = _to_cell_grid(edges, num_rows, num_cols)
        result: CharMap = [[[] for _ in range(num_cols)] for _ in range(num_rows)]
        for r in range(num_rows):
            for c in range(num_cols):
                intensity = float(grid[r, c])
                if intensity > self.threshold:
                    result[r][c].append(CharCell(char="─", intensity=intensity, layer_id=self.id))
        return result
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_layer_inputs.py tests/test_layers.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add ascii_combinator/layers/sobel_y.py tests/test_layer_inputs.py
git commit -m "perf(sobel_y): accept LayerInputs to reuse shared Sobel-Y gradient"
```

---

## Task 12: Win 2b — Plumb `inputs` through `DiagonalLayer`

**Files:**
- Modify: `ascii_combinator/layers/diagonal.py`
- Modify: `tests/test_layer_inputs.py`

- [ ] **Step 1: Extend test file**

Append to `tests/test_layer_inputs.py`:

```python
from ascii_combinator.layers.diagonal import DiagonalLayer


def test_diagonal_layer_with_and_without_inputs_match():
    img = _img(32, 32)
    layer = DiagonalLayer(threshold=0.05)
    without = layer.process(img, num_rows=4, num_cols=4)
    inputs = LayerInputs.from_image(img)
    with_ = layer.process(img, num_rows=4, num_cols=4, inputs=inputs)
    assert without == with_


def test_diagonal_layer_reuses_both_sobel_arrays():
    img = _img(32, 32)
    inputs = LayerInputs.from_image(img)
    _ = inputs.sobel_x
    _ = inputs.sobel_y
    cached_sx = id(inputs._sobel_x)
    cached_sy = id(inputs._sobel_y)
    DiagonalLayer(threshold=0.05).process(img, num_rows=4, num_cols=4, inputs=inputs)
    assert id(inputs._sobel_x) == cached_sx
    assert id(inputs._sobel_y) == cached_sy
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_layer_inputs.py -v`
Expected: the two new tests FAIL.

- [ ] **Step 3: Update `DiagonalLayer`**

Replace `ascii_combinator/layers/diagonal.py` with:

```python
import numpy as np
from scipy.ndimage import sobel
from PIL import Image
from ascii_combinator.types import CharCell, CharMap
from ascii_combinator.layers.base import Layer, LayerInputs, _to_cell_grid


class DiagonalLayer(Layer):
    id = "diagonal"

    def process(
        self,
        image: Image.Image,
        num_rows: int,
        num_cols: int,
        inputs: LayerInputs | None = None,
    ) -> CharMap:
        if inputs is not None:
            gx = inputs.sobel_x
            gy = inputs.sobel_y
        else:
            gray = np.array(image.convert("L"), dtype=float) / 255.0
            gx = sobel(gray, axis=1)
            gy = sobel(gray, axis=0)
        magnitude = np.hypot(gx, gy)
        norm = magnitude.max()
        if norm > 0:
            magnitude = magnitude / norm

        gx_grid = _to_cell_grid(gx, num_rows, num_cols)
        gy_grid = _to_cell_grid(gy, num_rows, num_cols)
        mag_grid = _to_cell_grid(magnitude, num_rows, num_cols)
        angle_grid = np.arctan2(gy_grid, gx_grid)

        result: CharMap = [[[] for _ in range(num_cols)] for _ in range(num_rows)]
        for r in range(num_rows):
            for c in range(num_cols):
                intensity = float(mag_grid[r, c])
                if intensity <= self.threshold:
                    continue
                a = float(angle_grid[r, c]) % np.pi
                if np.pi / 8 < a < 3 * np.pi / 8:
                    char = "/"
                elif 5 * np.pi / 8 < a < 7 * np.pi / 8:
                    char = "\\"
                else:
                    continue
                result[r][c].append(CharCell(char=char, intensity=intensity, layer_id=self.id))
        return result
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_layer_inputs.py tests/test_layers.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add ascii_combinator/layers/diagonal.py tests/test_layer_inputs.py
git commit -m "perf(diagonal): accept LayerInputs to reuse Sobel-X and Sobel-Y gradients"
```

---

## Task 13: Win 2c — Wire orchestrators (CLI, video, web_ui, benchmarks)

**Files:**
- Modify: `ascii_combinator/cli.py:_run_image`
- Modify: `ascii_combinator/video.py:FrameProcessor.process`
- Modify: `web_ui.py:_convert_image`
- Modify: `benchmarks/run.py:run_image_scenario` and `run_video_scenario`

This task has no new tests — the existing test suite (incl. `test_cli.py`, `test_video.py`, `test_web_ui.py`) already exercises these code paths end-to-end. They must continue to pass.

- [ ] **Step 1: Modify `ascii_combinator/cli.py`**

In `_run_image`, change the lines that currently read:

```python
    layers = [LAYER_REGISTRY[n](threshold=args.threshold) for n in layer_names]
    charmap_list = [layer.process(image, num_rows, num_cols) for layer in layers]
```

to:

```python
    from ascii_combinator.layers.base import LayerInputs
    layer_inputs = LayerInputs.from_image(image)
    layers = [LAYER_REGISTRY[n](threshold=args.threshold) for n in layer_names]
    charmap_list = [
        layer.process(image, num_rows, num_cols, inputs=layer_inputs)
        for layer in layers
    ]
```

- [ ] **Step 2: Modify `ascii_combinator/video.py:FrameProcessor.process`**

Change the body that currently reads:

```python
        layers = [_LAYER_REGISTRY[n](threshold=config.threshold) for n in config.layer_names]
        charmap_list = [layer.process(image, num_rows, num_cols) for layer in layers]
```

to:

```python
        from ascii_combinator.layers.base import LayerInputs
        layer_inputs = LayerInputs.from_image(image)
        layers = [_LAYER_REGISTRY[n](threshold=config.threshold) for n in config.layer_names]
        charmap_list = [
            layer.process(image, num_rows, num_cols, inputs=layer_inputs)
            for layer in layers
        ]
```

- [ ] **Step 3: Modify `web_ui.py:_convert_image`**

Change the lines that currently read:

```python
    layers = [_LAYER_REGISTRY[n](threshold=threshold) for n in layer_names if n in _LAYER_REGISTRY]
    charmap_list = [layer.process(image, num_rows, num_cols) for layer in layers]
```

to:

```python
    from ascii_combinator.layers.base import LayerInputs
    layer_inputs = LayerInputs.from_image(image)
    layers = [_LAYER_REGISTRY[n](threshold=threshold) for n in layer_names if n in _LAYER_REGISTRY]
    charmap_list = [
        layer.process(image, num_rows, num_cols, inputs=layer_inputs)
        for layer in layers
    ]
```

- [ ] **Step 4: Modify `benchmarks/run.py` — image runner**

In `run_image_scenario`, change:

```python
            layers = [LAYER_REGISTRY[n](threshold=scen.threshold) for n in scen.layers]
            charmap_list = []
            for layer in layers:
                with stage(f"{scen.id}.layer.{layer.id}.process", registry=reg):
                    charmap_list.append(layer.process(image, num_rows, num_cols))
```

to:

```python
            from ascii_combinator.layers.base import LayerInputs
            with stage(f"{scen.id}.layer.inputs.build", registry=reg):
                layer_inputs = LayerInputs.from_image(image)
                # Touch lazy properties so the build cost lands here, not in the first layer
                _ = layer_inputs.gray
                _ = layer_inputs.sobel_x
                _ = layer_inputs.sobel_y
            layers = [LAYER_REGISTRY[n](threshold=scen.threshold) for n in scen.layers]
            charmap_list = []
            for layer in layers:
                with stage(f"{scen.id}.layer.{layer.id}.process", registry=reg):
                    charmap_list.append(
                        layer.process(image, num_rows, num_cols, inputs=layer_inputs)
                    )
```

- [ ] **Step 5: Modify `benchmarks/run.py` — video runner**

In `run_video_scenario`, change:

```python
            for frame_in in frames_in:
                image = Image.open(frame_in)
                num_rows, num_cols = _grid_dims(image, scen.out_width, scen.font_size)
                charmap_list = []
                for layer in layers:
                    with stage(f"{scen.id}.frame.layer.{layer.id}", registry=reg):
                        charmap_list.append(layer.process(image, num_rows, num_cols))
```

to:

```python
            from ascii_combinator.layers.base import LayerInputs
            for frame_in in frames_in:
                image = Image.open(frame_in)
                num_rows, num_cols = _grid_dims(image, scen.out_width, scen.font_size)
                with stage(f"{scen.id}.frame.layer.inputs.build", registry=reg):
                    layer_inputs = LayerInputs.from_image(image)
                    _ = layer_inputs.gray
                    _ = layer_inputs.sobel_x
                    _ = layer_inputs.sobel_y
                charmap_list = []
                for layer in layers:
                    with stage(f"{scen.id}.frame.layer.{layer.id}", registry=reg):
                        charmap_list.append(
                            layer.process(image, num_rows, num_cols, inputs=layer_inputs)
                        )
```

- [ ] **Step 6: Run the full test suite**

Run: `pytest -v`
Expected: all tests PASS (21+ existing + new layer-inputs tests + fixture/instrument tests).

- [ ] **Step 7: Commit**

```bash
git add ascii_combinator/cli.py ascii_combinator/video.py web_ui.py benchmarks/run.py
git commit -m "perf: share LayerInputs across layers in CLI, video, web_ui, and bench"
```

---

## Task 14: Re-bench and decide on Win 3

**Files:**
- Modify: `benchmarks/baseline.json` (regenerated)
- Create: `benchmarks/visual_smoke/s1_after.png`

- [ ] **Step 1: Re-run the full benchmark**

Run:

```bash
python -m benchmarks.run --scenarios s1,s2,s3 --repeats 3 \
    --write-baseline --visual-tag after
```

Expected: prints three tables; writes `benchmarks/baseline.json` and `benchmarks/visual_smoke/s1_after.png`.

- [ ] **Step 2: Manual visual smoke check**

Open `benchmarks/visual_smoke/s1_baseline.png` and `benchmarks/visual_smoke/s1_after.png` side by side. They should look essentially identical (same seed, same logic). If they differ visibly, a bug was introduced — STOP and investigate before continuing.

- [ ] **Step 3: Compare numbers and decide on Win 3**

Compare `benchmarks/baseline.json` (after) against `benchmarks/baseline_initial.json` (before). Write down the delta for each scenario's `total_median_s` (expected: improvement, especially on S1/S2 which run all four layers).

Look at the new top-3 stages by `share_pct` for S1 and S2 (skipping `.total`). Decide:

- **If a layer's Python-loop `process` step (e.g. `s1.layer.brightness.process`) is in the top-3 AND vectorizing its loop would fit in ≤ ~30 LOC per layer:** proceed to Task 15.
- **Otherwise:** skip Task 15 entirely and proceed to Task 16.

Document the decision in a one-paragraph note appended to the commit message in Step 4 (something like "post-Win2 top stages: renderer.render (61%), compositor (12%), layer.brightness (8%) — Win 3 skipped; next perf spec should target renderer glyph cache").

- [ ] **Step 4: Commit**

```bash
git add benchmarks/baseline.json benchmarks/visual_smoke/s1_after.png
git commit -m "bench: post-Win1-Win2 baseline + visual smoke

<your one-paragraph note from Step 3>"
```

---

## Task 15: Win 3 — Vectorize layer inner loops (CONDITIONAL — only if Task 14 Step 3 said so)

**Files:**
- Modify: each layer file flagged in Task 14 Step 3
- Modify: `tests/test_layers.py` (only if existing tests don't already assert byte-equality)

Skip this task entirely if Task 14 Step 3 decided against it.

**Approach for each flagged layer:** replace the `for r in range(...): for c in range(...):` block that constructs `CharCell` objects with a numpy-vectorized mask + a single Python pass over only the cells that pass the threshold.

Example for `BrightnessLayer` (apply the same pattern to others as needed):

- [ ] **Step 1: Write a stronger test asserting CharMap equality before vs. after**

Append to `tests/test_layer_inputs.py`:

```python
def test_brightness_vectorization_preserves_output():
    """Brightness output must be unchanged after vectorization for a representative input."""
    img = _img(64, 64)
    inputs = LayerInputs.from_image(img)
    result = BrightnessLayer().process(img, num_rows=8, num_cols=8, inputs=inputs)
    # Spot check: at least one cell has a char from DENSITY
    from ascii_combinator.layers.brightness import DENSITY
    chars = {cell.char for row in result for cells in row for cell in cells}
    assert chars.issubset(set(DENSITY) - {" "})
```

- [ ] **Step 2: Run the existing layer tests to capture current behaviour**

Run: `pytest tests/test_layers.py tests/test_layer_inputs.py -v`
Expected: all PASS — confirms baseline behaviour before changing internals.

- [ ] **Step 3: Vectorize `BrightnessLayer` inner loop**

Replace the `for r ... for c ...` block in `ascii_combinator/layers/brightness.py:process` with:

```python
        # Vectorized: compute char-index grid first, then iterate only over non-space cells.
        intensity_grid = 1.0 - grid                     # shape (num_rows, num_cols)
        idx_grid = (intensity_grid * (len(DENSITY) - 1)).astype(int)
        idx_grid = np.clip(idx_grid, 0, len(DENSITY) - 1)
        chars_arr = np.array(list(DENSITY))             # shape (len(DENSITY),)
        char_grid = chars_arr[idx_grid]                 # shape (num_rows, num_cols), dtype <U1

        non_space_rs, non_space_cs = np.where(char_grid != " ")
        for r, c in zip(non_space_rs.tolist(), non_space_cs.tolist()):
            result[r][c].append(CharCell(
                char=str(char_grid[r, c]),
                intensity=float(intensity_grid[r, c]),
                layer_id=self.id,
            ))
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_layers.py tests/test_layer_inputs.py -v`
Expected: all PASS.

- [ ] **Step 5: Apply analogous vectorization to other flagged layers**

For each additional layer flagged in Task 14 Step 3, repeat Steps 1–4 with the appropriate vectorization (Sobel layers: build a single `intensity > threshold` boolean mask, iterate only over True positions; diagonal layer: compute `char_grid` from the angle masks, then iterate non-empty cells).

- [ ] **Step 6: Re-run bench and update baseline.json**

Run:

```bash
python -m benchmarks.run --scenarios s1,s2,s3 --repeats 3 \
    --write-baseline --visual-tag after
```

This overwrites `benchmarks/baseline.json` and `benchmarks/visual_smoke/s1_after.png` from Task 14.

- [ ] **Step 7: Commit**

```bash
git add ascii_combinator/layers/ tests/test_layer_inputs.py benchmarks/baseline.json benchmarks/visual_smoke/s1_after.png
git commit -m "perf(layers): vectorize inner CharCell construction loops"
```

---

## Task 16: Final sweep and README note

**Files:**
- Modify: `benchmarks/__init__.py` (one-liner)
- (Optional) update top-of-repo doc if there is one

- [ ] **Step 1: Add a docstring to `benchmarks/__init__.py`**

Replace `benchmarks/__init__.py` contents with:

```python
"""ASCII Combinator benchmark harness.

Run: `python -m benchmarks.run --scenarios s1,s2,s3 --repeats 3 --write-baseline`

Compare `benchmarks/baseline.json` (latest) to `benchmarks/baseline_initial.json`
(historical anchor) to track regressions and improvements.
"""
```

- [ ] **Step 2: Run the full test suite one last time**

Run: `pytest -v`
Expected: all tests PASS, including all the new ones from Tasks 1, 2, 8–12 (and 15 if executed).

- [ ] **Step 3: Verify the bench still runs cleanly**

Run: `python -m benchmarks.run --scenarios s2 --repeats 1`
Expected: exits 0, prints a table.

- [ ] **Step 4: Commit**

```bash
git add benchmarks/__init__.py
git commit -m "docs(bench): module docstring with run instructions"
```

---

## Done.

What was accomplished:
- Reproducible benchmark harness with three scenarios and stage timing.
- Historical anchor: `benchmarks/baseline_initial.json` (before any optimization).
- Current baseline: `benchmarks/baseline.json` (after optimizations).
- Visual smoke PNGs to catch unintended rendering changes.
- Two confirmed optimizations: font LRU cache + shared `LayerInputs`.
- Optional third optimization (layer-loop vectorization) applied if data warranted it.

Open follow-ups for future perf specs (do NOT do them here):
- Renderer glyph pre-rasterization (highest expected payoff).
- `CharMap` → numpy-backed representation.
- Web UI video pipeline parallelism with chunked progress.
- `rembg` segmentation optimization.

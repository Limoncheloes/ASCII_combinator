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
import statistics
import subprocess
import sys
import tempfile
import time
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

from benchmarks.fixtures import make_synthetic_image
from benchmarks.instrument import StageRegistry, stage
from benchmarks.scenarios import ALL_IMAGE, ALL_VIDEO, ImageScenario, VideoScenario, S2

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
    totals: list[float] = []

    for _ in range(repeats):
        t0 = time.perf_counter()
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
        totals.append(time.perf_counter() - t0)

    if save_visual_as is not None and last_result is not None:
        save_visual_as.parent.mkdir(parents=True, exist_ok=True)
        last_result.save(save_visual_as)

    return reg, statistics.median(totals)


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

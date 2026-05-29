# Performance: Profiling Harness + Easy Wins

**Date:** 2026-05-29
**Status:** Draft — awaiting user review
**Track:** C (Performance) — first of four planned tracks (C → B → D → A)

## Goal

1. Get a reproducible baseline of where time goes in the pipeline (image and video).
2. Apply only the optimizations that the baseline shows are worth it AND fit ≤ ~100 LOC each without changing public APIs (CLI/HTTP/Layer interface remain compatible).
3. Commit before/after numbers so future perf work has a starting point.

Non-goals: `rembg` optimization, GPU/numba/cython, structural rewrites of `CharMap`, parallelism changes in `web_ui` video flow. Each of these will be a separate spec if the baseline justifies it.

## Reference scenarios

| ID  | Input                         | Params                          | Target |
| --- | ----------------------------- | ------------------------------- | ------ |
| S1  | 4000×3000 synthetic noise PNG | `width=140`, `font_size=12`     | ≤ 2 s  |
| S2  | 4000×3000 synthetic noise PNG | `width=100`, `font_size=12`     | ≤ 5 s  |
| S3  | 720p × 10 s MP4 @ 24 fps      | `width=100`, `fps=10`, all layers | ≤ 60 s |

"Target" = user-agreed acceptable wall-clock for that scenario on the user's machine.
Synthetic inputs eliminate cross-machine drift in image content and keep the harness self-contained.

## Architecture

### New module: `benchmarks/`

```
benchmarks/
  __init__.py
  fixtures.py        # generate synthetic image/video inputs into /tmp (not committed)
  scenarios.py       # S1/S2/S3 definitions as dataclasses
  instrument.py      # Stage timer context manager + global registry
  run.py             # CLI entrypoint: `python -m benchmarks.run`
  baseline.json      # checked into git — current baseline numbers
```

`run.py` flags:
- `--scenarios s1,s2,s3` (default: all)
- `--repeats N` (default: 3, reports min + median)
- `--out path/to/result.json` (default: stdout + `baseline.json` if `--write-baseline`)
- `--profile` → also runs `cProfile` on S2 and dumps top-30 functions

### Instrumentation strategy

Lightweight stage timing via `time.perf_counter()` and a context manager:

```python
# benchmarks/instrument.py
with stage("layer.brightness"):
    cm = BrightnessLayer().process(image, rows, cols)
```

Stages we'll measure (covers ≥ 95% of pipeline wall time):

**Image pipeline:**
- `layer.<id>.process` — per layer
- `compositor.composite`
- `renderer.render`
- `image.io.load`, `image.io.save`

**Video pipeline:** all of the above per frame, plus:
- `video.ffmpeg.extract`
- `video.ffmpeg.assemble_mp4`

Instrumentation hooks live in `benchmarks/instrument.py` and are *imported by the bench harness only*. Source modules stay clean — the harness does the timing by wrapping public methods (monkey-patch in `run.py` setup, or simple wrapper subclasses). **No `time.perf_counter()` calls leak into `ascii_combinator/`.**

### Report format

`baseline.json` shape:

```json
{
  "generated_at": "2026-05-29T...",
  "git_sha": "...",
  "host": {"cpu_count": 8, "python": "3.11.x", "platform": "linux-..."},
  "scenarios": {
    "s1": {
      "total_median_s": 1.83,
      "stages": {
        "layer.brightness.process": {"median_s": 0.21, "share_pct": 11.5},
        "layer.sobel_x.process":    {"median_s": 0.18, "share_pct": 9.8},
        "...": {}
      }
    }
  }
}
```

Console output: a single table per scenario, sorted by share %.

## Inline wins (apply if baseline confirms them in top stages)

Based on a code read, these three optimizations look obvious:

### Win 1 — LRU-cache the font loader

`renderer._find_font` is called on every `render()`. For video, that's N frames × probing up to 4 font paths each. Trivial fix:

```python
from functools import lru_cache

@lru_cache(maxsize=8)
def _find_font(size: int): ...
```

Expected delta: small per-image, meaningful per-video (~1–5%). Cost: 1 line. Definitely include.

### Win 2 — Share precomputed grayscale across layers

`BrightnessLayer`, `SobelXLayer`, `SobelYLayer`, `DiagonalLayer` each do `np.array(image.convert("L"), dtype=float) / 255.0`. For 4 layers that's 4× the same grayscale conversion. For 4K image this is non-trivial.

Approach: introduce an optional `LayerInputs` dataclass passed to `Layer.process()`:

```python
# ascii_combinator/layers/base.py
@dataclass
class LayerInputs:
    gray: np.ndarray              # lazily computed on first access
    sobel_x: np.ndarray | None    # lazily computed on first access
    sobel_y: np.ndarray | None
    # ...
    @classmethod
    def from_image(cls, image): ...
```

Update `Layer.process(self, image, num_rows, num_cols, inputs: LayerInputs | None = None)`. Default `None` means "compute on the fly" — keeps single-layer external callers working. Pipeline orchestrators (`web_ui._convert_image`, `cli`, `video.FrameProcessor`) construct one `LayerInputs` and pass it to all layers.

Expected delta: 4× → 1× grayscale conversion + reused Sobel gradients (sobel_x/sobel_y/diagonal all use `sobel(gray)`). Could be 10–25% on multi-layer image scenarios.

Cost estimate: ~80 LOC across `layers/base.py` + 4 layer files + 3 orchestrator sites. Backward compatible.

### Win 3 — Vectorize the inner Python loops in layers

Each layer ends with `for r in range(num_rows): for c in range(num_cols):` building Python `CharCell` objects. For width=200, that's ~30k iterations per layer × 4 layers = ~120k objects per image. The character/threshold logic vectorizes cleanly with numpy boolean masks; only the final `CharCell` construction stays in Python (and only for cells above threshold).

Apply this **only if** the baseline shows layer Python loops as a top-3 stage. Otherwise defer.

Cost: ~30 LOC per layer if applied.

### Wins explicitly deferred (next spec)

- **Glyph pre-rasterization in `Renderer`** — replace ~10k `draw.text()` calls with one PIL `paste()` per glyph variant. Almost certainly the biggest single win, but structural (changes how `Renderer` works internally). Suspected #1 hotspot. Will spec separately once the baseline confirms it.
- **CharMap → numpy-backed structure** — would unlock vectorized Compositor and Renderer. Touches `types.py` (public-ish) and every layer. Substantial — own spec.
- **Parallel video rendering in `web_ui`** — currently sequential for progress UX. Tradeoff design needed (chunked progress vs. full parallelism).

## Acceptance criteria

1. `python -m benchmarks.run` runs end-to-end in < 60 s on the user's machine.
2. `benchmarks/baseline.json` committed with first numbers.
3. Wins 1 and 2 implemented (Win 3 only if baseline justifies it).
4. `benchmarks/baseline.json` re-generated after fixes — both `before/` and `after/` columns visible (we'll keep two JSONs: `baseline_initial.json` checked in once as historical reference, `baseline.json` updated to latest).
5. All existing 21+ tests pass.
6. New test in `tests/test_layer_inputs.py`: layer with and without `inputs=` produces identical `CharMap` for same image.
7. Visual smoke test: bench harness saves a rendered S1 sample PNG before and after fixes; opening them looks identical (no automated pixel diff — manual check, but seed is fixed).

## File-by-file plan (preview)

- `benchmarks/__init__.py` — new, empty
- `benchmarks/fixtures.py` — new, ~40 LOC
- `benchmarks/scenarios.py` — new, ~30 LOC
- `benchmarks/instrument.py` — new, ~50 LOC
- `benchmarks/run.py` — new, ~120 LOC
- `benchmarks/baseline.json` — new, generated
- `benchmarks/baseline_initial.json` — new, historical
- `ascii_combinator/renderer.py` — +1 line (`@lru_cache`)
- `ascii_combinator/layers/base.py` — +`LayerInputs` dataclass, ~30 LOC
- `ascii_combinator/layers/{brightness,sobel_x,sobel_y,diagonal}.py` — accept `inputs=`, ~5 LOC each
- `ascii_combinator/video.py`, `web_ui.py`, `ascii_combinator/cli.py` — construct `LayerInputs` once and pass through, ~10 LOC each
- `tests/test_layer_inputs.py` — new, ~40 LOC

Total: ~400 LOC new, ~50 LOC modified. Within "easy wins" budget.

## Open questions

None blocking. If Win 3 (layer vectorization) is needed, decide after baseline lands.

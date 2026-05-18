# Subject Segmentation & Background Control

**Date:** 2026-05-12
**Status:** Approved

## Summary

Add subject detection to ASCII Combinator so that only the main element of a photo is processed as full ASCII, while the background can be removed entirely or rendered with lighter symbols. Uses `rembg` (U2Net ML model) as an optional dependency for segmentation.

## New Components

### `ascii_combinator/bg_mode.py`

```python
from enum import Enum
from dataclasses import dataclass, field

class BgMode(Enum):
    KEEP   = "keep"    # no segmentation, current behaviour
    REMOVE = "remove"  # background cells are empty
    SOFT   = "soft"    # background cells rendered with reduced opacity and restricted chars

@dataclass
class SoftBgConfig:
    opacity: float = 0.25   # intensity multiplier for background CharCells
    chars: str = ".,"       # allowed characters for background cells
```

### `ascii_combinator/segmentation.py`

```python
SubjectMask = list[list[bool]]  # True = subject, False = background

class Segmenter:
    def segment(self, image: Image.Image, num_rows: int, num_cols: int) -> SubjectMask:
        ...
```

**Implementation:**
1. Call `rembg.remove(image)` → RGBA image
2. Extract alpha channel as numpy array
3. Downsample to `(num_rows, num_cols)` via block averaging (reuse `_to_cell_grid` from `layers/base.py`)
4. Threshold at 0.5 → boolean grid

`rembg` is imported lazily inside `Segmenter.segment()`. If not installed, raises `ImportError` with a helpful message.

## Modified Components

### `ascii_combinator/compositor.py`

Extended signature (backwards compatible — all new params have defaults):

```python
def composite(
    self,
    charmap_list: list[CharMap],
    mask: SubjectMask | None = None,
    bg_mode: BgMode = BgMode.KEEP,
    soft_cfg: SoftBgConfig | None = None,
) -> CharMap:
```

Post-merge logic applied cell by cell:

| Condition | Action |
|-----------|--------|
| `mask is None` or `bg_mode == KEEP` | return result unchanged |
| `mask[r][c] == True` (subject) | leave cell unchanged |
| `mask[r][c] == False` + `REMOVE` | `result[r][c] = []` |
| `mask[r][c] == False` + `SOFT` | filter chars to `soft_cfg.chars`, multiply `intensity` by `soft_cfg.opacity`; if no chars remain → `[]` |

`Renderer` requires no changes — it already skips empty cells and reads `intensity` via `profile.color_for()`.

## CLI Changes (`ascii_combinator/cli.py`)

New arguments:

| Argument | Type | Default | Notes |
|----------|------|---------|-------|
| `--bg-mode` | `keep\|remove\|soft` | `keep` | Selects segmentation mode |
| `--bg-opacity` | `float` | `0.25` | Only used with `soft` |
| `--bg-chars` | `str` | `".,"`  | Only used with `soft` |

Flow:

```
if --bg-mode == keep:
    mask = None
else:
    try import rembg → on ImportError: print install hint, exit
    mask = Segmenter().segment(image, num_rows, num_cols)

soft_cfg = SoftBgConfig(opacity, chars) if bg_mode == soft else None
charmap = Compositor().composite(charmap_list, mask, bg_mode, soft_cfg)
```

Example usage:

```bash
# subject only, background empty
python -m ascii_combinator photo.jpg --bg-mode remove

# background with light dots at 20% opacity
python -m ascii_combinator photo.jpg --bg-mode soft --bg-opacity 0.2 --bg-chars ".·"
```

## Data Flow

```
Image
  │
  ├─→ [existing] Layer × N → CharMap × N → Compositor (merge)
  │                                              │
  └─→ [new] Segmenter → SubjectMask ────────────┤
                                                 │ apply mask + BgMode
                                                 ▼
                                           CharMap (masked)
                                                 │
                                           Renderer → PNG
```

## Dependencies

- `rembg` — optional. Added to `requirements.txt` as a comment with install instructions. Not imported at module level.
- No changes to existing dependencies (Pillow, numpy, scipy).

## Testing

| Test | Description |
|------|-------------|
| `test_segmenter_mock` | Mock `rembg.remove` returning known RGBA; verify mask shape and values |
| `test_compositor_remove` | Mask with known background cells → those cells empty in output |
| `test_compositor_soft` | Background cells have reduced intensity and only `soft_cfg.chars` symbols |
| `test_compositor_keep` | `BgMode.KEEP` → output identical to current behaviour |
| `test_cli_bg_mode_keep` | CLI without `--bg-mode` → no segmentation called |
| `test_cli_rembg_missing` | `rembg` not installed → clear error message, non-zero exit |

## Edge Cases

- Image smaller than cell grid → already guarded in `_to_cell_grid`, same error applies to Segmenter
- All cells masked as background (solid-colour photo) → `REMOVE` returns all-empty CharMap; Renderer produces blank image — acceptable
- `--bg-opacity 0.0` → background cells get intensity 0 (invisible); effectively same as REMOVE but chars still present in CharMap — acceptable
- `--bg-chars ""` (empty string) → every background cell becomes empty in SOFT mode — same as REMOVE; no special handling needed

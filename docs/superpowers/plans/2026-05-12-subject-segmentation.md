# Subject Segmentation & Background Control — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `--bg-mode keep|remove|soft` to ASCII Combinator CLI to isolate the main subject using rembg and control background rendering.

**Architecture:** A new `Segmenter` produces a boolean `SubjectMask` from rembg's alpha output. `Compositor.composite()` accepts the mask and applies per-cell background logic after merging layers. rembg is an optional dependency — missing it raises a clear error only when `--bg-mode != keep`.

**Tech Stack:** Python 3.12, Pillow, NumPy, rembg (optional), pytest

---

## File Map

| Path | Action | Responsibility |
|------|--------|----------------|
| `ascii_combinator/bg_mode.py` | Create | BgMode enum + SoftBgConfig dataclass |
| `ascii_combinator/segmentation.py` | Create | Segmenter class + SubjectMask type |
| `ascii_combinator/compositor.py` | Modify | Accept mask + bg_mode in composite() |
| `ascii_combinator/cli.py` | Modify | --bg-mode, --bg-opacity, --bg-chars args |
| `tests/test_bg_mode.py` | Create | Tests for BgMode and SoftBgConfig defaults |
| `tests/test_segmentation.py` | Create | Tests for Segmenter (mocked rembg) |
| `tests/test_compositor.py` | Modify | Add mask-aware composite tests |
| `tests/test_cli.py` | Modify | Add --bg-mode integration tests |

---

### Task 1: BgMode and SoftBgConfig

**Files:**
- Create: `ascii_combinator/bg_mode.py`
- Create: `tests/test_bg_mode.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_bg_mode.py
from ascii_combinator.bg_mode import BgMode, SoftBgConfig


def test_bgmode_values():
    assert BgMode.KEEP.value == "keep"
    assert BgMode.REMOVE.value == "remove"
    assert BgMode.SOFT.value == "soft"


def test_softbgconfig_defaults():
    cfg = SoftBgConfig()
    assert cfg.opacity == 0.25
    assert cfg.chars == ".,"


def test_softbgconfig_custom():
    cfg = SoftBgConfig(opacity=0.1, chars=".")
    assert cfg.opacity == 0.1
    assert cfg.chars == "."
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_bg_mode.py -v
```
Expected: `ImportError: cannot import name 'BgMode'`

- [ ] **Step 3: Implement**

```python
# ascii_combinator/bg_mode.py
from dataclasses import dataclass
from enum import Enum


class BgMode(Enum):
    KEEP = "keep"
    REMOVE = "remove"
    SOFT = "soft"


@dataclass
class SoftBgConfig:
    opacity: float = 0.25
    chars: str = ".,"
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_bg_mode.py -v
```
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add ascii_combinator/bg_mode.py tests/test_bg_mode.py
git commit -m "feat: add BgMode enum and SoftBgConfig dataclass"
```

---

### Task 2: Segmenter

**Files:**
- Create: `ascii_combinator/segmentation.py`
- Create: `tests/test_segmentation.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_segmentation.py
import sys
import importlib
import numpy as np
import pytest
from unittest.mock import patch
from PIL import Image
from ascii_combinator.segmentation import Segmenter, SubjectMask


def _make_rgba(h: int, w: int, alpha: np.ndarray) -> Image.Image:
    arr = np.zeros((h, w, 4), dtype=np.uint8)
    arr[:, :, 3] = alpha
    return Image.fromarray(arr, mode="RGBA")


def test_segmenter_returns_correct_shape():
    """Mask must be num_rows × num_cols."""
    alpha = np.full((32, 32), 255, dtype=np.uint8)
    rgba_img = _make_rgba(32, 32, alpha)

    with patch("ascii_combinator.segmentation.rembg") as mock_rembg:
        mock_rembg.remove.return_value = rgba_img
        mask = Segmenter().segment(Image.new("RGB", (32, 32)), num_rows=4, num_cols=4)

    assert len(mask) == 4
    assert all(len(row) == 4 for row in mask)


def test_segmenter_subject_cells_true():
    """Cells with high alpha → True (subject)."""
    alpha = np.full((32, 32), 255, dtype=np.uint8)
    rgba_img = _make_rgba(32, 32, alpha)

    with patch("ascii_combinator.segmentation.rembg") as mock_rembg:
        mock_rembg.remove.return_value = rgba_img
        mask = Segmenter().segment(Image.new("RGB", (32, 32)), num_rows=4, num_cols=4)

    assert all(mask[r][c] is True for r in range(4) for c in range(4))


def test_segmenter_background_cells_false():
    """Cells with zero alpha → False (background)."""
    alpha = np.zeros((32, 32), dtype=np.uint8)
    rgba_img = _make_rgba(32, 32, alpha)

    with patch("ascii_combinator.segmentation.rembg") as mock_rembg:
        mock_rembg.remove.return_value = rgba_img
        mask = Segmenter().segment(Image.new("RGB", (32, 32)), num_rows=4, num_cols=4)

    assert all(mask[r][c] is False for r in range(4) for c in range(4))


def test_segmenter_mixed_mask():
    """Left half subject, right half background."""
    alpha = np.zeros((32, 32), dtype=np.uint8)
    alpha[:, :16] = 255
    rgba_img = _make_rgba(32, 32, alpha)

    with patch("ascii_combinator.segmentation.rembg") as mock_rembg:
        mock_rembg.remove.return_value = rgba_img
        mask = Segmenter().segment(Image.new("RGB", (32, 32)), num_rows=4, num_cols=4)

    for r in range(4):
        assert mask[r][0] is True
        assert mask[r][1] is True
        assert mask[r][2] is False
        assert mask[r][3] is False


def test_segmenter_rembg_missing():
    """Clear ImportError when rembg is not installed."""
    original = sys.modules.get("rembg")
    sys.modules["rembg"] = None  # type: ignore
    import ascii_combinator.segmentation as seg_mod
    importlib.reload(seg_mod)
    try:
        with pytest.raises(ImportError, match="pip install rembg"):
            seg_mod.Segmenter().segment(Image.new("RGB", (32, 32)), num_rows=4, num_cols=4)
    finally:
        if original is None:
            sys.modules.pop("rembg", None)
        else:
            sys.modules["rembg"] = original
        importlib.reload(seg_mod)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_segmentation.py -v
```
Expected: `ImportError: cannot import name 'Segmenter'`

- [ ] **Step 3: Implement**

```python
# ascii_combinator/segmentation.py
import numpy as np
from PIL import Image

SubjectMask = list[list[bool]]

try:
    import rembg as rembg
except ImportError:
    rembg = None  # type: ignore


class Segmenter:
    def segment(self, image: Image.Image, num_rows: int, num_cols: int) -> SubjectMask:
        if rembg is None:
            raise ImportError(
                "rembg is required for subject segmentation. "
                "Install it with: pip install rembg"
            )
        rgba = rembg.remove(image)
        alpha = np.array(rgba)[:, :, 3].astype(float) / 255.0

        h, w = alpha.shape
        alpha = alpha[: h - h % num_rows, : w - w % num_cols]
        block_h = h // num_rows
        block_w = w // num_cols
        averaged = alpha.reshape(num_rows, block_h, num_cols, block_w).mean(axis=(1, 3))

        return [[bool(averaged[r][c] >= 0.5) for c in range(num_cols)] for r in range(num_rows)]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_segmentation.py -v
```
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add ascii_combinator/segmentation.py tests/test_segmentation.py
git commit -m "feat: add Segmenter with rembg-based subject mask extraction"
```

---

### Task 3: Compositor mask support

**Files:**
- Modify: `ascii_combinator/compositor.py`
- Modify: `tests/test_compositor.py`

- [ ] **Step 1: Write failing tests**

Add these functions to `tests/test_compositor.py` (keep all existing tests, add imports at top):

```python
# Add to top of tests/test_compositor.py alongside existing imports:
import pytest
from ascii_combinator.bg_mode import BgMode, SoftBgConfig
from ascii_combinator.segmentation import SubjectMask


def _make_mask(rows: int, cols: int, default: bool) -> SubjectMask:
    return [[default for _ in range(cols)] for _ in range(rows)]


def test_compositor_keep_ignores_mask():
    """BgMode.KEEP must not modify result even when mask provided."""
    m: CharMap = _make_charmap(2, 2)
    m[0][0].append(CharCell(char=".", intensity=0.5, layer_id="brightness"))
    mask = _make_mask(2, 2, False)  # all background

    result = Compositor().composite([m], mask=mask, bg_mode=BgMode.KEEP)
    assert len(result[0][0]) == 1


def test_compositor_remove_clears_background():
    """Background cells must be empty in REMOVE mode."""
    m: CharMap = _make_charmap(2, 2)
    m[0][0].append(CharCell(char=".", intensity=0.5, layer_id="brightness"))
    m[0][1].append(CharCell(char="#", intensity=0.9, layer_id="brightness"))
    mask = _make_mask(2, 2, False)
    mask[0][0] = True  # [0][0] is subject

    result = Compositor().composite([m], mask=mask, bg_mode=BgMode.REMOVE)
    assert len(result[0][0]) == 1
    assert len(result[0][1]) == 0


def test_compositor_soft_reduces_intensity():
    """Background cells get a single soft.chars[0] cell with reduced intensity."""
    m: CharMap = _make_charmap(1, 2)
    m[0][0].append(CharCell(char="#", intensity=0.9, layer_id="brightness"))
    m[0][1].append(CharCell(char="#", intensity=0.9, layer_id="brightness"))
    mask = _make_mask(1, 2, False)
    mask[0][0] = True  # subject

    cfg = SoftBgConfig(opacity=0.3, chars=".,")
    result = Compositor().composite([m], mask=mask, bg_mode=BgMode.SOFT, soft_cfg=cfg)

    assert result[0][0][0].char == "#"
    assert result[0][0][0].intensity == pytest.approx(0.9)

    assert len(result[0][1]) == 1
    assert result[0][1][0].char == "."
    assert result[0][1][0].intensity == pytest.approx(0.3)


def test_compositor_soft_empty_cell_stays_empty():
    """Empty background cells remain empty in SOFT mode."""
    m: CharMap = _make_charmap(1, 1)
    mask = _make_mask(1, 1, False)
    cfg = SoftBgConfig()

    result = Compositor().composite([m], mask=mask, bg_mode=BgMode.SOFT, soft_cfg=cfg)
    assert result[0][0] == []


def test_compositor_none_mask_unchanged():
    """None mask with any mode must not modify result."""
    m: CharMap = _make_charmap(1, 1)
    m[0][0].append(CharCell(char="@", intensity=0.8, layer_id="brightness"))

    result = Compositor().composite([m], mask=None, bg_mode=BgMode.REMOVE)
    assert len(result[0][0]) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_compositor.py -v
```
Expected: 5 new tests FAIL with `TypeError: composite() got unexpected keyword argument 'mask'`

- [ ] **Step 3: Implement**

Replace the full content of `ascii_combinator/compositor.py`:

```python
from ascii_combinator.types import CharCell, CharMap
from ascii_combinator.bg_mode import BgMode, SoftBgConfig
from ascii_combinator.segmentation import SubjectMask


class Compositor:
    def composite(
        self,
        charmap_list: list[CharMap],
        mask: SubjectMask | None = None,
        bg_mode: BgMode = BgMode.KEEP,
        soft_cfg: SoftBgConfig | None = None,
    ) -> CharMap:
        if not charmap_list:
            raise ValueError("No CharMaps to composite")

        num_rows = len(charmap_list[0])
        num_cols = len(charmap_list[0][0]) if num_rows > 0 else 0

        result: CharMap = [[[] for _ in range(num_cols)] for _ in range(num_rows)]
        for charmap in charmap_list:
            for r in range(num_rows):
                for c in range(num_cols):
                    result[r][c].extend(charmap[r][c])

        if mask is None or bg_mode == BgMode.KEEP:
            return result

        soft = soft_cfg or SoftBgConfig()
        for r in range(num_rows):
            for c in range(num_cols):
                if mask[r][c]:
                    continue
                if bg_mode == BgMode.REMOVE:
                    result[r][c] = []
                elif bg_mode == BgMode.SOFT and result[r][c]:
                    result[r][c] = [
                        CharCell(char=soft.chars[0], intensity=soft.opacity, layer_id="background")
                    ]
        return result
```

- [ ] **Step 4: Run all compositor tests**

```bash
pytest tests/test_compositor.py -v
```
Expected: all 9 tests pass (4 existing + 5 new)

- [ ] **Step 5: Commit**

```bash
git add ascii_combinator/compositor.py tests/test_compositor.py
git commit -m "feat: add mask-aware composite with REMOVE and SOFT background modes"
```

---

### Task 4: CLI integration

**Files:**
- Modify: `ascii_combinator/cli.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Write failing tests**

Add to the bottom of `tests/test_cli.py`:

```python
def test_cli_bg_mode_keep_is_default(tmp_path):
    """--bg-mode keep must work without rembg installed."""
    input_img = tmp_path / "test.jpg"
    output_img = tmp_path / "out.png"
    _make_test_image(input_img)

    result = subprocess.run(
        [sys.executable, "-m", "ascii_combinator", str(input_img),
         "-o", str(output_img), "--width", "20", "--bg-mode", "keep"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert output_img.exists()


def test_cli_bg_mode_invalid(tmp_path):
    """Unknown --bg-mode value must cause non-zero exit."""
    input_img = tmp_path / "test.jpg"
    _make_test_image(input_img)

    result = subprocess.run(
        [sys.executable, "-m", "ascii_combinator", str(input_img),
         "--width", "20", "--bg-mode", "invalid"],
        capture_output=True, text=True,
    )
    assert result.returncode != 0


def test_cli_bg_chars_empty_soft_mode_error(tmp_path):
    """--bg-chars '' with --bg-mode soft must print an error and exit non-zero."""
    input_img = tmp_path / "test.jpg"
    _make_test_image(input_img)

    result = subprocess.run(
        [sys.executable, "-m", "ascii_combinator", str(input_img),
         "--width", "20", "--bg-mode", "soft", "--bg-chars", ""],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "bg-chars" in result.stderr.lower() or "bg-chars" in result.stdout.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_cli.py::test_cli_bg_mode_keep_is_default tests/test_cli.py::test_cli_bg_mode_invalid tests/test_cli.py::test_cli_bg_chars_empty_soft_mode_error -v
```
Expected: first test fails with `error: argument --bg-mode: invalid choice`, others fail similarly

- [ ] **Step 3: Implement**

Replace the full content of `ascii_combinator/cli.py`:

```python
import argparse
import sys
from pathlib import Path
from PIL import Image
from ascii_combinator.layers.brightness import BrightnessLayer
from ascii_combinator.layers.sobel_x import SobelXLayer
from ascii_combinator.layers.sobel_y import SobelYLayer
from ascii_combinator.layers.diagonal import DiagonalLayer
from ascii_combinator.compositor import Compositor
from ascii_combinator.renderer import Renderer
from ascii_combinator.profiles.monochrome import MonochromeProfile
from ascii_combinator.bg_mode import BgMode, SoftBgConfig

LAYER_REGISTRY = {
    "brightness": BrightnessLayer,
    "sobel_x": SobelXLayer,
    "sobel_y": SobelYLayer,
    "diagonal": DiagonalLayer,
}

PROFILE_REGISTRY = {
    "monochrome": MonochromeProfile,
}


def main():
    parser = argparse.ArgumentParser(description="ASCII Combinator — multilayer image to ASCII PNG")
    parser.add_argument("input", type=Path, help="Input image path")
    parser.add_argument("-o", "--output", type=Path, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--profile", default="monochrome", choices=list(PROFILE_REGISTRY))
    parser.add_argument("--layers", default=",".join(LAYER_REGISTRY.keys()))
    parser.add_argument("--jitter", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0.15)
    parser.add_argument("--font-size", type=int, default=12)
    parser.add_argument("--bg-mode", default="keep", choices=["keep", "remove", "soft"],
                        dest="bg_mode")
    parser.add_argument("--bg-opacity", type=float, default=0.25, dest="bg_opacity")
    parser.add_argument("--bg-chars", type=str, default=".,", dest="bg_chars")
    args = parser.parse_args()

    image = Image.open(args.input)
    font_size = args.font_size
    cell_w = max(int(font_size * 0.6), 1)
    cell_h = max(font_size, 1)

    num_cols = args.width if args.width is not None else max(image.width // cell_w, 10)
    num_rows = max(1, int(num_cols * cell_w / cell_h * image.height / image.width))

    layer_names = [n.strip() for n in args.layers.split(",") if n.strip()]
    unknown = [n for n in layer_names if n not in LAYER_REGISTRY]
    if unknown:
        parser.error(f"unknown layer(s): {', '.join(unknown)}. Choose from: {list(LAYER_REGISTRY)}")
    if not layer_names:
        parser.error("--layers cannot be empty")

    bg_mode = BgMode(args.bg_mode)

    if bg_mode == BgMode.SOFT and not args.bg_chars:
        parser.error("--bg-chars cannot be empty when --bg-mode is soft")

    mask = None
    if bg_mode != BgMode.KEEP:
        try:
            from ascii_combinator.segmentation import Segmenter
            mask = Segmenter().segment(image, num_rows, num_cols)
        except ImportError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)

    soft_cfg = SoftBgConfig(opacity=args.bg_opacity, chars=args.bg_chars) if bg_mode == BgMode.SOFT else None
    layers = [LAYER_REGISTRY[n](threshold=args.threshold) for n in layer_names]
    charmap_list = [layer.process(image, num_rows, num_cols) for layer in layers]
    charmap = Compositor().composite(charmap_list, mask=mask, bg_mode=bg_mode, soft_cfg=soft_cfg)

    profile = PROFILE_REGISTRY[args.profile]()
    result = Renderer().render(charmap, profile, font_size=font_size, jitter=args.jitter)

    output = args.output or args.input.with_name(args.input.stem + "_ascii.png")
    result.save(output)
    print(f"Saved: {output}")
```

- [ ] **Step 4: Run all tests**

```bash
pytest tests/ -v
```
Expected: all tests pass (21 existing + new ones)

- [ ] **Step 5: Commit**

```bash
git add ascii_combinator/cli.py tests/test_cli.py
git commit -m "feat: add --bg-mode CLI flag with remove/soft/keep options"
```

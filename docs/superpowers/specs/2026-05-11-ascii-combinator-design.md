# ASCII Combinator — Design Spec
**Date:** 2026-05-11

## Overview

A multilayer image-to-ASCII transformer that renders output as a PNG image. Each layer extracts a different visual feature from the source image and contributes its own character set. All layers are composited simultaneously — like multiple typewriter passes on the same page — producing a letterpress/linogravure aesthetic.

---

## Goals

- Convert any raster image (JPEG, PNG) into a PNG with typewriter-style ASCII art
- Multiple independent layers, each targeting a different visual feature
- All layers visible simultaneously in the output — no layer "wins", all contribute
- Monochrome by default; extensible to color profiles later
- CLI interface with sensible defaults and configurable overrides

## Non-Goals (v1)

- Terminal/ANSI output
- Animated GIF support
- Web UI
- Real-time preview

---

## Architecture

### Pipeline

```
image.jpg
  → [BrightnessLayer, SobelXLayer, SobelYLayer, DiagonalLayer]  (parallel)
  → Compositor           → CharMap (2D grid, each cell has list of CharCells)
  → Renderer + Profile   → PIL Image
  → output.png
```

### File Structure

```
ascii_combinator/
  __init__.py
  cli.py          — argparse entry point
  compositor.py   — merges CharMaps from all layers into one grid
  renderer.py     — draws PIL Image from composite CharMap + ColorProfile
  layers/
    __init__.py
    base.py        — abstract Layer class
    brightness.py  — grayscale density → . , ; : % # @
    sobel_x.py     — vertical edges (Sobel X kernel) → |
    sobel_y.py     — horizontal edges (Sobel Y kernel) → ─
    diagonal.py    — diagonal edges (combined Sobel) → / \
  profiles/
    __init__.py
    base.py        — abstract ColorProfile
    monochrome.py  — black chars on cream paper (#F5F0E8)
```

---

## Data Structures

```python
@dataclass
class CharCell:
    char: str        # character to draw
    intensity: float # 0.0–1.0, signal strength from the filter
    layer_id: str    # "brightness" | "sobel_x" | "sobel_y" | "diagonal"

# CharMap: 2D grid
# grid[row][col] → list[CharCell]  (one entry per contributing layer)
CharMap = list[list[list[CharCell]]]
```

---

## Layers

### BrightnessLayer
- Convert image to grayscale
- Divide into cells of `cell_w × cell_h` pixels (determined by font size and target width)
- Average brightness per cell → map to char from density string `" .,:;+*?%S#@"` (light to dark)
- Every cell gets a CharCell (even blank space is `" "`)

### SobelXLayer
- Apply Sobel X kernel (detects vertical edges)
- Threshold: only emit CharCell where `intensity > threshold` (default 0.15)
- Char: `|`

### SobelYLayer
- Apply Sobel Y kernel (detects horizontal edges)
- Threshold: same default 0.15
- Char: `─`

### DiagonalLayer
- Compute `arctan2(sobel_y, sobel_x)` per cell to get edge direction
- Emit `/` for angles near 45°, `\` for angles near 135°
- Only where combined edge magnitude > threshold

---

## Compositor

Iterates all layers, collects their CharMaps, merges into a single grid:

```python
grid[row][col] = [cell for layer in layers for cell in layer.charmap[row][col]]
```

No conflict resolution — all cells from all layers are kept. Renderer draws them all.

---

## Renderer

1. Create PIL Image with paper background color from profile
2. Load monospace font (bundled or system fallback)
3. For each `(row, col)` in grid:
   - For each `CharCell` in `grid[row][col]`:
     - Compute pixel position `(x, y)` from `(col * cell_w, row * cell_h)`
     - Apply jitter: `x += random.randint(-jitter, jitter)`, same for y (default jitter=1px)
     - Get color from `profile.color_for(cell)` — in monochrome: black with `alpha = int(cell.intensity * 255)`
     - Draw character
4. Save PNG

### Typewriter aesthetic details
- Slight per-character position jitter (±1px default, configurable)
- Characters drawn with opacity proportional to signal intensity
- Cream paper background (#F5F0E8 default)

---

## ColorProfile Interface

```python
class ColorProfile:
    background: tuple[int, int, int]         # RGB paper color
    def color_for(self, cell: CharCell) -> tuple[int, int, int, int]:
        # returns RGBA
        ...
```

### MonochromeProfile
- Background: `(245, 240, 232)` — warm cream
- All chars: `(20, 15, 10)` — near-black ink
- Alpha: `int(cell.intensity * 220) + 35` — always slightly visible, max near-opaque

---

## CLI

```
python -m ascii_combinator INPUT [-o OUTPUT] [--width N] [--profile PROFILE]
                                 [--layers LAYER,LAYER,...] [--jitter N]
                                 [--font PATH] [--threshold FLOAT]
```

| Flag | Default | Description |
|------|---------|-------------|
| `INPUT` | required | Path to input image |
| `-o OUTPUT` | `input_ascii.png` | Output PNG path |
| `--width N` | auto (image_width / cell_w, min 80) | Output width in characters |
| `--profile` | `monochrome` | Color profile name |
| `--layers` | all | Comma-separated layer list |
| `--jitter N` | `1` | Max pixel jitter for typewriter effect |
| `--threshold` | `0.15` | Edge detection threshold (0.0–1.0) |
| `--font PATH` | bundled Courier | Path to TTF monospace font |

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `Pillow` | Image I/O, drawing |
| `numpy` | Array ops, Sobel filter |
| `scipy` | `ndimage.sobel` for edge detection |

No other dependencies. Python 3.11+.

---

## Out of Scope for v1

- Color profiles (photo-color, duotone) — architecture supports them, not implemented
- Additional layers (Laplacian, corner detection)
- Batch processing
- Font auto-detection

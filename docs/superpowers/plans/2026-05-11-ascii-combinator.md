# ASCII Combinator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a multilayer image-to-ASCII transformer that renders a PNG with a typewriter/letterpress aesthetic by compositing brightness and edge-detection layers.

**Architecture:** Each layer (brightness, Sobel X/Y, diagonal) independently processes the input image and produces a `CharMap` — a 2D grid of `CharCell` objects. The `Compositor` merges all `CharMap`s into one grid (all layers present simultaneously). The `Renderer` draws every `CharCell` onto a PIL canvas with per-character jitter and ink opacity.

**Tech Stack:** Python 3.11+, Pillow, numpy, scipy, pytest

---

## File Map

| File | Responsibility |
|------|---------------|
| `requirements.txt` | Pillow, numpy, scipy, pytest |
| `ascii_combinator/__init__.py` | Package marker |
| `ascii_combinator/types.py` | `CharCell` dataclass, `CharMap` type alias |
| `ascii_combinator/layers/base.py` | Abstract `Layer` base class |
| `ascii_combinator/layers/brightness.py` | Grayscale density → `. , ; : % # @` |
| `ascii_combinator/layers/sobel_x.py` | Vertical edges (Sobel X) → `\|` |
| `ascii_combinator/layers/sobel_y.py` | Horizontal edges (Sobel Y) → `─` |
| `ascii_combinator/layers/diagonal.py` | Diagonal edges → `/ \` |
| `ascii_combinator/compositor.py` | Merges list of `CharMap` into one grid |
| `ascii_combinator/profiles/base.py` | Abstract `ColorProfile` |
| `ascii_combinator/profiles/monochrome.py` | Black ink on cream paper |
| `ascii_combinator/renderer.py` | PIL drawing with jitter + opacity |
| `ascii_combinator/cli.py` | argparse entry point |
| `tests/test_types.py` | CharCell tests |
| `tests/test_layers.py` | Layer tests with synthetic images |
| `tests/test_compositor.py` | Compositor merge tests |
| `tests/test_profiles.py` | ColorProfile tests |
| `tests/test_renderer.py` | Renderer smoke tests |
| `tests/test_cli.py` | End-to-end integration test |

---

### Task 1: Project Setup

**Files:**
- Create: `requirements.txt`
- Create: `pytest.ini`
- Create: `ascii_combinator/__init__.py`
- Create: `ascii_combinator/layers/__init__.py`
- Create: `ascii_combinator/profiles/__init__.py`
- Create: `tests/__init__.py`

- [ ] **Step 1: Create requirements.txt**

```
Pillow>=10.0.0
numpy>=1.24.0
scipy>=1.10.0
pytest>=7.4.0
```

- [ ] **Step 2: Create pytest.ini**

```ini
[pytest]
testpaths = tests
```

- [ ] **Step 3: Create all `__init__.py` files**

Each file is empty. Create:
- `ascii_combinator/__init__.py`
- `ascii_combinator/layers/__init__.py`
- `ascii_combinator/profiles/__init__.py`
- `tests/__init__.py`

- [ ] **Step 4: Install dependencies**

```bash
pip install -r requirements.txt
```

Expected: no errors, all packages installed.

- [ ] **Step 5: Commit**

```bash
git init
git add requirements.txt pytest.ini ascii_combinator/ tests/
git commit -m "chore: project setup — package structure and dependencies"
```

---

### Task 2: CharCell and CharMap types

**Files:**
- Create: `ascii_combinator/types.py`
- Create: `tests/test_types.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_types.py
from ascii_combinator.types import CharCell, CharMap

def test_charcell_fields():
    cell = CharCell(char="|", intensity=0.8, layer_id="sobel_x")
    assert cell.char == "|"
    assert cell.intensity == 0.8
    assert cell.layer_id == "sobel_x"

def test_charmap_type():
    grid: CharMap = [[[] for _ in range(3)] for _ in range(2)]
    grid[0][0].append(CharCell(char=".", intensity=0.1, layer_id="brightness"))
    assert len(grid[0][0]) == 1
    assert grid[0][1] == []
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_types.py -v
```

Expected: `ModuleNotFoundError: No module named 'ascii_combinator.types'`

- [ ] **Step 3: Write implementation**

```python
# ascii_combinator/types.py
from dataclasses import dataclass

@dataclass
class CharCell:
    char: str
    intensity: float  # 0.0–1.0
    layer_id: str

# grid[row][col] → list of CharCells from all contributing layers
CharMap = list[list[list[CharCell]]]
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_types.py -v
```

Expected: 2 tests PASSED.

- [ ] **Step 5: Commit**

```bash
git add ascii_combinator/types.py tests/test_types.py
git commit -m "feat: add CharCell dataclass and CharMap type"
```

---

### Task 3: Layer base class

**Files:**
- Create: `ascii_combinator/layers/base.py`

- [ ] **Step 1: Write implementation** (abstract class — tested through subclasses)

```python
# ascii_combinator/layers/base.py
from abc import ABC, abstractmethod
from PIL import Image
from ascii_combinator.types import CharMap

class Layer(ABC):
    id: str = "base"

    def __init__(self, threshold: float = 0.15):
        self.threshold = threshold

    @abstractmethod
    def process(self, image: Image.Image, num_rows: int, num_cols: int) -> CharMap:
        ...

def _to_cell_grid(arr, num_rows: int, num_cols: int):
    """Downsample 2D numpy array to (num_rows, num_cols) by block averaging."""
    import numpy as np
    h, w = arr.shape
    arr = arr[:h - h % num_rows, :w - w % num_cols]
    return arr.reshape(num_rows, h // num_rows, num_cols, w // num_cols).mean(axis=(1, 3))
```

- [ ] **Step 2: Commit**

```bash
git add ascii_combinator/layers/base.py
git commit -m "feat: add abstract Layer base class"
```

---

### Task 4: BrightnessLayer

**Files:**
- Create: `ascii_combinator/layers/brightness.py`
- Create: `tests/test_layers.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_layers.py
import numpy as np
from PIL import Image
from ascii_combinator.layers.brightness import BrightnessLayer

def _make_image(pixels: list[list[int]]) -> Image.Image:
    """Create grayscale image from 2D list of 0-255 values."""
    arr = np.array(pixels, dtype=np.uint8)
    return Image.fromarray(arr, mode="L").convert("RGB")

def test_brightness_all_black():
    """All-black image → dense chars everywhere."""
    img = _make_image([[0, 0], [0, 0]])
    layer = BrightnessLayer()
    result = layer.process(img, num_rows=2, num_cols=2)
    # Each cell should have one CharCell with a dense char
    for r in range(2):
        for c in range(2):
            assert len(result[r][c]) == 1
            assert result[r][c][0].layer_id == "brightness"
            assert result[r][c][0].char in "@#S%?*+"

def test_brightness_all_white():
    """All-white image → sparse/empty chars (spaces skipped)."""
    img = _make_image([[255, 255], [255, 255]])
    layer = BrightnessLayer()
    result = layer.process(img, num_rows=2, num_cols=2)
    for r in range(2):
        for c in range(2):
            # White maps to space which is skipped
            assert result[r][c] == []

def test_brightness_intensity_range():
    """Intensity values are in [0, 1]."""
    img = _make_image([[100, 200], [50, 150]])
    layer = BrightnessLayer()
    result = layer.process(img, num_rows=2, num_cols=2)
    for row in result:
        for cells in row:
            for cell in cells:
                assert 0.0 <= cell.intensity <= 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_layers.py -v
```

Expected: `ModuleNotFoundError: No module named 'ascii_combinator.layers.brightness'`

- [ ] **Step 3: Write implementation**

```python
# ascii_combinator/layers/brightness.py
import numpy as np
from PIL import Image
from ascii_combinator.types import CharCell, CharMap
from ascii_combinator.layers.base import Layer, _to_cell_grid

DENSITY = " .,:;+*?%S#@"  # index 0 = white/light, index 11 = black/dark

class BrightnessLayer(Layer):
    id = "brightness"

    def process(self, image: Image.Image, num_rows: int, num_cols: int) -> CharMap:
        gray = np.array(image.convert("L"), dtype=float) / 255.0
        grid = _to_cell_grid(gray, num_rows, num_cols)
        result: CharMap = [[[] for _ in range(num_cols)] for _ in range(num_rows)]
        for r in range(num_rows):
            for c in range(num_cols):
                intensity = float(grid[r, c])
                # Invert: bright pixel → low index (sparse), dark → high index (dense)
                idx = int((1.0 - intensity) * (len(DENSITY) - 1))
                char = DENSITY[idx]
                if char != " ":
                    result[r][c].append(CharCell(
                        char=char,
                        intensity=1.0 - intensity,  # high intensity = dark area
                        layer_id=self.id,
                    ))
        return result
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_layers.py -v
```

Expected: 3 tests PASSED.

- [ ] **Step 5: Commit**

```bash
git add ascii_combinator/layers/brightness.py tests/test_layers.py
git commit -m "feat: add BrightnessLayer — grayscale density mapping"
```

---

### Task 5: SobelXLayer and SobelYLayer

**Files:**
- Create: `ascii_combinator/layers/sobel_x.py`
- Create: `ascii_combinator/layers/sobel_y.py`
- Modify: `tests/test_layers.py`

- [ ] **Step 1: Add failing tests**

Append to `tests/test_layers.py`:

```python
from ascii_combinator.layers.sobel_x import SobelXLayer
from ascii_combinator.layers.sobel_y import SobelYLayer

def test_sobel_x_detects_vertical_edge():
    """Image with a sharp left/right boundary → SobelX emits | chars."""
    # Left half black, right half white → strong vertical edge in middle
    pixels = [[0, 0, 255, 255]] * 4
    img = _make_image(pixels)
    layer = SobelXLayer(threshold=0.05)
    result = layer.process(img, num_rows=2, num_cols=2)
    # Middle column boundary: cells at col=0 and col=1 straddle the edge
    all_chars = [cell.char for row in result for cells in row for cell in cells]
    assert "|" in all_chars

def test_sobel_y_detects_horizontal_edge():
    """Image with a sharp top/bottom boundary → SobelY emits ─ chars."""
    pixels = [[0, 0], [0, 0], [255, 255], [255, 255]]
    img = _make_image(pixels)
    layer = SobelYLayer(threshold=0.05)
    result = layer.process(img, num_rows=2, num_cols=2)
    all_chars = [cell.char for row in result for cells in row for cell in cells]
    assert "─" in all_chars

def test_sobel_uniform_image_no_edges():
    """Uniform image → no edges detected."""
    img = _make_image([[128, 128], [128, 128]])
    assert SobelXLayer().process(img, 2, 2) == [[[], []], [[], []]]
    assert SobelYLayer().process(img, 2, 2) == [[[], []], [[], []]]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_layers.py::test_sobel_x_detects_vertical_edge -v
```

Expected: `ModuleNotFoundError`

- [ ] **Step 3: Write SobelXLayer**

```python
# ascii_combinator/layers/sobel_x.py
import numpy as np
from scipy.ndimage import sobel
from PIL import Image
from ascii_combinator.types import CharCell, CharMap
from ascii_combinator.layers.base import Layer, _to_cell_grid

class SobelXLayer(Layer):
    id = "sobel_x"

    def process(self, image: Image.Image, num_rows: int, num_cols: int) -> CharMap:
        gray = np.array(image.convert("L"), dtype=float) / 255.0
        edges = np.abs(sobel(gray, axis=1))  # axis=1 → detects vertical edges
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

- [ ] **Step 4: Write SobelYLayer**

```python
# ascii_combinator/layers/sobel_y.py
import numpy as np
from scipy.ndimage import sobel
from PIL import Image
from ascii_combinator.types import CharCell, CharMap
from ascii_combinator.layers.base import Layer, _to_cell_grid

class SobelYLayer(Layer):
    id = "sobel_y"

    def process(self, image: Image.Image, num_rows: int, num_cols: int) -> CharMap:
        gray = np.array(image.convert("L"), dtype=float) / 255.0
        edges = np.abs(sobel(gray, axis=0))  # axis=0 → detects horizontal edges
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

- [ ] **Step 5: Run all layer tests**

```bash
pytest tests/test_layers.py -v
```

Expected: 6 tests PASSED.

- [ ] **Step 6: Commit**

```bash
git add ascii_combinator/layers/sobel_x.py ascii_combinator/layers/sobel_y.py tests/test_layers.py
git commit -m "feat: add SobelXLayer and SobelYLayer for edge detection"
```

---

### Task 6: DiagonalLayer

**Files:**
- Create: `ascii_combinator/layers/diagonal.py`
- Modify: `tests/test_layers.py`

- [ ] **Step 1: Add failing test**

Append to `tests/test_layers.py`:

```python
from ascii_combinator.layers.diagonal import DiagonalLayer

def test_diagonal_detects_slash():
    """Image with a / diagonal edge emits / chars."""
    # Create a 8x8 image: white bottom-left triangle, black top-right
    import numpy as np
    arr = np.zeros((8, 8), dtype=np.uint8)
    for i in range(8):
        for j in range(8):
            if j > i:
                arr[i, j] = 255
    img = Image.fromarray(arr, mode="L").convert("RGB")
    layer = DiagonalLayer(threshold=0.05)
    result = layer.process(img, num_rows=4, num_cols=4)
    all_chars = [cell.char for row in result for cells in row for cell in cells]
    assert "/" in all_chars or "\\" in all_chars  # diagonal detected
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_layers.py::test_diagonal_detects_slash -v
```

Expected: `ModuleNotFoundError`

- [ ] **Step 3: Write implementation**

```python
# ascii_combinator/layers/diagonal.py
import numpy as np
from scipy.ndimage import sobel
from PIL import Image
from ascii_combinator.types import CharCell, CharMap
from ascii_combinator.layers.base import Layer, _to_cell_grid

class DiagonalLayer(Layer):
    id = "diagonal"

    def process(self, image: Image.Image, num_rows: int, num_cols: int) -> CharMap:
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
        angle_grid = np.arctan2(gy_grid, gx_grid)  # -π to π

        result: CharMap = [[[] for _ in range(num_cols)] for _ in range(num_rows)]
        for r in range(num_rows):
            for c in range(num_cols):
                intensity = float(mag_grid[r, c])
                if intensity <= self.threshold:
                    continue
                # Normalize angle to [0, π)
                a = float(angle_grid[r, c]) % np.pi
                # π/8 to 3π/8 → "/" diagonal
                if np.pi / 8 < a < 3 * np.pi / 8:
                    char = "/"
                # 5π/8 to 7π/8 → "\" diagonal
                elif 5 * np.pi / 8 < a < 7 * np.pi / 8:
                    char = "\\"
                else:
                    continue
                result[r][c].append(CharCell(char=char, intensity=intensity, layer_id=self.id))
        return result
```

- [ ] **Step 4: Run all layer tests**

```bash
pytest tests/test_layers.py -v
```

Expected: 7 tests PASSED.

- [ ] **Step 5: Commit**

```bash
git add ascii_combinator/layers/diagonal.py tests/test_layers.py
git commit -m "feat: add DiagonalLayer — diagonal edge detection"
```

---

### Task 7: Compositor

**Files:**
- Create: `ascii_combinator/compositor.py`
- Create: `tests/test_compositor.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_compositor.py
from ascii_combinator.types import CharCell, CharMap
from ascii_combinator.compositor import Compositor

def _make_charmap(rows: int, cols: int) -> CharMap:
    return [[[] for _ in range(cols)] for _ in range(rows)]

def test_compositor_merges_two_maps():
    """Cells from both maps appear in the composite."""
    m1: CharMap = _make_charmap(2, 2)
    m1[0][0].append(CharCell(char=".", intensity=0.3, layer_id="brightness"))

    m2: CharMap = _make_charmap(2, 2)
    m2[0][0].append(CharCell(char="|", intensity=0.7, layer_id="sobel_x"))

    result = Compositor().composite([m1, m2])
    assert len(result[0][0]) == 2
    chars = {cell.char for cell in result[0][0]}
    assert chars == {".", "|"}

def test_compositor_empty_cells_stay_empty():
    """Cells with no content in any layer remain empty."""
    m1: CharMap = _make_charmap(2, 2)
    m2: CharMap = _make_charmap(2, 2)
    result = Compositor().composite([m1, m2])
    assert result[1][1] == []

def test_compositor_single_map():
    """Compositing a single map returns identical content."""
    m: CharMap = _make_charmap(1, 1)
    m[0][0].append(CharCell(char="#", intensity=0.9, layer_id="brightness"))
    result = Compositor().composite([m])
    assert result[0][0] == m[0][0]

def test_compositor_raises_on_empty_input():
    with __import__("pytest").raises(ValueError, match="No CharMaps"):
        Compositor().composite([])
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_compositor.py -v
```

Expected: `ModuleNotFoundError`

- [ ] **Step 3: Write implementation**

```python
# ascii_combinator/compositor.py
from ascii_combinator.types import CharMap

class Compositor:
    def composite(self, charmap_list: list[CharMap]) -> CharMap:
        if not charmap_list:
            raise ValueError("No CharMaps to composite")
        num_rows = len(charmap_list[0])
        num_cols = len(charmap_list[0][0]) if num_rows > 0 else 0
        result: CharMap = [[[] for _ in range(num_cols)] for _ in range(num_rows)]
        for charmap in charmap_list:
            for r in range(num_rows):
                for c in range(num_cols):
                    result[r][c].extend(charmap[r][c])
        return result
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_compositor.py -v
```

Expected: 4 tests PASSED.

- [ ] **Step 5: Commit**

```bash
git add ascii_combinator/compositor.py tests/test_compositor.py
git commit -m "feat: add Compositor — merges layer CharMaps without conflict"
```

---

### Task 8: ColorProfiles

**Files:**
- Create: `ascii_combinator/profiles/base.py`
- Create: `ascii_combinator/profiles/monochrome.py`
- Create: `tests/test_profiles.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_profiles.py
from ascii_combinator.types import CharCell
from ascii_combinator.profiles.monochrome import MonochromeProfile

def test_monochrome_background():
    profile = MonochromeProfile()
    assert len(profile.background) == 3
    # Should be a warm cream color
    r, g, b = profile.background
    assert r > 200 and g > 190 and b > 180

def test_monochrome_color_for_returns_rgba():
    profile = MonochromeProfile()
    cell = CharCell(char="|", intensity=0.8, layer_id="sobel_x")
    color = profile.color_for(cell)
    assert len(color) == 4  # RGBA
    r, g, b, a = color
    assert 0 <= r <= 255
    assert 0 <= a <= 255

def test_monochrome_higher_intensity_more_opaque():
    profile = MonochromeProfile()
    low = profile.color_for(CharCell(char=".", intensity=0.1, layer_id="brightness"))
    high = profile.color_for(CharCell(char="@", intensity=0.9, layer_id="brightness"))
    assert high[3] > low[3]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_profiles.py -v
```

Expected: `ModuleNotFoundError`

- [ ] **Step 3: Write base profile**

```python
# ascii_combinator/profiles/base.py
from abc import ABC, abstractmethod
from ascii_combinator.types import CharCell

class ColorProfile(ABC):
    background: tuple[int, int, int]

    @abstractmethod
    def color_for(self, cell: CharCell) -> tuple[int, int, int, int]:
        """Return RGBA color for the given CharCell."""
        ...
```

- [ ] **Step 4: Write MonochromeProfile**

```python
# ascii_combinator/profiles/monochrome.py
from ascii_combinator.types import CharCell
from ascii_combinator.profiles.base import ColorProfile

class MonochromeProfile(ColorProfile):
    background = (245, 240, 232)  # warm cream paper

    def color_for(self, cell: CharCell) -> tuple[int, int, int, int]:
        # Near-black ink, opacity driven by signal intensity
        # Min alpha=35 so even weak signals are slightly visible
        alpha = int(cell.intensity * 220) + 35
        alpha = min(alpha, 255)
        return (20, 15, 10, alpha)
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/test_profiles.py -v
```

Expected: 3 tests PASSED.

- [ ] **Step 6: Commit**

```bash
git add ascii_combinator/profiles/base.py ascii_combinator/profiles/monochrome.py tests/test_profiles.py
git commit -m "feat: add ColorProfile base and MonochromeProfile"
```

---

### Task 9: Renderer

**Files:**
- Create: `ascii_combinator/renderer.py`
- Create: `tests/test_renderer.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_renderer.py
from PIL import Image
from ascii_combinator.types import CharCell, CharMap
from ascii_combinator.profiles.monochrome import MonochromeProfile
from ascii_combinator.renderer import Renderer

def _simple_charmap() -> CharMap:
    grid: CharMap = [[[] for _ in range(4)] for _ in range(3)]
    grid[0][0].append(CharCell(char="@", intensity=0.9, layer_id="brightness"))
    grid[1][2].append(CharCell(char="|", intensity=0.7, layer_id="sobel_x"))
    grid[2][3].append(CharCell(char="/", intensity=0.5, layer_id="diagonal"))
    return grid

def test_renderer_returns_pil_image():
    result = Renderer().render(_simple_charmap(), MonochromeProfile(), font_size=10)
    assert isinstance(result, Image.Image)
    assert result.mode == "RGB"

def test_renderer_output_dimensions():
    """Output size = num_cols * glyph_w × num_rows * glyph_h."""
    charmap = _simple_charmap()  # 3 rows, 4 cols
    result = Renderer().render(charmap, MonochromeProfile(), font_size=10)
    # Width must be at least 4 pixels wide, height at least 3 pixels tall
    assert result.width >= 4
    assert result.height >= 3

def test_renderer_background_color():
    """Background should be the profile's background color."""
    profile = MonochromeProfile()
    charmap: CharMap = [[[] for _ in range(2)] for _ in range(2)]
    result = Renderer().render(charmap, profile, font_size=10)
    # Top-left corner pixel should be close to the background color
    px = result.getpixel((0, 0))
    assert abs(px[0] - profile.background[0]) < 30
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_renderer.py -v
```

Expected: `ModuleNotFoundError`

- [ ] **Step 3: Write implementation**

```python
# ascii_combinator/renderer.py
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

class Renderer:
    def render(
        self,
        charmap: CharMap,
        profile: ColorProfile,
        font_size: int = 12,
        jitter: int = 1,
        seed: int = 42,
    ) -> Image.Image:
        num_rows = len(charmap)
        num_cols = len(charmap[0]) if num_rows > 0 else 0
        font = _find_font(font_size)

        try:
            bbox = font.getbbox("M")
            glyph_w = max(bbox[2] - bbox[0], 1)
            glyph_h = max(bbox[3] - bbox[1], 1)
        except AttributeError:
            glyph_w, glyph_h = font_size, font_size

        out_w = num_cols * glyph_w
        out_h = num_rows * glyph_h
        img = Image.new("RGBA", (out_w, out_h), (*profile.background, 255))
        draw = ImageDraw.Draw(img)

        rng = random.Random(seed)
        for r in range(num_rows):
            for c in range(num_cols):
                for cell in charmap[r][c]:
                    x = c * glyph_w + rng.randint(-jitter, jitter)
                    y = r * glyph_h + rng.randint(-jitter, jitter)
                    color = profile.color_for(cell)
                    draw.text((x, y), cell.char, font=font, fill=color)

        return img.convert("RGB")
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_renderer.py -v
```

Expected: 3 tests PASSED.

- [ ] **Step 5: Commit**

```bash
git add ascii_combinator/renderer.py tests/test_renderer.py
git commit -m "feat: add Renderer — PIL drawing with jitter and opacity"
```

---

### Task 10: CLI and Integration Test

**Files:**
- Create: `ascii_combinator/cli.py`
- Create: `tests/test_cli.py`

- [ ] **Step 1: Write the failing integration test**

```python
# tests/test_cli.py
import subprocess
import sys
from pathlib import Path
import numpy as np
from PIL import Image

def _make_test_image(path: Path):
    """Save a small 32x32 gradient image for testing."""
    arr = np.zeros((32, 32, 3), dtype=np.uint8)
    for i in range(32):
        arr[i, :, :] = i * 8  # gradient
    arr[16, :, :] = 0  # horizontal edge
    arr[:, 16, :] = 0  # vertical edge
    Image.fromarray(arr).save(path)

def test_cli_produces_output(tmp_path):
    input_img = tmp_path / "test.jpg"
    output_img = tmp_path / "out.png"
    _make_test_image(input_img)

    result = subprocess.run(
        [sys.executable, "-m", "ascii_combinator", str(input_img), "-o", str(output_img), "--width", "20"],
        capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert output_img.exists()
    img = Image.open(output_img)
    assert img.width > 0 and img.height > 0

def test_cli_default_output_name(tmp_path):
    input_img = tmp_path / "photo.jpg"
    _make_test_image(input_img)
    expected_output = tmp_path / "photo_ascii.png"

    result = subprocess.run(
        [sys.executable, "-m", "ascii_combinator", str(input_img), "--width", "20"],
        capture_output=True, text=True,
        cwd=str(tmp_path)
    )
    assert result.returncode == 0, result.stderr
    assert expected_output.exists()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_cli.py -v
```

Expected: error — no `__main__.py` or `cli.py`

- [ ] **Step 3: Write CLI**

```python
# ascii_combinator/cli.py
import argparse
from pathlib import Path
from PIL import Image
from ascii_combinator.layers.brightness import BrightnessLayer
from ascii_combinator.layers.sobel_x import SobelXLayer
from ascii_combinator.layers.sobel_y import SobelYLayer
from ascii_combinator.layers.diagonal import DiagonalLayer
from ascii_combinator.compositor import Compositor
from ascii_combinator.renderer import Renderer
from ascii_combinator.profiles.monochrome import MonochromeProfile

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
    parser.add_argument("--width", type=int, default=None, help="Output width in characters")
    parser.add_argument("--profile", default="monochrome", choices=list(PROFILE_REGISTRY))
    parser.add_argument("--layers", default=",".join(LAYER_REGISTRY.keys()))
    parser.add_argument("--jitter", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0.15)
    parser.add_argument("--font-size", type=int, default=12)
    args = parser.parse_args()

    image = Image.open(args.input)
    font_size = args.font_size
    cell_w = max(int(font_size * 0.6), 1)
    cell_h = max(font_size, 1)

    num_cols = args.width if args.width is not None else max(image.width // cell_w, 10)
    num_rows = max(1, int(num_cols * cell_w / cell_h * image.height / image.width))

    layer_names = [n.strip() for n in args.layers.split(",") if n.strip() in LAYER_REGISTRY]
    layers = [LAYER_REGISTRY[n](threshold=args.threshold) for n in layer_names]

    charmap_list = [layer.process(image, num_rows, num_cols) for layer in layers]
    charmap = Compositor().composite(charmap_list)

    profile = PROFILE_REGISTRY[args.profile]()
    result = Renderer().render(charmap, profile, font_size=font_size, jitter=args.jitter)

    output = args.output or args.input.with_name(args.input.stem + "_ascii.png")
    result.save(output)
    print(f"Saved: {output}")
```

- [ ] **Step 4: Create `__main__.py` so `python -m ascii_combinator` works**

```python
# ascii_combinator/__main__.py
from ascii_combinator.cli import main
main()
```

- [ ] **Step 5: Run all tests**

```bash
pytest -v
```

Expected: all tests PASSED (types, layers, compositor, profiles, renderer, cli).

- [ ] **Step 6: Final commit**

```bash
git add ascii_combinator/cli.py ascii_combinator/__main__.py tests/test_cli.py
git commit -m "feat: add CLI entry point and integration test — MVP complete"
```

---

## Manual Smoke Test

After all tasks complete, verify the output visually:

```bash
# Download any test image or use your own
python -m ascii_combinator your_photo.jpg -o result.png --width 80
```

Open `result.png` — should look like a typewriter/letterpress rendering with visible layered characters.

import numpy as np
from PIL import Image
from ascii_combinator.layers.brightness import BrightnessLayer
from ascii_combinator.layers.sobel_x import SobelXLayer
from ascii_combinator.layers.sobel_y import SobelYLayer
from ascii_combinator.layers.diagonal import DiagonalLayer


def _make_image(pixels: list[list[int]]) -> Image.Image:
    """Create grayscale image from 2D list of 0-255 values."""
    arr = np.array(pixels, dtype=np.uint8)
    return Image.fromarray(arr, mode="L").convert("RGB")


def test_brightness_all_black():
    """All-black image → dense chars everywhere."""
    img = _make_image([[0, 0], [0, 0]])
    layer = BrightnessLayer()
    result = layer.process(img, num_rows=2, num_cols=2)
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


def test_sobel_x_detects_vertical_edge():
    """Image with a sharp left/right boundary → SobelX emits | chars."""
    pixels = [[0, 0, 255, 255]] * 4
    img = _make_image(pixels)
    layer = SobelXLayer(threshold=0.05)
    result = layer.process(img, num_rows=2, num_cols=2)
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


def test_diagonal_detects_slash():
    """Image with a / diagonal edge emits / chars."""
    arr = np.zeros((8, 8), dtype=np.uint8)
    for i in range(8):
        for j in range(8):
            if j > i:
                arr[i, j] = 255
    img = Image.fromarray(arr, mode="L").convert("RGB")
    layer = DiagonalLayer(threshold=0.05)
    result = layer.process(img, num_rows=4, num_cols=4)
    all_chars = [cell.char for row in result for cells in row for cell in cells]
    assert "\\" in all_chars  # \ edge from top-left to bottom-right triangle boundary

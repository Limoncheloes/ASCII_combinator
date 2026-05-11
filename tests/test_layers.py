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

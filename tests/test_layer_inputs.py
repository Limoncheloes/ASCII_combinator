import numpy as np
from PIL import Image

from ascii_combinator.layers.base import LayerInputs


def _img(w: int = 16, h: int = 16) -> Image.Image:
    arr = (np.arange(w * h) % 256).astype(np.uint8).reshape(h, w)
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

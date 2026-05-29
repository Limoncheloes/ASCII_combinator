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

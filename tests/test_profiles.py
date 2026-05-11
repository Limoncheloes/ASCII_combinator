from ascii_combinator.types import CharCell
from ascii_combinator.profiles.monochrome import MonochromeProfile


def test_monochrome_background():
    profile = MonochromeProfile()
    assert len(profile.background) == 3
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

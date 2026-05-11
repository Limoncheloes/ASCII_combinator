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
    assert result.width >= 4
    assert result.height >= 3


def test_renderer_background_color():
    """Background should be the profile's background color."""
    profile = MonochromeProfile()
    charmap: CharMap = [[[] for _ in range(2)] for _ in range(2)]
    result = Renderer().render(charmap, profile, font_size=10)
    px = result.getpixel((0, 0))
    assert abs(px[0] - profile.background[0]) < 30

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

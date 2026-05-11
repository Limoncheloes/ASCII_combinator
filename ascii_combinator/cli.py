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

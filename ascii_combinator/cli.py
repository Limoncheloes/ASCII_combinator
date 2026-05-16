import argparse
import os
import sys
from pathlib import Path

from PIL import Image

from ascii_combinator.bg_mode import BgMode, SoftBgConfig
from ascii_combinator.compositor import Compositor
from ascii_combinator.layers.brightness import BrightnessLayer
from ascii_combinator.layers.diagonal import DiagonalLayer
from ascii_combinator.layers.sobel_x import SobelXLayer
from ascii_combinator.layers.sobel_y import SobelYLayer
from ascii_combinator.profiles.monochrome import MonochromeProfile
from ascii_combinator.renderer import Renderer

LAYER_REGISTRY = {
    "brightness": BrightnessLayer,
    "sobel_x": SobelXLayer,
    "sobel_y": SobelYLayer,
    "diagonal": DiagonalLayer,
}

PROFILE_REGISTRY = {
    "monochrome": MonochromeProfile,
}


def _opacity_type(v: str) -> float:
    try:
        f = float(v)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid float value: '{v}'")
    if not 0.0 <= f <= 1.0:
        raise argparse.ArgumentTypeError("--bg-opacity must be between 0.0 and 1.0")
    return f


def _add_shared_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("input", type=Path, help="Input file path")
    parser.add_argument("-o", "--output", type=Path, default=None)
    parser.add_argument("--width", type=int, default=None, help="Output width in characters")
    parser.add_argument("--profile", default="monochrome", choices=list(PROFILE_REGISTRY))
    parser.add_argument("--layers", default=",".join(LAYER_REGISTRY.keys()))
    parser.add_argument("--jitter", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0.15)
    parser.add_argument("--font-size", type=int, default=12)
    parser.add_argument("--bg-mode", default="keep", choices=["keep", "remove", "soft"],
                        dest="bg_mode")
    parser.add_argument("--bg-opacity", type=_opacity_type, default=0.25, dest="bg_opacity")
    parser.add_argument("--bg-chars", type=str, default=".,", dest="bg_chars")


def _add_video_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--fps", type=float, default=None,
                        help="Frames per second to extract (and output fps)")
    parser.add_argument("--frame-step", type=int, default=None, dest="frame_step",
                        help="Extract every N-th frame (overrides --fps for extraction)")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1,
                        help="Parallel worker processes")
    parser.add_argument("--preview", action="store_true",
                        help="Render first frame only as a preview PNG")
    parser.add_argument("--gif", action="store_true",
                        help="Also produce a GIF alongside the MP4")
    parser.add_argument("--gif-fps", type=float, default=10.0, dest="gif_fps",
                        help="FPS for GIF output")


def _parse_shared(args: argparse.Namespace, parser: argparse.ArgumentParser):
    layer_names = [n.strip() for n in args.layers.split(",") if n.strip()]
    unknown = [n for n in layer_names if n not in LAYER_REGISTRY]
    if unknown:
        parser.error(f"unknown layer(s): {', '.join(unknown)}. Choose from: {list(LAYER_REGISTRY)}")
    if not layer_names:
        parser.error("--layers cannot be empty")

    bg_mode = BgMode(args.bg_mode)
    if bg_mode == BgMode.SOFT and not args.bg_chars:
        parser.error("--bg-chars cannot be empty when --bg-mode is soft")

    soft_cfg = SoftBgConfig(opacity=args.bg_opacity, chars=args.bg_chars) if bg_mode == BgMode.SOFT else None
    return layer_names, bg_mode, soft_cfg


def _run_image(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    layer_names, bg_mode, soft_cfg = _parse_shared(args, parser)

    image = Image.open(args.input)
    font_size = args.font_size
    cell_w = max(int(font_size * 0.6), 1)
    cell_h = max(font_size, 1)

    num_cols = args.width if args.width is not None else max(image.width // cell_w, 10)
    num_rows = max(1, int(num_cols * cell_w / cell_h * image.height / image.width))

    if bg_mode != BgMode.KEEP:
        try:
            from ascii_combinator.segmentation import Segmenter
            mask = Segmenter().segment(image, num_rows, num_cols)
        except ImportError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)
    else:
        mask = None

    layers = [LAYER_REGISTRY[n](threshold=args.threshold) for n in layer_names]
    charmap_list = [layer.process(image, num_rows, num_cols) for layer in layers]
    charmap = Compositor().composite(charmap_list, mask=mask, bg_mode=bg_mode, soft_cfg=soft_cfg)

    profile = PROFILE_REGISTRY[args.profile]()
    result = Renderer().render(charmap, profile, font_size=font_size, jitter=args.jitter)

    output = args.output or args.input.with_name(args.input.stem + "_ascii.png")
    result.save(output)
    print(f"Saved: {output}")


def _run_video(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    from ascii_combinator.video import VideoConfig, VideoProcessor

    if BgMode(args.bg_mode) == BgMode.REMOVE:
        parser.error("--bg-mode remove is not supported in video mode (too slow)")

    layer_names, bg_mode, soft_cfg = _parse_shared(args, parser)

    effective_fps = args.fps if args.fps is not None else 10.0

    if args.frame_step is not None and args.fps is not None:
        print(
            "Warning: both --frame-step and --fps specified; "
            "--frame-step takes priority for extraction, --fps used for output.",
            file=sys.stderr,
        )

    config = VideoConfig(
        width=args.width,
        profile_name=args.profile,
        layer_names=layer_names,
        jitter=args.jitter,
        threshold=args.threshold,
        font_size=args.font_size,
        bg_mode=bg_mode,
        soft_cfg=soft_cfg,
    )

    output = args.output or args.input.with_name(args.input.stem + "_ascii.mp4")

    VideoProcessor().process(
        video_path=args.input,
        output=output,
        config=config,
        fps=effective_fps,
        frame_step=args.frame_step,
        workers=args.workers,
        preview=args.preview,
        make_gif=args.gif,
        gif_fps=args.gif_fps,
    )


def main() -> None:
    # Backwards compatibility: if first arg is not a known subcommand, treat as `image`
    if len(sys.argv) > 1 and sys.argv[1] not in ("video", "image") and not sys.argv[1].startswith("-"):
        sys.argv.insert(1, "image")

    parser = argparse.ArgumentParser(description="ASCII Combinator — multilayer image/video to ASCII")
    subparsers = parser.add_subparsers(dest="command", required=True)

    image_parser = subparsers.add_parser("image", help="Convert a single image to ASCII PNG")
    _add_shared_args(image_parser)

    video_parser = subparsers.add_parser("video", help="Convert a video to ASCII animation")
    _add_shared_args(video_parser)
    # Override --bg-mode choices for video (remove not allowed)
    for action in video_parser._actions:
        if hasattr(action, "dest") and action.dest == "bg_mode":
            action.choices = ["keep", "soft"]
            break
    _add_video_args(video_parser)

    args = parser.parse_args()
    if args.command == "image":
        _run_image(args, parser)
    else:
        _run_video(args, video_parser)

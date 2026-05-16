import sys
import subprocess as _subprocess
import numpy as np
import pytest
from pathlib import Path
from PIL import Image
from unittest.mock import MagicMock, patch
from ascii_combinator.video import (
    FrameExtractor, FrameProcessor, VideoAssembler, VideoConfig, VideoProcessor,
)
from ascii_combinator.bg_mode import BgMode


def _make_png(path: Path, h: int = 32, w: int = 32) -> Path:
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[h // 2, :, :] = 120
    arr[:, w // 2, :] = 80
    Image.fromarray(arr).save(path)
    return path


def _default_config() -> VideoConfig:
    return VideoConfig(
        width=20,
        profile_name="monochrome",
        layer_names=["brightness"],
        jitter=0,
        threshold=0.15,
        font_size=8,
        bg_mode=BgMode.KEEP,
        soft_cfg=None,
    )


def test_frame_processor_produces_output(tmp_path):
    """FrameProcessor converts a real PNG into an ASCII PNG."""
    frame_in = _make_png(tmp_path / "frame_000001.png")
    frame_out = tmp_path / "frame_000001_ascii.png"

    result = FrameProcessor().process(frame_in, frame_out, _default_config())

    assert result == frame_out
    assert frame_out.exists()
    img = Image.open(frame_out)
    assert img.width > 0 and img.height > 0


def test_frame_processor_returns_frame_out(tmp_path):
    """process() return value is always frame_out."""
    frame_in = _make_png(tmp_path / "in.png")
    frame_out = tmp_path / "out.png"
    assert FrameProcessor().process(frame_in, frame_out, _default_config()) is frame_out


def test_video_config_is_picklable():
    """VideoConfig must survive pickle round-trip (needed for ProcessPoolExecutor)."""
    import pickle
    cfg = _default_config()
    assert pickle.loads(pickle.dumps(cfg)) == cfg


def _stub_ffmpeg_run(returncode: int = 0):
    return MagicMock(returncode=returncode, stderr="fake error")


def test_frame_extractor_fps_filter(tmp_path):
    """`fps` mode passes -vf fps=N to ffmpeg."""
    (tmp_path / "frame_000001.png").touch()  # fake extracted frame

    with patch("ascii_combinator.video.shutil.which", return_value="/usr/bin/ffmpeg"), \
         patch("ascii_combinator.video.subprocess.run", return_value=_stub_ffmpeg_run()) as mock_run:
        FrameExtractor().extract(Path("input.mp4"), tmp_path, fps=10.0, frame_step=None)

    args = mock_run.call_args[0][0]
    assert "-vf" in args
    vf_value = args[args.index("-vf") + 1]
    assert "fps=10.0" in vf_value


def test_frame_extractor_step_filter(tmp_path):
    """`frame_step` mode uses select filter, not fps."""
    (tmp_path / "frame_000001.png").touch()

    with patch("ascii_combinator.video.shutil.which", return_value="/usr/bin/ffmpeg"), \
         patch("ascii_combinator.video.subprocess.run", return_value=_stub_ffmpeg_run()) as mock_run:
        FrameExtractor().extract(Path("input.mp4"), tmp_path, fps=10.0, frame_step=3)

    args = mock_run.call_args[0][0]
    assert "-vf" in args
    vf_value = args[args.index("-vf") + 1]
    assert "select" in vf_value
    assert "3" in vf_value
    assert "fps=" not in vf_value


def test_frame_extractor_no_ffmpeg(tmp_path):
    """Missing ffmpeg binary raises FileNotFoundError."""
    with patch("ascii_combinator.video.shutil.which", return_value=None):
        with pytest.raises(FileNotFoundError, match="ffmpeg not found"):
            FrameExtractor().extract(Path("input.mp4"), tmp_path, fps=10.0, frame_step=None)


def test_frame_extractor_ffmpeg_failure(tmp_path):
    """ffmpeg non-zero exit raises RuntimeError with stderr."""
    with patch("ascii_combinator.video.shutil.which", return_value="/usr/bin/ffmpeg"), \
         patch("ascii_combinator.video.subprocess.run", return_value=_stub_ffmpeg_run(returncode=1)):
        with pytest.raises(RuntimeError, match="ffmpeg failed"):
            FrameExtractor().extract(Path("input.mp4"), tmp_path, fps=10.0, frame_step=None)


def test_frame_extractor_no_frames(tmp_path):
    """Empty output directory raises RuntimeError."""
    with patch("ascii_combinator.video.shutil.which", return_value="/usr/bin/ffmpeg"), \
         patch("ascii_combinator.video.subprocess.run", return_value=_stub_ffmpeg_run()):
        with pytest.raises(RuntimeError, match="No frames extracted"):
            FrameExtractor().extract(Path("input.mp4"), tmp_path, fps=10.0, frame_step=None)


def test_frame_extractor_returns_sorted_paths(tmp_path):
    """Returns sorted list of extracted frame paths."""
    for i in [3, 1, 2]:
        (tmp_path / f"frame_{i:06d}.png").touch()

    with patch("ascii_combinator.video.shutil.which", return_value="/usr/bin/ffmpeg"), \
         patch("ascii_combinator.video.subprocess.run", return_value=_stub_ffmpeg_run()):
        frames = FrameExtractor().extract(Path("input.mp4"), tmp_path, fps=5.0, frame_step=None)

    assert [f.name for f in frames] == [
        "frame_000001.png", "frame_000002.png", "frame_000003.png"
    ]


def test_assembler_mp4_calls_ffmpeg(tmp_path):
    """assemble_mp4 invokes ffmpeg with libx264 and yuv420p."""
    with patch("ascii_combinator.video.subprocess.run", return_value=_stub_ffmpeg_run()) as mock_run:
        VideoAssembler().assemble_mp4(tmp_path, tmp_path / "out.mp4", fps=10.0)

    args = mock_run.call_args[0][0]
    assert "ffmpeg" in args
    assert "-c:v" in args
    assert "libx264" in args[args.index("-c:v") + 1]
    assert "-pix_fmt" in args
    assert "yuv420p" in args[args.index("-pix_fmt") + 1]
    assert str(tmp_path / "out.mp4") in args


def test_assembler_mp4_raises_on_failure(tmp_path):
    """assemble_mp4 raises RuntimeError when ffmpeg fails."""
    with patch("ascii_combinator.video.subprocess.run", return_value=_stub_ffmpeg_run(returncode=1)):
        with pytest.raises(RuntimeError, match="ffmpeg failed assembling MP4"):
            VideoAssembler().assemble_mp4(tmp_path, tmp_path / "out.mp4", fps=10.0)


def test_assembler_gif_calls_ffmpeg_twice(tmp_path):
    """assemble_gif calls ffmpeg exactly twice (palettegen + paletteuse)."""
    with patch("ascii_combinator.video.subprocess.run", return_value=_stub_ffmpeg_run()) as mock_run:
        VideoAssembler().assemble_gif(tmp_path / "in.mp4", tmp_path / "out.gif", fps=10.0)

    assert mock_run.call_count == 2
    first_args = mock_run.call_args_list[0][0][0]
    second_args = mock_run.call_args_list[1][0][0]
    assert "palettegen" in " ".join(first_args)
    assert "paletteuse" in " ".join(second_args)
    assert str(tmp_path / "out.gif") in second_args


def test_assembler_gif_raises_on_palettegen_failure(tmp_path):
    """assemble_gif raises RuntimeError if palettegen step fails."""
    with patch("ascii_combinator.video.subprocess.run", return_value=_stub_ffmpeg_run(returncode=1)):
        with pytest.raises(RuntimeError, match="palettegen failed"):
            VideoAssembler().assemble_gif(tmp_path / "in.mp4", tmp_path / "out.gif", fps=10.0)


def test_video_processor_preview_mode(tmp_path):
    """Preview mode processes only the first frame and saves a PNG."""
    frames_in = [_make_png(tmp_path / f"frame_{i:06d}.png") for i in range(1, 4)]
    output = tmp_path / "out.mp4"

    processed = []

    def fake_process(frame_in, frame_out, config):
        processed.append(frame_in)
        Image.new("RGB", (10, 10)).save(frame_out)
        return frame_out

    with patch.object(FrameProcessor, "process", side_effect=fake_process), \
         patch.object(FrameExtractor, "extract", return_value=frames_in), \
         patch.object(VideoAssembler, "assemble_mp4"), \
         patch.object(VideoAssembler, "assemble_gif"):
        VideoProcessor().process(
            video_path=Path("in.mp4"),
            output=output,
            config=_default_config(),
            fps=10.0,
            frame_step=None,
            workers=1,
            preview=True,
            make_gif=False,
            gif_fps=10.0,
        )

    assert len(processed) == 1
    assert processed[0] == frames_in[0]
    preview_png = tmp_path / "out_preview.png"
    assert preview_png.exists()


def test_video_processor_calls_assembler(tmp_path):
    """Normal mode calls assemble_mp4 after processing all frames."""
    frames_in = [_make_png(tmp_path / f"frame_{i:06d}.png") for i in range(1, 3)]
    output = tmp_path / "out.mp4"

    def fake_process(frame_in, frame_out, config):
        Image.new("RGB", (10, 10)).save(frame_out)
        return frame_out

    with patch.object(FrameProcessor, "process", side_effect=fake_process), \
         patch.object(FrameExtractor, "extract", return_value=frames_in), \
         patch.object(VideoAssembler, "assemble_mp4") as mock_mp4, \
         patch.object(VideoAssembler, "assemble_gif") as mock_gif:
        VideoProcessor().process(
            video_path=Path("in.mp4"),
            output=output,
            config=_default_config(),
            fps=10.0,
            frame_step=None,
            workers=1,
            preview=False,
            make_gif=False,
            gif_fps=10.0,
        )

    mock_mp4.assert_called_once()
    mock_gif.assert_not_called()


def test_video_processor_gif_flag(tmp_path):
    """make_gif=True calls assemble_gif with gif_fps=5.0."""
    frames_in = [_make_png(tmp_path / f"frame_{i:06d}.png") for i in range(1, 3)]
    output = tmp_path / "out.mp4"

    def fake_process(frame_in, frame_out, config):
        Image.new("RGB", (10, 10)).save(frame_out)
        return frame_out

    with patch.object(FrameProcessor, "process", side_effect=fake_process), \
         patch.object(FrameExtractor, "extract", return_value=frames_in), \
         patch.object(VideoAssembler, "assemble_mp4"), \
         patch.object(VideoAssembler, "assemble_gif") as mock_gif:
        VideoProcessor().process(
            video_path=Path("in.mp4"),
            output=output,
            config=_default_config(),
            fps=10.0,
            frame_step=None,
            workers=1,
            preview=False,
            make_gif=True,
            gif_fps=5.0,
        )

    mock_gif.assert_called_once()
    # assemble_gif(mp4_path, gif_output, gif_fps) — third positional arg is gif_fps
    assert mock_gif.call_args[0][2] == 5.0


def _make_test_video(path: Path) -> Path:
    """Create a tiny 1-second test video using ffmpeg."""
    img_path = path.parent / "_tmp_frame.png"
    arr = np.zeros((32, 32, 3), dtype=np.uint8)
    arr[16, :, :] = 100
    Image.fromarray(arr).save(img_path)
    _subprocess.run(
        ["ffmpeg", "-y", "-loop", "1", "-i", str(img_path),
         "-t", "0.3", "-r", "5", "-c:v", "libx264", "-pix_fmt", "yuv420p",
         str(path)],
        capture_output=True, check=True,
    )
    img_path.unlink(missing_ok=True)
    return path


def test_video_cli_preview(tmp_path):
    """`video --preview` renders first frame as PNG and exits 0."""
    video = _make_test_video(tmp_path / "test.mp4")
    output = tmp_path / "out.mp4"

    result = _subprocess.run(
        [sys.executable, "-m", "ascii_combinator", "video", str(video),
         "-o", str(output), "--width", "20", "--preview"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    preview = tmp_path / "out_preview.png"
    assert preview.exists()


def test_video_cli_invalid_bg_mode(tmp_path):
    """`--bg-mode remove` is rejected in video mode."""
    video = _make_test_video(tmp_path / "test.mp4")

    result = _subprocess.run(
        [sys.executable, "-m", "ascii_combinator", "video", str(video),
         "--width", "20", "--bg-mode", "remove"],
        capture_output=True, text=True,
    )
    assert result.returncode != 0


def test_video_cli_fps_and_step_both_warn(tmp_path):
    """When both --fps and --frame-step are given, frame-step wins and a warning is printed to stderr."""
    video = _make_test_video(tmp_path / "test.mp4")
    output = tmp_path / "out.mp4"

    result = _subprocess.run(
        [sys.executable, "-m", "ascii_combinator", "video", str(video),
         "-o", str(output), "--width", "20", "--fps", "5", "--frame-step", "2", "--preview"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "frame-step" in result.stderr.lower()

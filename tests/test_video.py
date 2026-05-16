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

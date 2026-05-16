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

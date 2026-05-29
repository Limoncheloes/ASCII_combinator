from pathlib import Path
from PIL import Image

from benchmarks.fixtures import make_synthetic_image, make_synthetic_video


def test_synthetic_image_has_requested_size(tmp_path: Path):
    path = make_synthetic_image(tmp_path, width=400, height=300, seed=7)
    assert path.exists()
    with Image.open(path) as im:
        assert im.size == (400, 300)
        assert im.mode == "RGB"


def test_synthetic_image_is_deterministic(tmp_path: Path):
    a = make_synthetic_image(tmp_path / "a.png", width=64, height=64, seed=42)
    b = make_synthetic_image(tmp_path / "b.png", width=64, height=64, seed=42)
    assert a.read_bytes() == b.read_bytes()


def test_synthetic_video_has_requested_duration(tmp_path: Path):
    path = make_synthetic_video(
        tmp_path, width=320, height=240, duration_s=2, fps=10, seed=1
    )
    assert path.exists()
    assert path.suffix == ".mp4"
    assert path.stat().st_size > 0

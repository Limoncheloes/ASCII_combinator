import subprocess
import sys
from pathlib import Path
import numpy as np
from PIL import Image


def _make_test_image(path: Path):
    """Save a small 32x32 gradient image for testing."""
    arr = np.zeros((32, 32, 3), dtype=np.uint8)
    for i in range(32):
        arr[i, :, :] = i * 8  # gradient
    arr[16, :, :] = 0  # horizontal edge
    arr[:, 16, :] = 0  # vertical edge
    Image.fromarray(arr).save(path)


def test_cli_produces_output(tmp_path):
    input_img = tmp_path / "test.jpg"
    output_img = tmp_path / "out.png"
    _make_test_image(input_img)

    result = subprocess.run(
        [sys.executable, "-m", "ascii_combinator", str(input_img), "-o", str(output_img), "--width", "20"],
        capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert output_img.exists()
    img = Image.open(output_img)
    assert img.width > 0 and img.height > 0


def test_cli_default_output_name(tmp_path):
    input_img = tmp_path / "photo.jpg"
    _make_test_image(input_img)
    expected_output = tmp_path / "photo_ascii.png"

    result = subprocess.run(
        [sys.executable, "-m", "ascii_combinator", str(input_img), "--width", "20"],
        capture_output=True, text=True,
        cwd=str(tmp_path)
    )
    assert result.returncode == 0, result.stderr
    assert expected_output.exists()

import sys
import importlib
import numpy as np
import pytest
from unittest.mock import patch
from PIL import Image
from ascii_combinator.segmentation import Segmenter, SubjectMask


def _make_rgba(h: int, w: int, alpha: np.ndarray) -> Image.Image:
    arr = np.zeros((h, w, 4), dtype=np.uint8)
    arr[:, :, 3] = alpha
    return Image.fromarray(arr, mode="RGBA")


def test_segmenter_returns_correct_shape():
    """Mask must be num_rows × num_cols."""
    alpha = np.full((32, 32), 255, dtype=np.uint8)
    rgba_img = _make_rgba(32, 32, alpha)

    with patch("ascii_combinator.segmentation.rembg") as mock_rembg:
        mock_rembg.remove.return_value = rgba_img
        mask = Segmenter().segment(Image.new("RGB", (32, 32)), num_rows=4, num_cols=4)

    assert len(mask) == 4
    assert all(len(row) == 4 for row in mask)


def test_segmenter_subject_cells_true():
    """Cells with high alpha → True (subject)."""
    alpha = np.full((32, 32), 255, dtype=np.uint8)
    rgba_img = _make_rgba(32, 32, alpha)

    with patch("ascii_combinator.segmentation.rembg") as mock_rembg:
        mock_rembg.remove.return_value = rgba_img
        mask = Segmenter().segment(Image.new("RGB", (32, 32)), num_rows=4, num_cols=4)

    assert all(mask[r][c] is True for r in range(4) for c in range(4))


def test_segmenter_background_cells_false():
    """Cells with zero alpha → False (background)."""
    alpha = np.zeros((32, 32), dtype=np.uint8)
    rgba_img = _make_rgba(32, 32, alpha)

    with patch("ascii_combinator.segmentation.rembg") as mock_rembg:
        mock_rembg.remove.return_value = rgba_img
        mask = Segmenter().segment(Image.new("RGB", (32, 32)), num_rows=4, num_cols=4)

    assert all(mask[r][c] is False for r in range(4) for c in range(4))


def test_segmenter_mixed_mask():
    """Left half subject, right half background."""
    alpha = np.zeros((32, 32), dtype=np.uint8)
    alpha[:, :16] = 255
    rgba_img = _make_rgba(32, 32, alpha)

    with patch("ascii_combinator.segmentation.rembg") as mock_rembg:
        mock_rembg.remove.return_value = rgba_img
        mask = Segmenter().segment(Image.new("RGB", (32, 32)), num_rows=4, num_cols=4)

    for r in range(4):
        assert mask[r][0] is True
        assert mask[r][1] is True
        assert mask[r][2] is False
        assert mask[r][3] is False


def test_segmenter_rembg_missing():
    """Clear ImportError when rembg is not installed."""
    original = sys.modules.get("rembg")
    sys.modules["rembg"] = None  # type: ignore
    import ascii_combinator.segmentation as seg_mod
    importlib.reload(seg_mod)
    try:
        with pytest.raises(ImportError, match="pip install rembg"):
            seg_mod.Segmenter().segment(Image.new("RGB", (32, 32)), num_rows=4, num_cols=4)
    finally:
        if original is None:
            sys.modules.pop("rembg", None)
        else:
            sys.modules["rembg"] = original
        importlib.reload(seg_mod)

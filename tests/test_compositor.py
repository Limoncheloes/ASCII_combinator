from ascii_combinator.types import CharCell, CharMap
from ascii_combinator.compositor import Compositor


def _make_charmap(rows: int, cols: int) -> CharMap:
    return [[[] for _ in range(cols)] for _ in range(rows)]


def test_compositor_merges_two_maps():
    """Cells from both maps appear in the composite."""
    m1: CharMap = _make_charmap(2, 2)
    m1[0][0].append(CharCell(char=".", intensity=0.3, layer_id="brightness"))

    m2: CharMap = _make_charmap(2, 2)
    m2[0][0].append(CharCell(char="|", intensity=0.7, layer_id="sobel_x"))

    result = Compositor().composite([m1, m2])
    assert len(result[0][0]) == 2
    chars = {cell.char for cell in result[0][0]}
    assert chars == {".", "|"}


def test_compositor_empty_cells_stay_empty():
    """Cells with no content in any layer remain empty."""
    m1: CharMap = _make_charmap(2, 2)
    m2: CharMap = _make_charmap(2, 2)
    result = Compositor().composite([m1, m2])
    assert result[1][1] == []


def test_compositor_single_map():
    """Compositing a single map returns identical content."""
    m: CharMap = _make_charmap(1, 1)
    m[0][0].append(CharCell(char="#", intensity=0.9, layer_id="brightness"))
    result = Compositor().composite([m])
    assert result[0][0] == m[0][0]


def test_compositor_raises_on_empty_input():
    import pytest

    with pytest.raises(ValueError, match="No CharMaps"):
        Compositor().composite([])


import pytest
from ascii_combinator.bg_mode import BgMode, SoftBgConfig
from ascii_combinator.segmentation import SubjectMask


def _make_mask(rows: int, cols: int, default: bool) -> SubjectMask:
    return [[default for _ in range(cols)] for _ in range(rows)]


def test_compositor_keep_ignores_mask():
    """BgMode.KEEP must not modify result even when mask provided."""
    m: CharMap = _make_charmap(2, 2)
    m[0][0].append(CharCell(char=".", intensity=0.5, layer_id="brightness"))
    mask = _make_mask(2, 2, False)  # all background

    result = Compositor().composite([m], mask=mask, bg_mode=BgMode.KEEP)
    assert len(result[0][0]) == 1


def test_compositor_remove_clears_background():
    """Background cells must be empty in REMOVE mode."""
    m: CharMap = _make_charmap(2, 2)
    m[0][0].append(CharCell(char=".", intensity=0.5, layer_id="brightness"))
    m[0][1].append(CharCell(char="#", intensity=0.9, layer_id="brightness"))
    mask = _make_mask(2, 2, False)
    mask[0][0] = True  # [0][0] is subject

    result = Compositor().composite([m], mask=mask, bg_mode=BgMode.REMOVE)
    assert len(result[0][0]) == 1
    assert len(result[0][1]) == 0


def test_compositor_soft_reduces_intensity():
    """Background cells get a single soft.chars[0] cell with reduced intensity."""
    m: CharMap = _make_charmap(1, 2)
    m[0][0].append(CharCell(char="#", intensity=0.9, layer_id="brightness"))
    m[0][1].append(CharCell(char="#", intensity=0.9, layer_id="brightness"))
    mask = _make_mask(1, 2, False)
    mask[0][0] = True  # subject

    cfg = SoftBgConfig(opacity=0.3, chars=".,")
    result = Compositor().composite([m], mask=mask, bg_mode=BgMode.SOFT, soft_cfg=cfg)

    assert result[0][0][0].char == "#"
    assert result[0][0][0].intensity == pytest.approx(0.9)

    assert len(result[0][1]) == 1
    assert result[0][1][0].char == "."
    assert result[0][1][0].intensity == pytest.approx(0.3)


def test_compositor_soft_empty_cell_stays_empty():
    """Empty background cells remain empty in SOFT mode."""
    m: CharMap = _make_charmap(1, 1)
    mask = _make_mask(1, 1, False)
    cfg = SoftBgConfig()

    result = Compositor().composite([m], mask=mask, bg_mode=BgMode.SOFT, soft_cfg=cfg)
    assert result[0][0] == []


def test_compositor_none_mask_unchanged():
    """None mask with any mode must not modify result."""
    m: CharMap = _make_charmap(1, 1)
    m[0][0].append(CharCell(char="@", intensity=0.8, layer_id="brightness"))

    result = Compositor().composite([m], mask=None, bg_mode=BgMode.REMOVE)
    assert len(result[0][0]) == 1

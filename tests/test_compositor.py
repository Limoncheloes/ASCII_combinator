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

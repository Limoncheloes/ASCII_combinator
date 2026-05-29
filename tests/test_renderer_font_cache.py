from ascii_combinator.renderer import _find_font


def test_find_font_returns_same_object_for_same_size():
    a = _find_font(12)
    b = _find_font(12)
    assert a is b, "LRU cache should return identical font object for same size"


def test_find_font_returns_different_object_for_different_size():
    a = _find_font(12)
    b = _find_font(18)
    assert a is not b

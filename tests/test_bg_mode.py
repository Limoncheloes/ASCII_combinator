from ascii_combinator.bg_mode import BgMode, SoftBgConfig


def test_bgmode_values():
    assert BgMode.KEEP.value == "keep"
    assert BgMode.REMOVE.value == "remove"
    assert BgMode.SOFT.value == "soft"


def test_softbgconfig_defaults():
    cfg = SoftBgConfig()
    assert cfg.opacity == 0.25
    assert cfg.chars == ".,"


def test_softbgconfig_custom():
    cfg = SoftBgConfig(opacity=0.1, chars=".")
    assert cfg.opacity == 0.1
    assert cfg.chars == "."

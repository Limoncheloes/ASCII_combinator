from ascii_combinator.types import CharCell, CharMap


def test_charcell_fields():
    cell = CharCell(char="|", intensity=0.8, layer_id="sobel_x")
    assert cell.char == "|"
    assert cell.intensity == 0.8
    assert cell.layer_id == "sobel_x"


def test_charmap_type():
    grid: CharMap = [[[] for _ in range(3)] for _ in range(2)]
    grid[0][0].append(CharCell(char=".", intensity=0.1, layer_id="brightness"))
    assert len(grid[0][0]) == 1
    assert grid[0][1] == []

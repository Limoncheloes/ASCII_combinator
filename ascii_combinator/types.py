from dataclasses import dataclass


@dataclass
class CharCell:
    char: str
    intensity: float  # 0.0–1.0
    layer_id: str


# grid[row][col] → list of CharCells from all contributing layers
CharMap = list[list[list[CharCell]]]

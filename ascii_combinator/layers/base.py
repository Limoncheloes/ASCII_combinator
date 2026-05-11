from abc import ABC, abstractmethod
from PIL import Image
from ascii_combinator.types import CharMap


class Layer(ABC):
    id: str = "base"

    def __init__(self, threshold: float = 0.15):
        self.threshold = threshold

    @abstractmethod
    def process(self, image: Image.Image, num_rows: int, num_cols: int) -> CharMap:
        ...


def _to_cell_grid(arr, num_rows: int, num_cols: int):
    """Downsample 2D numpy array to (num_rows, num_cols) by block averaging."""
    import numpy as np
    h, w = arr.shape
    arr = arr[:h - h % num_rows, :w - w % num_cols]
    return arr.reshape(num_rows, h // num_rows, num_cols, w // num_cols).mean(axis=(1, 3))

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from PIL import Image
from scipy.ndimage import sobel

from ascii_combinator.types import CharMap


@dataclass
class LayerInputs:
    """Precomputed numpy arrays shared between layers for a single image.

    Constructing one `LayerInputs` per image (instead of recomputing inside
    every layer) eliminates duplicate grayscale conversions and Sobel passes
    when multiple layers run together.
    """
    image: Image.Image
    _gray: Optional[np.ndarray] = field(default=None, repr=False)
    _sobel_x: Optional[np.ndarray] = field(default=None, repr=False)
    _sobel_y: Optional[np.ndarray] = field(default=None, repr=False)

    @classmethod
    def from_image(cls, image: Image.Image) -> "LayerInputs":
        return cls(image=image)

    @property
    def gray(self) -> np.ndarray:
        if self._gray is None:
            self._gray = np.array(self.image.convert("L"), dtype=float) / 255.0
        return self._gray

    @property
    def sobel_x(self) -> np.ndarray:
        if self._sobel_x is None:
            self._sobel_x = sobel(self.gray, axis=1)
        return self._sobel_x

    @property
    def sobel_y(self) -> np.ndarray:
        if self._sobel_y is None:
            self._sobel_y = sobel(self.gray, axis=0)
        return self._sobel_y


class Layer(ABC):
    id: str = "base"

    def __init__(self, threshold: float = 0.15):
        self.threshold = threshold

    @abstractmethod
    def process(
        self,
        image: Image.Image,
        num_rows: int,
        num_cols: int,
        inputs: LayerInputs | None = None,
    ) -> CharMap:
        ...


def _to_cell_grid(arr, num_rows: int, num_cols: int):
    """Downsample 2D numpy array to (num_rows, num_cols) by block averaging."""
    h, w = arr.shape
    if h < num_rows or w < num_cols:
        raise ValueError(f"Image ({h}x{w}) is smaller than requested grid ({num_rows}x{num_cols})")
    arr = arr[:h - h % num_rows, :w - w % num_cols]
    return arr.reshape(num_rows, h // num_rows, num_cols, w // num_cols).mean(axis=(1, 3))

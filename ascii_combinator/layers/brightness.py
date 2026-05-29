import numpy as np
from PIL import Image
from ascii_combinator.types import CharCell, CharMap
from ascii_combinator.layers.base import Layer, LayerInputs, _to_cell_grid

DENSITY = " .,:;+*?%S#@"  # index 0 = white/light, index 11 = black/dark


class BrightnessLayer(Layer):
    id = "brightness"

    def process(
        self,
        image: Image.Image,
        num_rows: int,
        num_cols: int,
        inputs: LayerInputs | None = None,
    ) -> CharMap:
        gray = inputs.gray if inputs is not None else (
            np.array(image.convert("L"), dtype=float) / 255.0
        )
        grid = _to_cell_grid(gray, num_rows, num_cols)
        result: CharMap = [[[] for _ in range(num_cols)] for _ in range(num_rows)]
        for r in range(num_rows):
            for c in range(num_cols):
                intensity = float(grid[r, c])
                idx = int((1.0 - intensity) * (len(DENSITY) - 1))
                char = DENSITY[idx]
                if char != " ":
                    result[r][c].append(CharCell(
                        char=char,
                        intensity=1.0 - intensity,
                        layer_id=self.id,
                    ))
        return result

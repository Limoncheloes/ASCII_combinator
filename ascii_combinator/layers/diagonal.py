import numpy as np
from scipy.ndimage import sobel
from PIL import Image
from ascii_combinator.types import CharCell, CharMap
from ascii_combinator.layers.base import Layer, LayerInputs, _to_cell_grid


class DiagonalLayer(Layer):
    id = "diagonal"

    def process(
        self,
        image: Image.Image,
        num_rows: int,
        num_cols: int,
        inputs: LayerInputs | None = None,
    ) -> CharMap:
        if inputs is not None:
            gx = inputs.sobel_x
            gy = inputs.sobel_y
        else:
            gray = np.array(image.convert("L"), dtype=float) / 255.0
            gx = sobel(gray, axis=1)
            gy = sobel(gray, axis=0)
        magnitude = np.hypot(gx, gy)
        norm = magnitude.max()
        if norm > 0:
            magnitude = magnitude / norm

        gx_grid = _to_cell_grid(gx, num_rows, num_cols)
        gy_grid = _to_cell_grid(gy, num_rows, num_cols)
        mag_grid = _to_cell_grid(magnitude, num_rows, num_cols)
        angle_grid = np.arctan2(gy_grid, gx_grid)

        result: CharMap = [[[] for _ in range(num_cols)] for _ in range(num_rows)]
        for r in range(num_rows):
            for c in range(num_cols):
                intensity = float(mag_grid[r, c])
                if intensity <= self.threshold:
                    continue
                a = float(angle_grid[r, c]) % np.pi
                if np.pi / 8 < a < 3 * np.pi / 8:
                    char = "/"
                elif 5 * np.pi / 8 < a < 7 * np.pi / 8:
                    char = "\\"
                else:
                    continue
                result[r][c].append(CharCell(char=char, intensity=intensity, layer_id=self.id))
        return result

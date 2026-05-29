import numpy as np
from scipy.ndimage import sobel
from PIL import Image
from ascii_combinator.types import CharCell, CharMap
from ascii_combinator.layers.base import Layer, LayerInputs, _to_cell_grid


class SobelXLayer(Layer):
    id = "sobel_x"

    def process(
        self,
        image: Image.Image,
        num_rows: int,
        num_cols: int,
        inputs: LayerInputs | None = None,
    ) -> CharMap:
        if inputs is not None:
            sx = inputs.sobel_x
        else:
            gray = np.array(image.convert("L"), dtype=float) / 255.0
            sx = sobel(gray, axis=1)
        edges = np.abs(sx)
        norm = edges.max()
        if norm > 0:
            edges = edges / norm
        grid = _to_cell_grid(edges, num_rows, num_cols)
        result: CharMap = [[[] for _ in range(num_cols)] for _ in range(num_rows)]
        for r in range(num_rows):
            for c in range(num_cols):
                intensity = float(grid[r, c])
                if intensity > self.threshold:
                    result[r][c].append(CharCell(char="|", intensity=intensity, layer_id=self.id))
        return result

import numpy as np
from PIL import Image

SubjectMask = list[list[bool]]

try:
    import rembg
except ImportError:
    rembg = None  # type: ignore


class Segmenter:
    def segment(self, image: Image.Image, num_rows: int, num_cols: int) -> SubjectMask:
        if rembg is None:
            raise ImportError(
                "rembg is required for subject segmentation. "
                "Install it with: pip install rembg"
            )
        if num_rows > image.height or num_cols > image.width:
            raise ValueError(
                f"Grid ({num_rows}×{num_cols}) exceeds image size "
                f"({image.height}×{image.width})."
            )
        rgba = rembg.remove(image)
        alpha = np.array(rgba)[:, :, 3].astype(float) / 255.0

        h, w = alpha.shape
        alpha = alpha[: h - h % num_rows, : w - w % num_cols]
        block_h = h // num_rows
        block_w = w // num_cols
        averaged = alpha.reshape(num_rows, block_h, num_cols, block_w).mean(axis=(1, 3))

        return [[bool(averaged[r][c] >= 0.5) for c in range(num_cols)] for r in range(num_rows)]

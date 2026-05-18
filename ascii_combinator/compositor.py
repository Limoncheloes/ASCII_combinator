from ascii_combinator.types import CharCell, CharMap
from ascii_combinator.bg_mode import BgMode, SoftBgConfig
from ascii_combinator.segmentation import SubjectMask


class Compositor:
    def composite(
        self,
        charmap_list: list[CharMap],
        mask: SubjectMask | None = None,
        bg_mode: BgMode = BgMode.KEEP,
        soft_cfg: SoftBgConfig | None = None,
    ) -> CharMap:
        if not charmap_list:
            raise ValueError("No CharMaps to composite")

        num_rows = len(charmap_list[0])
        num_cols = len(charmap_list[0][0]) if num_rows > 0 else 0

        result: CharMap = [[[] for _ in range(num_cols)] for _ in range(num_rows)]
        for charmap in charmap_list:
            for r in range(num_rows):
                for c in range(num_cols):
                    result[r][c].extend(charmap[r][c])

        if mask is None or bg_mode == BgMode.KEEP:
            return result

        if mask is not None and bg_mode != BgMode.KEEP:
            if len(mask) != num_rows or (num_rows > 0 and len(mask[0]) != num_cols):
                raise ValueError(
                    f"Mask size {len(mask)}×{len(mask[0]) if mask else 0} "
                    f"does not match CharMap {num_rows}×{num_cols}"
                )

        soft = None
        for r in range(num_rows):
            for c in range(num_cols):
                if mask[r][c]:
                    continue
                if bg_mode == BgMode.REMOVE:
                    result[r][c] = []
                elif bg_mode == BgMode.SOFT and result[r][c]:
                    if soft is None:
                        soft = soft_cfg or SoftBgConfig()
                    result[r][c] = [
                        CharCell(char=soft.chars[0], intensity=soft.opacity, layer_id="background")
                    ]
        return result

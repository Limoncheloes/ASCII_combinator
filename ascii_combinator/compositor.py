from ascii_combinator.types import CharMap


class Compositor:
    def composite(self, charmap_list: list[CharMap]) -> CharMap:
        if not charmap_list:
            raise ValueError("No CharMaps to composite")

        num_rows = len(charmap_list[0])
        num_cols = len(charmap_list[0][0]) if num_rows > 0 else 0

        result: CharMap = [[[] for _ in range(num_cols)] for _ in range(num_rows)]

        for charmap in charmap_list:
            for r in range(num_rows):
                for c in range(num_cols):
                    result[r][c].extend(charmap[r][c])

        return result

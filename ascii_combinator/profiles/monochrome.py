from ascii_combinator.types import CharCell
from ascii_combinator.profiles.base import ColorProfile


class MonochromeProfile(ColorProfile):
    background = (245, 240, 232)  # warm cream paper

    def color_for(self, cell: CharCell) -> tuple[int, int, int, int]:
        # Near-black ink, opacity driven by signal intensity
        # Min alpha=35 so even weak signals are slightly visible
        alpha = int(cell.intensity * 220) + 35
        alpha = min(alpha, 255)
        return (20, 15, 10, alpha)

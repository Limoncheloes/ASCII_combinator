from abc import ABC, abstractmethod
from ascii_combinator.types import CharCell


class ColorProfile(ABC):
    background: tuple[int, int, int]

    @abstractmethod
    def color_for(self, cell: CharCell) -> tuple[int, int, int, int]:
        """Return RGBA color for the given CharCell."""
        ...

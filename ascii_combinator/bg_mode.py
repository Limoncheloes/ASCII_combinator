from dataclasses import dataclass
from enum import Enum


class BgMode(Enum):
    KEEP = "keep"
    REMOVE = "remove"
    SOFT = "soft"


@dataclass
class SoftBgConfig:
    opacity: float = 0.25
    chars: str = ".,"

    def __post_init__(self) -> None:
        if not self.chars:
            raise ValueError("SoftBgConfig.chars must not be empty")

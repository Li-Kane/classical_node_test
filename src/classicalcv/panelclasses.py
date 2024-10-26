from dataclasses import dataclass
from enum import Enum, auto
from typing import Tuple


class PanelColor(Enum):
    RED = auto()
    BLUE = auto()


@dataclass
class LightBar:
    upper: Tuple[int, int]
    lower: Tuple[int, int]
    rect: Tuple[Tuple[int, int], Tuple[int, int], float]


@dataclass
class LightBarPair:
    left: LightBar
    right: LightBar


class PanelSize(Enum):
    LARGE = auto()
    SMALL = auto()


class RobotType(Enum):
    HERO = auto()
    SENTRY = auto()
    INFANTRY = auto()
    BASE = auto()


@dataclass
class ArmorPanel:
    color: PanelColor
    size: PanelSize
    lightbars: LightBarPair

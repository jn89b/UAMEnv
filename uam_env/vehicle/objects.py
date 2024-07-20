from abc import ABC
from typing import TYPE_CHECKING, Optional, Sequence, Tuple

import numpy as np

# from highway_env import utils

# if TYPE_CHECKING:
#     from highway_env.road.lane import AbstractLane
#     from highway_env.road.road import Road

LaneIndex = Tuple[str, str, int]

class CorridorObject(ABC):
    """
    Common base class for all corridor objects, something in the way
    """
    def __init__(self) -> None:
        super().__init__()

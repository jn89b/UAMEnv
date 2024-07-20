from abc import ABC
from typing import TYPE_CHECKING, Optional, Sequence, Tuple

import numpy as np
from uam_env.utils import Vector

if TYPE_CHECKING:
    from uam_env.corridor.corridor import Corridor 

# from highway_env import utils

# if TYPE_CHECKING:
#     from highway_env.road.lane import AbstractLane
#     from highway_env.road.road import Road

LaneIndex = Tuple[str, str, int]

class CorridorObject(ABC):
    """
    Common base/interface class for all corridor objects 
    """
    def __init__(
        self,
        corridor: "Corridor", 
        position:Vector, 
        roll_dg:float=0,
        pitch_dg:float=0, 
        yaw_dg:float=0, 
        speed:float=15) -> None:
        super().__init__()
        self.corridor = corridor
        self.position = position
        self.roll_dg = roll_dg
        self.pitch_dg = pitch_dg
        self.yaw_dg = yaw_dg
        self.speed = speed
    
        self.attitudes_in_rad()
    
    def attitudes_in_rad(self) -> None:
        self.roll_rad = np.deg2rad(self.roll_dg)
        self.pitch_rad = np.deg2rad(self.pitch_dg)
        self.yaw_rad = np.deg2rad(self.yaw_dg)
        
    def get_position(self) -> Vector:
        return self.position
    
    

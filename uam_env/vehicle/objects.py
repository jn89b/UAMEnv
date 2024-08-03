from abc import ABC
from typing import TYPE_CHECKING, Optional, Sequence, Tuple

import numpy as np
from uam_env.utils import Vector
# from uam_env.corridor.lane import StraightLane
if TYPE_CHECKING:
    from uam_env.corridor.corridor import Corridor
    from uam_env.corridor.corridor import StraightLane

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
    
    @property
    def direction(self) -> np.ndarray:
        return np.array([np.cos(self.heading), np.sin(self.heading), 0])

    @property
    def velocity(self) -> np.ndarray:
        return self.speed * self.direction
    
    def attitudes_in_rad(self) -> Tuple[float, float, float]:
        self.roll_rad = np.deg2rad(self.roll_dg)
        self.pitch_rad = np.deg2rad(self.pitch_dg)
        self.yaw_rad = np.deg2rad(self.yaw_dg)
        return self.roll_rad, self.pitch_rad, self.yaw_rad
        
    def get_position(self) -> Vector:
        return self.position
    
    def lane_distance_to(
        self, other:"CorridorObject", 
        lane:"StraightLane") -> float:
        """
        Compute the signed distance to another object along the lane
        
        :param other: the other object
        :param lane: the lane
        :return: the distance to the other object [m]
        """
        if not other:
            return np.nan
        if not lane:
            lane = self.lane 
            
        distance = (lane.local_coordinates(other.position)[0] -
                    lane.local_coordinates(self.position)[0])
        return distance
    
    def __str__(self):
        return f"{self.__class__.__name__} #{id(self) % 1000}: at {self.position}"

    def __repr__(self):
        return self.__str__()

    
    

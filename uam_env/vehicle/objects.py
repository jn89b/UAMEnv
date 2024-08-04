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
        self.corridor = corridor
        self.position = position
        self.roll_dg = roll_dg
        self.pitch_dg = pitch_dg
        self.yaw_dg = yaw_dg
        self.speed = speed
        self.attitudes_in_rad()
        
        # Enable collision with other collidables
        self.collidable = True

        # Collisions have physical effects
        self.solid = True

        # If False, this object will not check its own collisions, but it can still collides with other objects that do
        # check their collisions.
        self.check_collisions = True

        self.diagonal = np.sqrt(self.LENGTH**2 + self.WIDTH**2)
        self.crashed = False
        self.hit = False
        self.impact = np.zeros(self.position.shape)
    
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
    
    def handle_collisions(self, other: "CorridorObject", 
                          dt: float = 0) -> None:
        """
        Check for collision with another vehicle.

        :param other: the other vehicle or object
        :param dt: timestep to check for future collisions (at constant velocity)
        """
        if other is self or not (self.check_collisions or other.check_collisions):
            return
        if not (self.collidable and other.collidable):
            return
        intersecting, will_intersect, transition = self._is_colliding(other, dt)
        if will_intersect:
            if self.solid and other.solid:
                if isinstance(other, Obstacle):
                    self.impact = transition
                elif isinstance(self, Obstacle):
                    other.impact = transition
                else:
                    self.impact = transition / 2
                    other.impact = -transition / 2
        if intersecting:
            if self.solid and other.solid:
                self.crashed = True
                other.crashed = True
            if not self.solid:
                self.hit = True
            if not other.solid:
                other.hit = True
    
    def _is_colliding(self, other, dt):
        # Fast spherical pre-check
        distance = np.linalg.norm(other.position - self.position)
        if distance > (self.diagonal + other.diagonal) / 2 + self.speed * dt:
            return False, False, np.zeros(2)
        
        else:
            intersection = True
            will_intersect = True
            transition = self.position 
            return intersection, will_intersect, transition
        
        #TODO: Implement the accurate check
        projection_of_other = other.position + other.velocity * dt
        projection_of_self = self.position + self.velocity * dt
        
        vector_self = projection_of_self - self.position
        vector_other = projection_of_other - other.position
        
        #                

        # if (
        #     np.linalg.norm(other.position - self.position)
        #     > (self.diagonal + other.diagonal) / 2 + self.speed * dt
        # ):
        #     return (
        #         False,
        #         False,
        #         np.zeros(
        #             2,
        #         ),
        #     )
        # # Accurate rectangular check
        # return utils.are_polygons_intersecting(
        #     self.polygon(), other.polygon(), self.velocity * dt, other.velocity * dt
        # )
    
    def polygon(self) -> np.ndarray:
        points = np.array(
            [
                [-self.LENGTH / 2, -self.WIDTH / 2],
                [-self.LENGTH / 2, +self.WIDTH / 2],
                [+self.LENGTH / 2, +self.WIDTH / 2],
                [+self.LENGTH / 2, -self.WIDTH / 2],
            ]
        ).T
        c, s = np.cos(self.heading), np.sin(self.heading)
        rotation = np.array([[c, -s], [s, c]])
        points = (rotation @ points).T + np.tile(self.position, (4, 1))
        return np.vstack([points, points[0:1]])
    
    def __str__(self):
        return f"{self.__class__.__name__} #{id(self) % 1000}: at {self.position}"

    def __repr__(self):
        return self.__str__()


class Obstacle(CorridorObject):

    """Obstacles on the road."""

    def __init__(
        self, corridor, position: Sequence[float], heading: float = 0, speed: float = 0
    ):
        super().__init__(corridor, position, heading, speed)
        self.solid = True
    
    

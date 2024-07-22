import numpy as np
from typing import List, Optional, Tuple, Union

from uam_env import utils
from uam_env.config import kinematics_config
from uam_env.corridor.corridor import Corridor, CorridorObject, StraightLane
from uam_env.vehicle.kinematics import Vehicle
from uam_env.utils import Vector


class IDMVehicle(Vehicle):
    """
    TODO: Refer to https://en.wikipedia.org/wiki/Intelligent_driver_model
    """
    ACC_MAX = kinematics_config.ACC_MAX # m/s^2
    
    COMFORT_ACC_MAX = kinematics_config.COMFORT_ACC_MAX # m/s^2
    
    COMFORT_ACC_MIN = kinematics_config.COMFORT_ACC_MIN # m/s^2
    
    DISTANCE_WANTED = kinematics_config.BUFFER_SPACING_M + \
        kinematics_config.LENGTH_M
        
    TIME_WANTED = kinematics_config.TIME_WANTED # seconds
    
    def __init__(
        self,
        corridor:Corridor,
        position:Vector,
        roll_dg:float=0,
        pitch_dg:float=0,
        heading_dg:float=0,
        speed:float=15,
        target_lane:str=None,
        timer:float=None) -> None:
        super().__init__(
            corridor=corridor,
            position=position,
            roll_dg=roll_dg,
            pitch_dg=pitch_dg,
            heading_dg=heading_dg,
            speed=speed
        )

        self.target_lane = target_lane
        self.timer = timer
        
    def randomize_behavior(self) -> None:
        pass
    
    @classmethod
    def create_from(cls, vehicle:Vehicle) -> "IDMVehicle":
        return cls(
            corridor=vehicle.corridor,
            position=vehicle.position,
            roll_dg=vehicle.roll_dg,
            pitch_dg=vehicle.pitch_dg,
            heading_dg=vehicle.heading_dg,
            speed=vehicle.speed,
            vehicle=vehicle.target_lane,
            timer=getattr(vehicle, "timer", None)
        )
    
    def act(self, action: Union[dict,str]=None) -> None:
        """
        Execute an action.
        
        For now, no action is supported because the 
        vehicle makes its own decisions.
        """
        if self.crashed:
            return
        
        action = {}
        
    def lane_distance_to(
        self, other:"CorridorObject", 
        lane:StraightLane) -> float:
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
    
    def desired_gap(
        self, 
        ego_vehicle: Vehicle,
        front_vehicle: Vehicle = None,
        projected: bool = False) -> float:
        """
        Compute the desired distance between a vehicle and its leading vehicle.
        
        :param ego_vehicle: the vehicle whose desired gap is to be computed
        :param front_vehicle: the vehicle in front of the ego vehicle
        :param projected: project the current speed on the front vehicle speed
        :return: the desired distance in meters
        """
        d0 = self.DISTANCE_WANTED
        tau = self.TIME_WANTED
        ab = -self.COMFORT_ACC_MAX * self.COMFORT_ACC_MIN
        # dv = (
        #     np.dot(ego_vehicle.velocity - front_vehicle.velocity,)
        # )
        if projected:
            dv = np.dot(ego_vehicle.velocity, front_vehicle.velocity)
        else:
            dv = ego_vehicle.speed - front_vehicle.speed
            
        desired_gap = (
            d0 + ego_vehicle.speed * tau + 
            ego_vehicle.speed * dv / (2 * np.sqrt(ab))
        )
         
        return desired_gap
          
    def change_lane_policy(self) -> None:
        """
        Change lane policy
        
        Based on:
        - Frequency
        - Closeness of the vehicle in the target lane
        - MOBIL model
        """
        if self.lane_index != self.target_lane:
            for v in self.road.vehicles:
                if ( v is not self 
                    and v.lane_index != self.target_lane_index
                    and isinstance(v, IDMVehicle) 
                    and v.lane_index == self.lane_index):
                    distance = self.lane_distance_to(v)
                    desired_distance = self.desired_gap(self, v)
                    
                    if 0 < distance < desired_distance:
                        if self._is_safe_to_change(v):
                            self.target_lane_index = self.target_lane_index
                            break
        return 
    
    
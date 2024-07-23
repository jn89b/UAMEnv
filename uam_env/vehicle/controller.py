import copy 
from typing import List, Optional, Tuple, Union

import numpy as np

from uam_env.corridor.corridor import Corridor, StraightLane
from uam_env.utils import Vector
from uam_env.vehicle.kinematics import Vehicle
from uam_env.config import controller_config
from uam_env import utils

class Controller():
    KP_LATERAL = controller_config.KP_LATERAL
    TAU_ACC = controller_config.TAU_ACC
    TAU_HEADING = controller_config.TAU_HEADING
    TAU_LATERAL = controller_config.TAU_LATERAL
    TAU_PURSUIT = controller_config.TAU_PURSUIT
    TAU_ROLL = 0.5
    
    KP_HEADING = 1 / TAU_HEADING
    KP_LATERAL = 1 / TAU_LATERAL
    KP_PHI = 1 / TAU_PURSUIT
    
    def __init__(self) -> None:
        pass
    
    def steering_control(self, target_lane:StraightLane,
                         ego_position:Vector,
                         ego_speed:float,
                         ego_heading_rad:float,
                         ego_roll_rad:float) -> Tuple[float, float, float, float]:
        """
        Steer the vehicle to the target lane 
        
        1. Lateral position is controlled by a proportional controller 
            yielding a lateral speed command
        2. Lateral speed command is converted to a heading reference
        3. Heading is controlled by a proportional controller yielding 
            a heading rate command
        4. Heading rate command is converted to a steering angle

        :param target_lane_index: index of the lane to follow
        :return: a steering wheel angle command [rad]
        """
        lane_coords = target_lane.local_coordinates(ego_position)
        lane_next_coords = lane_coords[0] + ego_speed * self.TAU_PURSUIT
        lane_future_heading = target_lane.heading_at(lane_next_coords)
        
        #lateral position control
        lateral_speed_command = -self.KP_LATERAL * lane_coords[1]
        #lateral speed to heading reference
        heading_ref = np.arcsin(
            np.clip(lateral_speed_command / utils.not_zero(ego_speed), -1, 1)
        )
        heading_error = utils.wrap_to_pi(heading_ref - ego_heading_rad)

        heading_rate_command = self.KP_HEADING * heading_error
        roll_desired = np.arctan2(heading_rate_command * ego_speed, 9.81)
        #make sure not nan
        if np.isnan(roll_desired):
            roll_desired = 0.0

        #check if the desired roll is within the limits
        roll = np.clip(roll_desired, 
                       -controller_config.ROLL_MAX, 
                       controller_config.ROLL_MAX)
        
        roll_rate_command = (roll - ego_roll_rad) / self.TAU_ROLL
        
        return (heading_ref, heading_rate_command, roll, roll_rate_command)
    
    def pitch_control(self, target_lane:StraightLane, 
                             ego_vehicle:Vehicle) -> Tuple[float, float]:
        lane_coords = target_lane.local_coordinates(ego_vehicle.position)
        desired_altitude = lane_coords[2] - ego_vehicle.position[2]
        
        pitch_command = self.KP_PHI * (desired_altitude - ego_vehicle.pitch_rad)
        pitch_rate = (pitch_command - ego_vehicle.pitch_rad) / self.TAU_ACC
        
        return (pitch_rate, pitch_command)
        
    
    def acceleration_control(self, ego_vehicle:Vehicle,
                             front_vehicle:Vehicle, rear_vehicle:Vehicle) -> float:
        pass
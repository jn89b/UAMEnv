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
    
    KP_HEADING = controller_config.KP_HEADING
    KD_HEADING = controller_config.KD_HEADING
    KP_ROLL = 0.1
    KP_LATERAL = 1 / TAU_LATERAL
    KP_PHI = 1 / TAU_PURSUIT
    
    def __init__(self) -> None:
        self.old_heading_error = 0.0
    
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
        long, lat = target_lane.local_coordinates(ego_position)
        lane_next_coords = long + (ego_speed * self.TAU_PURSUIT)
        lane_future_heading = target_lane.heading_at(lane_next_coords)
        #lateral position control
        lateral_speed_command = -self.KP_LATERAL * lat
        #lateral speed to heading reference
        dx = lane_next_coords - long
        dy = lateral_speed_command
        heading_command = np.arctan2(dy, dx)     
        heading_ref = lane_future_heading + np.clip(heading_command,
                    -np.pi/4, np.pi/4)
        #heading control
        heading_error = utils.wrap_to_pi(heading_ref - ego_heading_rad)
        P_heading = self.KP_HEADING * heading_error
        D_heading = self.KD_HEADING * (heading_error - self.old_heading_error)/0.1
        heading_gain = P_heading + D_heading
        heading_rate_command = self.KP_ROLL * heading_error
        roll_desired = np.arctan2(heading_rate_command * ego_speed, 9.81)
        
        if abs(heading_error) <= np.deg2rad(1):
            roll_desired = 0.0
            heading_gain = 0.0
        
        #make sure not nan
        if np.isnan(roll_desired):
            roll_desired = 0.0
        
        #check if the desired roll is within the limits
        roll = np.clip(roll_desired, 
                       -controller_config.ROLL_MAX, 
                       controller_config.ROLL_MAX)
        roll_rate_command = (roll - ego_roll_rad) / self.TAU_ROLL
        self.old_heading_error = heading_error
        
        return (heading_gain, heading_rate_command, roll, roll_rate_command)
    
    def pitch_control(self, target_lane:StraightLane, 
                             ego_vehicle:Vehicle) -> Tuple[float, float]:
        lane_coords = target_lane.local_coordinates(ego_vehicle.position)
        target_lane_z = target_lane.start[2]
        desired_altitude = target_lane_z - ego_vehicle.position[2]
        
        current_pitch_rad = np.deg2rad(ego_vehicle.pitch_dg)
        pitch_command = self.KP_PHI * (desired_altitude - current_pitch_rad)
        pitch_rate = (pitch_command - current_pitch_rad) / self.TAU_ACC
        
        return (pitch_rate, pitch_command)
        
    
import numpy as np
from typing import List, Optional, Tuple, Union

from uam_env import utils
from uam_env.config import kinematics_config
from uam_env.corridor.corridor import Corridor, CorridorObject, StraightLane
from uam_env.vehicle.kinematics import Vehicle
from uam_env.utils import Vector
from uam_env.vehicle.controller import Controller

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
    
    DELTA = 4.0 # exponent for the velocity term

    POLITENESS = 0.3 # coefficient for politeness [0, 1]
    
    LANE_CHANGE_MIN_ACC_GAIN = 0.2 # m/s^2
    
    LANE_CHANGE_MAX_BRAKING_IMPOSED = 2.0 # m/s^2
    
    def __init__(
        self,
        corridor:Corridor,
        position:Vector,
        roll_dg:float=0,
        pitch_dg:float=0,
        heading_dg:float=0,
        speed:float=15,
        target_lane:str=None,
        timer:float=None,
        controller:Controller=None) -> None:
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
        if controller is None:
            self.controller = Controller()
        else:
            self.controller = controller
        
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
    
    def follow_corridor(self) -> None:
        """
        The vehicle follows the corridor
        """
        pass
                
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

        # # else, at a given frequency,
        if not utils.do_every(self.LANE_CHANGE_DELAY, self.timer):
            return
        self.timer = 0

        if self.mobil(self, self.target_lane_index):
            self.target_lane_index = self.target_lane_index
            
    def mobil(self, ego_vehicle:Vehicle,
              lane_index: str) -> bool:
        """
        MOBIL lane change model: Minimizing Overall Braking Induced by a Lane change

            The vehicle should change lane only if:
            - after changing it (and/or following vehicles) can accelerate more;
            - it doesn't impose an unsafe braking on its new following vehicle.

        :param lane_index: the candidate lane for the change
        :return: whether the lane change should be performed
        """
        # is the maneuver unsafe for the new following vehicle?
        new_preceding, new_following = self.corridor.neighbor_vehicles(self, lane_index)
        new_following_a = self.acceleration(
            ego_vehicle=new_following, front_vehicle=new_preceding
        )
        new_following_pred_a = self.acceleration(
            ego_vehicle=new_following, front_vehicle=self
        )
        if new_following_pred_a < -self.LANE_CHANGE_MAX_BRAKING_IMPOSED:
            return False

        # Do I have a planned route for a specific lane which is safe for me to access?
        old_preceding, old_following = self.corridor.neighbour_vehicles(self)
        self_pred_a = self.acceleration(ego_vehicle=self, front_vehicle=new_preceding)
    
        self_a = self.acceleration(ego_vehicle=self, front_vehicle=old_preceding)
        old_following_a = self.acceleration(
            ego_vehicle=old_following, front_vehicle=self
        )
        old_following_pred_a = self.acceleration(
            ego_vehicle=old_following, front_vehicle=old_preceding
        )
        jerk = (
            self_pred_a
            - self_a
            + self.POLITENESS
            * (
                new_following_pred_a
                - new_following_a
                + old_following_pred_a
                - old_following_a
            )
        )
        if jerk < self.LANE_CHANGE_MIN_ACC_GAIN:
            return False

        # All clear, let's go!
        return True

    def acceleration(
        self, 
        ego_vehicle:Vehicle,
        front_vehicle:Vehicle,
        rear_vehicle:Vehicle) -> float:
        """
        Compute an acceleration command with the Intelligent Driver Model.

        The acceleration is chosen so as to:
        - reach a target speed;
        - maintain a minimum safety distance (and safety time) 
        w.r.t the front vehicle.

        :param ego_vehicle: the vehicle whose desired acceleration is to be computed. It does not have to be an
                            IDM vehicle, which is why this method is a class method. This allows an IDM vehicle to
                            reason about other vehicles behaviors even though they may not IDMs.
        :param front_vehicle: the vehicle preceding the ego-vehicle
        :param rear_vehicle: the vehicle following the ego-vehicle
        :return: the acceleration command for the ego-vehicle [m/s2]
        """
        if not ego_vehicle or not isinstance(ego_vehicle, Vehicle):
            return 0
        ego_target_speed = getattr(ego_vehicle, "target_speed", 0)
        if ego_vehicle.lane and ego_vehicle.lane.speed_limit is not None:
            ego_target_speed = np.clip(
                ego_target_speed, 0, ego_vehicle.lane.speed_limit
            )
        acceleration = self.COMFORT_ACC_MAX * (
            1
            - np.power(
                max(ego_vehicle.speed, 0) / abs(utils.not_zero(ego_target_speed)),
                self.DELTA,
            )
        )

        if front_vehicle:
            d = ego_vehicle.lane_distance_to(front_vehicle)
            acceleration -= self.COMFORT_ACC_MAX * np.power(
                self.desired_gap(ego_vehicle, front_vehicle) / utils.not_zero(d), 2
            )
        return acceleration


    def act(self, action: Union[dict,str]=None) -> None:
        """
        Execute an action.
        
        For now, no action is supported because the 
        vehicle makes its own decisions.
        """
        if self.crashed:
            return
        # corridor: Corridor = self.corridor
        self.follow_corridor()
        action = {
            'roll_rate_cmd': None,
            'pitch_rate_cmd': None,
            'heading_rate_cmd': None,
            'speed_cmd': None
        }
        # Lateral Control
        heading_rate_cmd, roll_rate_cmd = self.controller.steering_control(
            target_lane=self.corridor.lanes[self.target_lane],
            ego_position=self.position,
            ego_speed=self.speed,
            ego_heading_rad=self.heading,
            ego_roll_rad=self.roll
        )
        
        pitch_rate_cmd, pitch_command = self.controller.pitch_control(
            target_lane=self.corridor.lanes[self.target_lane],
            vehicle = self,
        )
        # Longitudinal Control
        front_vehicle, rear_vehicle = self.corridor.neighbor_vehicles(
            ego_vehicle=self, 
            lane_id=self.lane_index)
        
        acceleration = self.acceleration(
            ego_vehicle=self, 
            front_vehicle=front_vehicle, 
            rear_vehicle=rear_vehicle)
        
        # When changing lane, check both current and target lanes
        if self.lane_index != self.target_lane_index:
            front_vehicle, rear_vehicle = self.corridor.neighbor_vehicles(
                self, self.target_lane_index
            )
            target_idm_acceleration = self.acceleration(
                ego_vehicle=self, front_vehicle=front_vehicle, rear_vehicle=rear_vehicle
            )
            action["acceleration"] = min(
                action["acceleration"], target_idm_acceleration
            )
        # action['acceleration'] = self.recover_from_stop(action['acceleration'])
        action["acceleration"] = np.clip(
            action["acceleration"], -self.ACC_MAX, self.ACC_MAX
        )
        
        Vehicle.act(
            self, action
        )  # Skip ControlledVehicle.act(), or the command will be overriden.

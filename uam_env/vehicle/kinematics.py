import copy 
import casadi as ca
import random
from collections import deque
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

# from uam_env.corridor.corridor import Corridor
from uam_env.utils import Vector
from uam_env.config import kinematics_config
from uam_env.vehicle.objects import CorridorObject

if TYPE_CHECKING:
    from uam_env.corridor.corridor import Corridor

import numpy as np

class Vehicle(CorridorObject):
    """
    A moving vehicle in the environment 
    This vehicle represents the kinematics of a non-holonomic fixed-wing aircraft
    TODO: Refer to https://en.wikipedia.org/wiki/Intelligent_driver_model
    """
    LENGTH_m = kinematics_config.LENGTH_m #Length of the vehicle in meters
    WIDTH_m = kinematics_config.WIDTH_m  #width of the vehicle in meters 
    MAX_SPEED_MS = kinematics_config.MAX_SPEED_MS #Maximum speed of the vehicle in m/s
    MIN_SPEED_MS = kinematics_config.MIN_SPEED_MS #Minimum speed of the vehicle in m/s
    HISTORY_SIZE = kinematics_config.HISTORY_SIZE #Size of the history buffer
    
    def __init__(self,
                 corridor:"Corridor",
                 position:Vector,
                 roll_dg:float = 0,
                 pitch_dg:float = 0,
                 heading_dg:float = 0,
                 speed:float = 15) -> None:
        self.corridor = corridor
        self.position = position
        self.heading = heading_dg
        self.pitch_dg = pitch_dg
        self.roll_dg = roll_dg
        self.speed = speed
                
        # this is for 
        self.action = None
        self.crashed = False
        self.impact = None
        self.log = []
        self.history = deque(maxlen=self.HISTORY_SIZE)

    @classmethod
    def create_random(
        cls,
        corridor:"Corridor",
        speed: float = None,
        lane_from: Optional[str] = None,
        lane_to: Optional[str] = None,
        lane_id: Optional[int] = None,
        spacing: float = 0.0
    ) -> "Vehicle":
        """
        Create a random vehicle on the road.

        The lane and /or speed are chosen randomly, 
        while longitudinal position is chosen behind the last
        vehicle in the road with density based on the number of lanes.

        :param road: the road where the vehicle is driving
        :param speed: initial speed in [m/s]. If None, will be chosen randomly
        :param lane_from: start node of the lane to spawn in
        :param lane_to: end node of the lane to spawn in
        :param lane_id: id of the lane to spawn in
        :param spacing: ratio of spacing to the front vehicle, 1 being the default
        :return: A vehicle with random position and/or speed
        """
        if speed is None:
            speed = np.random.uniform(cls.MIN_SPEED_MS, cls.MAX_SPEED_MS)
        
        lane_names = list(corridor.lane_network.lanes.keys())
        _from = lane_from or random.choice(lane_names)
        _to = lane_to or random.choice(lane_names)
        _id = _from
        lane = corridor.lane_network.lanes[_from]
        if speed is None:
            speed = np.random.uniform(
                cls.MIN_SPEED_MS, 
                cls.MAX_SPEED_MS)

        default_spacing = kinematics_config.BUFFER_SPACING_M \
            + (1 * speed)
            
        # position = lane.position(
        #     longitudinal=lane.length_m,
        #     lateral=0
        # )
        offset = spacing*default_spacing   
        if len(corridor.vehicles):
            x0 = np.max([v.position[0] for v in corridor.vehicles])
        else:
            x0 = 3 * default_spacing     

        x0 += offset * corridor.np_random.uniform(0.8, 1.2)
        position = lane.position(longitudinal=x0,lateral=0)

        lane_heading = lane.heading_at()
        vehicle = cls(corridor=corridor, 
                      position=position, 
                      speed=speed,
                      heading_dg=np.rad2deg(lane_heading))
        
        return vehicle 

class DataHandler():
    def __init__(self) -> None:
        self.x = []
        self.y = []
        self.z = []
        self.roll = []
        self.pitch = []
        self.yaw = []
        self.u = []
        self.time = []
        self.rewards = []
        
    def update_data(self,info_array:np.ndarray):
        self.x.append(info_array[0])
        self.y.append(info_array[1])
        self.z.append(info_array[2])
        self.roll.append(info_array[3])
        self.pitch.append(info_array[4])
        self.yaw.append(info_array[5])
        self.u.append(info_array[6])
        # self.time.append(info_array[7])
        
    def update_reward(self, reward:float) -> None:
        self.rewards.append(reward)
        
    def update_time(self, time:float) -> None:
        self.time.append(time)


class Plane():
    def __init__(self, 
                 include_time:bool=False,
                 dt_val:float=0.05,
                 max_roll_dg:float=45,
                 max_pitch_dg:float=25,
                 min_airspeed_ms:float=12,
                 max_airspeed_ms:float=30) -> None:
        self.include_time = include_time
        self.dt_val = dt_val
        self.define_states()
        self.define_controls() 
        
        self.max_roll_rad = np.deg2rad(max_roll_dg)
        self.max_pitch_rad = np.deg2rad(max_pitch_dg)
        self.min_airspeed_ms = min_airspeed_ms
        self.max_airspeed_ms = max_airspeed_ms
        self.airspeed_tau = 0.05 #response of system to airspeed command
        self.pitch_tau = 0.02 #response of system to pitch command
        self.state_info = None
        self.data_handler = DataHandler()
    
    def set_info(self, state_info:np.ndarray) -> None:
        self.state_info = state_info
        self.data_handler.update_data(state_info)
    
    def set_time(self, time:float) -> None:
        self.data_handler.update_time(time)
    
    def get_info(self) -> np.ndarray:
        return self.state_info
    
    def define_states(self):
        """define the states of your system"""
        #positions off the world in NED Frame
        self.x_f = ca.SX.sym('x_f')
        self.y_f = ca.SX.sym('y_f')
        self.z_f = ca.SX.sym('z_f')

        #attitude
        self.phi_f = ca.SX.sym('phi_f')
        self.theta_f = ca.SX.sym('theta_f')
        self.psi_f = ca.SX.sym('psi_f')
        self.v = ca.SX.sym('v')

        if self.include_time:
            self.states = ca.vertcat(
                self.x_f,
                self.y_f,
                self.z_f,
                self.phi_f,
                self.theta_f,
                self.psi_f, 
                self.v)
        else:
            self.states = ca.vertcat(
                self.x_f,
                self.y_f,
                self.z_f,
                self.phi_f,
                self.theta_f,
                self.psi_f,
                self.v 
            )

        self.n_states = self.states.size()[0] #is a column vector 

    def define_controls(self):
        """
        controls for your system
        The controls are the roll, pitch, yaw, and airspeed
        If u_psi is 0 the plane will fly straight
        """
        self.u_phi = ca.SX.sym('u_phi')
        self.u_theta = ca.SX.sym('u_theta')
        self.u_psi = ca.SX.sym('u_psi')
        self.v_cmd = ca.SX.sym('v_cmd')

        self.controls = ca.vertcat(
            self.u_phi,
            self.u_theta,
            self.u_psi,
            self.v_cmd
        )
        self.n_controls = self.controls.size()[0] 

    def set_state_space(self):
        """
        define the state space of your system
        NED Frame
        """
        self.g = 9.81 #m/s^2
        #body to inertia frame
        self.v_dot = (self.v_cmd - self.v)*(self.dt_val/self.airspeed_tau)
        self.x_fdot = self.v * ca.cos(self.theta_f) * ca.cos(self.psi_f) 
        self.y_fdot = self.v * ca.cos(self.theta_f) * ca.sin(self.psi_f)
        self.z_fdot = -self.v * ca.sin(self.theta_f)
        
        self.phi_fdot   = (self.u_phi - self.phi_f) *(self.dt_val/self.pitch_tau)
        self.theta_fdot = (self.u_theta - self.theta_f) *(self.dt_val/self.pitch_tau)
        
        #check if the denominator is zero
        self.psi_fdot   = self.u_psi + (self.g * (ca.tan(self.phi_f) / self.v_cmd))
        
        # self.t_dot = self.t 
        
        if self.include_time:
            self.z_dot = ca.vertcat(
                self.x_fdot,
                self.y_fdot,
                self.z_fdot,
                self.phi_fdot,
                self.theta_fdot,
                self.psi_fdot,
                self.v_dot
            )
        else:
            self.z_dot = ca.vertcat(
                self.x_fdot,
                self.y_fdot,
                self.z_fdot,
                self.phi_fdot,
                self.theta_fdot,
                self.psi_fdot,
                self.v_dot
            )

        #ODE function
        self.function = ca.Function('f', 
            [self.states, self.controls], 
            [self.z_dot])
        
    def update_reward(self, reward:float) -> None:
        self.data_handler.update_reward(reward)
        
    def rk45(self, x, u, dt, use_numeric:bool=True):
        """
        Runge-Kutta 4th order integration
        x is the current state
        u is the current control input
        dt is the time step
        use_numeric is a boolean to return the result as a numpy array
        """
        k1 = self.function(x, u)
        k2 = self.function(x + dt/2 * k1, u)
        k3 = self.function(x + dt/2 * k2, u)
        k4 = self.function(x + dt * k3, u)
        
        next_step = x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        
        #clip the values of the angles
        next_step[3] = np.clip(next_step[3], 
                               -self.max_roll_rad, 
                               self.max_roll_rad)
        next_step[4] = np.clip(next_step[4], 
                               -self.max_pitch_rad, 
                               self.max_pitch_rad)
                       
        #wrap yaw from -pi to pi
        if next_step[5] > np.pi:
            next_step[5] -= 2*np.pi
        elif next_step[5] < -np.pi:
            next_step[5] += 2*np.pi
            
        #clip the airspeed
        next_step[6] = np.clip(next_step[6], 
                               self.min_airspeed_ms, 
                               self.max_airspeed_ms)            
                       
        #return as numpy row vector
        if use_numeric:
            next_step = np.array(next_step).flatten()
            return next_step
        else:
            return x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    

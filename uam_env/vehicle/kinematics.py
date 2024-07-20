import copy 
from collections import deque
from typing import List, Optional, Tuple, Union

from uam_env.corridor.corridor import Corridor
from uam_env.utils import Vector
from uam_env.config import kinematics_config

import numpy as np

class Kinematics():
    """
    A moving vehicle in the environment 
    This vehicle represents the kinematics of a non-holonomic fixed-wing aircraft
    """
    LENGTH_m = kinematics_config.LENGTH_m #Length of the vehicle in meters
    WIDTH_m = kinematics_config.WIDTH_m  #width of the vehicle in meters 
    MAX_SPEED_MS = kinematics_config.MAX_SPEED_MS #Maximum speed of the vehicle in m/s
    MIN_SPEED_MS = kinematics_config.MIN_SPEED_MS #Minimum speed of the vehicle in m/s
    HISTORY_SIZE = kinematics_config.HISTORY_SIZE #Size of the history buffer
    
    def __init__(self,
                 corridor:Corridor,
                 position:Vector,
                 heading:float = 0,
                 speed:float = 15) -> None:
        self.corridor = corridor
        self.position = position
        self.heading = heading
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
        corridor:Corridor,
        speed: float = None,
    ) -> "Kinematics":
        """
        Create a random vehicle
        """
        if speed is None:
            speed = np.random.uniform(cls.MIN_SPEED_MS, cls.MAX_SPEED_MS)
            
        lane_names = list(corridor.lanes.keys())
        lane_to_use = np.random.choice(lane_names)
        
from typing import Dict, Text
from uam_env.corridor.corridor import Corridor
from uam_env.vehicle.kinematics import Vehicle
from uam_env.vehicle.behavior import IDMVehicle
from uam_env.config import kinematics_config
from typing import Dict, List, Optional, Text, Tuple, TypeVar
from uam_env.config import env_config
import numpy as np
import gymnasium as gym

"""
REMEMBER KEEP THIS SIMPLE THEN APPLY MORE COMPLEXITY

Let's keep this simple
For UAM from our literature review, we need to consider the following:
- We will have lateral and vertical passing lanes
- We will assume that we are assigned a corridor from a 
    Universal Service Supplier and a passing lane to navigate through traffic
- Our agent needs has an action space of the following:
    - Stay on lateral lane
    - Get to the lateral passing zone 
    - Stay on the vertical lane 
    - Get to the vertical passing zone
    - Stay on lateral lane and get to the vertical passing zone
    - Stay on vertical lane and get to the lateral passing zone
    
    
"""

class UAMEnv(gym.Env):
    """
    This is the main environment for the UAM
    """
    def __init__(self) -> None:
        # super().__init__()
        self.config = self.default_config()
        self.dt = 0.1
        
        #Running variables
        self.time = 0 #simulation time 
        self.steps = 0 
        self.done = False
    
    @classmethod
    def default_config(cls) -> dict:
        #config = super().default_config()
        config = {}
        # eventually we want to have a configuration file
        config.update({
            "n_vehicles": 10,
            "n_corridors": 1,
            "non_controlled_vehicles": env_config.NON_CONTROLLED_VEHICLES,
            "controlled_vehicles": env_config.CONTROLLED_VEHICLES,
            "lane_network": None,
            "record_history": True,
            "initial_lane_id": None,
            'ego_spacing': 2.0,
            "duration": env_config.DURATION, # seconds
            "n_observation": 10, #TODO: Define observation spaceS
            "n_actions": 6, #TODO: Define Number of actions
            "n_rewards": 1, #TODO: Define Number of rewards
            "n_steps": 1000, #TODO: Define Number of steps
            "n_episodes": 100 #TODO: Define Number of episodes
        })
        
        return config 
    
    def _create_corridors(self) -> None:
        """
        User needs to define how many corridors are in the environment
        """
        #TODO: Let's just build one corridor for now
        self.corridors = Corridor(
            lane_network=self.config["lane_network"],
            np_random = self.np_random,
            record_history=self.config["record_history"]
        )
        
    def reset(self) -> None:
        """
        Reset the environment
        """
        self._create_corridors()
        self._create_vehicles()
        
    def _create_vehicles(self) -> None:
        """
        User needs to define how many vehicles are 
        in the environment  
        """
        self.controlled_vehicles = []
        other_vehicles = []
        for _ in range(self.config["non_controlled_vehicles"]):
            random_speed = np.random.uniform(
                kinematics_config.MIN_SPEED_MS,
                kinematics_config.MAX_SPEED_MS
            )
            
            vehicle = IDMVehicle.create_random(
                corridor=self.corridors,
                speed=random_speed,
                lane_from=self.config["initial_lane_id"],
                lane_id=self.config["initial_lane_id"],
                spacing=self.config["ego_spacing"]                
            )
            
            other_vehicles.append(vehicle)
        
        #combine the controlled vehicles and other vehicles
        self.corridors.vehicles = other_vehicles
        # overall_vehicles = self.controlled_vehicles + other_vehicles    
        # self.corridors.vehicles = overall_vehicles
        
    def simulate(self, action=None) -> None:
        """
        Simulate the environment
        """
        self.corridors.act()
        self.corridors.step(self.dt)
        self.steps += 1
    
        
        
    
    
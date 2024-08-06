from typing import Dict, Text
from uam_env.corridor.corridor import Corridor
from uam_env.vehicle.kinematics import Vehicle
from uam_env.vehicle.behavior import IDMVehicle, DiscreteVehicle
from uam_env.config import kinematics_config
from typing import Dict, List, Optional, Text, Tuple, TypeVar
from uam_env.config import env_config, kinematics_config, lane_config
import numpy as np
import gymnasium as gym
from gym import spaces

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
        self.n_neighbors = 2
        self.action_space = spaces.Discrete(
            env_config.NUM_ACTIONS)
        
        self.state_constraints = kinematics_config.state_constraints
        self.ego_obs_space = self.init_ego_observation()
        
        self.observation_space = spaces.Dict(
            {
                "ego": self.ego_obs_space
            })

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
    
    @property
    def vehicle(self) -> Vehicle:
        """First (default) controlled vehicle."""
        return self.controlled_vehicles[0] \
            if self.controlled_vehicles else None

    def init_ego_observation(self) -> spaces.Box:
        """
        State orders are as follows:
        x, (east) (m)
        y, (north) (m)
        z, (up) (m)
        roll, (rad)
        pitch, (rad)
        yaw, (rad)
        airspeed (m/s)
        
        For our neighbors we want to consider 
        relative positions and velocities of the vehicles
        """

        low_obs = [
            self.state_constraints['x_min'],
            self.state_constraints['y_min'],
            self.state_constraints['z_min'],
            self.state_constraints['phi_min'],
            self.state_constraints['theta_min'],
            self.state_constraints['psi_min'],
            self.state_constraints['airspeed_min']
        ]
        
        high_obs = [
            self.state_constraints['x_max'],
            self.state_constraints['y_max'],
            self.state_constraints['z_max'],
            self.state_constraints['phi_max'],
            self.state_constraints['theta_max'],
            self.state_constraints['psi_max'],
            self.state_constraints['airspeed_max']
        ]
        
        #TODO: update this to allow user to define n nearest neighbors

        #Order will be:
        # neighbor 1 (relative distance, relative velocity)
        n_states = 2
        relative_max_distance = lane_config.LANE_LENGTH_M
        for _ in range(self.n_neighbors):
            for _ in range(n_states):
                low_obs.append(-relative_max_distance)
                high_obs.append(relative_max_distance)
                low_obs.append(-kinematics_config.MAX_SPEED_MS)
                high_obs.append(kinematics_config.MAX_SPEED_MS)
        
        obs_space = spaces.Box(low=np.array(low_obs),
                                            high=np.array(high_obs),
                                            dtype=np.float32)
                
        return obs_space
    
    def _create_corridors(self) -> None:
        """
        User needs to define how many corridors are in the environment
        """
        #TODO: Let's just build one corridor for now
        self.corridors = Corridor(
            lane_network=self.config["lane_network"],
            np_random = self.np_random,
            record_history=self.config["record_history"])
        
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
        
        n_non_controlled = self.config["non_controlled_vehicles"]
        total_vehicles = self.config["controlled_vehicles"] \
            + n_non_controlled
        
        #random index to spawn the controlled vehicle
        random_number = np.random.randint(0, total_vehicles)
        
        for i in range(total_vehicles):
            random_speed = np.random.uniform(
                kinematics_config.MIN_SPEED_MS,
                kinematics_config.MAX_SPEED_MS
            )
            
            if i == random_number:
                vehicle = DiscreteVehicle.create_random(
                    corridor=self.corridors,
                    speed=random_speed,
                    lane_from=self.config["initial_lane_id"],
                    lane_id=self.config["initial_lane_id"],
                    spacing=self.config["ego_spacing"]                
                )
                self.controlled_vehicles.append(vehicle)
            else:
                vehicle = IDMVehicle.create_random(
                    corridor=self.corridors,
                    speed=random_speed,
                    lane_from=self.config["initial_lane_id"],
                    lane_id=self.config["initial_lane_id"],
                    spacing=self.config["ego_spacing"]                
                )
            
            self.corridors.vehicles.append(vehicle)
            
    def _get_observation(self) -> dict:
        """
        For the observation we want to get:
            - Ego vehicle state [x, y, z, roll, pitch, yaw, airspeed]
            - Then we want to query our 2 nearest neighbors for their 
                relative distance and velocity
        """
        ego_state = self.vehicle.plane.get_info()
        #now we need to get the nearest neighbors
        n_neighbors = self.n_neighbors
        neighbors = self.corridors.close_objects_to(
            vehicle=self.vehicle,
            distance_threshold=250,
            count=n_neighbors,
            see_behind=True,
            sort=True,
            vehicles_only=True)
        for k, v in neighbors.items():
            #auto fill the relative position and velocity
            rel_pos_vel = np.array([
                lane_config.LANE_LENGTH_M,
                lane_config.SPEED_LIMIT_MS,
            ])
            if v is not None:
                rel_pos_vel[0] = v['distance']
                rel_pos_vel[1] = v['velocity']
        
            #append the ego state to the neighbors
            ego_state = np.append(ego_state, rel_pos_vel)

        return {
            "ego": ego_state
        }
    
    def simulate(self, action=None) -> None:
        """
        Simulate the environment
        """
        self.corridors.act(action)
        self.corridors.step(self.dt)
        self.steps += 1
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Perform an action and step the environment dynamics.

        The action is executed by the ego-vehicle, and all other vehicles on the road performs their default behaviour
        for several simulation timesteps until the next decision making step.

        :param action: the action performed by the ego-vehicle
        :return: a tuple (observation, reward, terminated, truncated, info)
        
        For now the action will be an int since we are using a discrete action space
        
        """
        
        #TODO: Add controled vehicle and test it out
        if self.corridors is None or self.vehicle is None:
            raise NotImplementedError(
                "The corridor and vehicle must be initialized in the environment implementation"
            )

        reward = 0
        terminated = False
        truncated = False
        info = {}
    
        obs = self._get_observation()
        print(obs)
        # self.time += 1 / self.config["policy_frequency"]
        self.time += 1
        self.simulate(action)

        # obs = self.observation_type.observe()
        # reward = self._reward(action)
        # terminated = self._is_terminated()
        # truncated = self._is_truncated()
        # info = self._info(obs, action)
        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info
        
    
    
"""
- Spawn corridor with the vehicles
- Check if locations are correct
- Have vehicles move in the corridor
- See if logic is correct
- Visualize the corridor and vehicles
- Animate the trajectory of the vehicles
"""

from uam_env.envs.uam_env import UAMEnv

import numpy as np

environment = UAMEnv()
environment.default_config()
environment._create_corridors()
environment._create_vehicles()
#print the configuration of the environment
print(environment.config)
from uam_env.envs.uam_env import UAMEnv
from uam_env.visualizer.visualizer import Visualizer
import numpy as np
import matplotlib.pyplot as plt
import copy 

# Create the environment
uam_env = UAMEnv()
uam_env.reset()

# Simulate the environment
for i in range(100):
    # Take a random action from the action space (discrete action)
    action = uam_env.action_space.sample()
    obs, reward, terminated, truncated, info = uam_env.step(action)
    print("reward: ", reward)
    

vehicles = uam_env.corridors.vehicles
#check if crash
num_vehicles = len(vehicles)
num_crash = 0
crashed_vehicles = []
for v in vehicles:
    if v.crashed:
        num_crash += 1
        crashed_vehicles.append(v)
print("Percentage of vehicles that crashed: ", num_crash/num_vehicles)

# # Visualize the environment
vis = Visualizer()
fig, ax = vis.show_lanes_3d(
    uam_env.corridors.lane_network, 
    uam_env=uam_env, 
    plot_vehicles=True, zoom_in=False,
    show_crash=False)
vis.animate_vehicles(uam_env=uam_env, show_crash=False)
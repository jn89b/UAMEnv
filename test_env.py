from uam_env.envs.uam_env import UAMEnv
from uam_env.visualizer.visualizer import Visualizer
import numpy as np
import matplotlib.pyplot as plt

# Create the environment
uam_env = UAMEnv()
uam_env.reset()
print("uam_env: ", uam_env.corridors)

# Simulate the environment
for i in range(300):
    uam_env.simulate()

vehicles = uam_env.corridors.vehicles
#check if crash
num_vehicles = len(vehicles)
num_crash = 0
for v in vehicles:
    if v.crashed:
        num_crash += 1
print("Percentage of vehicles that crashed: ", num_crash/num_vehicles)


# Visualize the environment
vis = Visualizer()
fig, ax = vis.show_lanes_3d(
    uam_env.corridors.lane_network, uam_env=uam_env, 
    plot_vehicles=True, zoom_in=False)

vis.animate_vehicles(uam_env)


plt.show()
    
    
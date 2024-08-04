from uam_env.envs.uam_env import UAMEnv
from uam_env.visualizer.visualizer import Visualizer
import numpy as np
import matplotlib.pyplot as plt

# Create the environment
uam_env = UAMEnv()
uam_env.reset()
print("uam_env: ", uam_env.corridors)

# Simulate the environment
for i in range(100):
    uam_env.simulate()

# Visualize the environment
vis = Visualizer()
fig, ax = vis.show_lanes_3d(
    uam_env.corridors.lane_network, uam_env=uam_env, 
    plot_vehicles=True)

vis.animate_vehicles(uam_env)


plt.show()
    
    
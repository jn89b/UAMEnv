from uam_env.envs.uam_env import UAMEnv
from uam_env.visualizer.visualizer import Visualizer
import numpy as np
import matplotlib.pyplot as plt
import copy 

# Create the environment
#set the seed
np.random.seed(0)
uam_env = UAMEnv()
uam_env.reset()

# Simulate the environment
N = 300
for n_sim in range(10):
    for i in range(550):
        # Take a random action from the action space (discrete action)
        action = 1
        if i > 300/2:
            action = 7
        obs, reward, terminated, truncated, info = uam_env.step(action)
        if terminated:
            break

vehicles = uam_env.corridors.vehicles
#check if crash
num_vehicles = len(vehicles)
num_crash = 0
crashed_vehicles = []
for v in vehicles:
    if v.agent == False:
        continue
    if v.crashed:
        #compute the distance to the ego vehicle
        distance = np.linalg.norm(v.position - uam_env.vehicle.position)
        print("Crash at distance: ", distance)
        num_crash += 1
        crashed_vehicles.append(v)

# # Visualize the environment
vis = Visualizer()
fig, ax = vis.show_lanes_3d(
    uam_env.corridors.lane_network, 
    uam_env=uam_env, 
    plot_vehicles=True, zoom_in=False,
    show_crash=False)
fig, ax = vis.plot_ego_state(uam_env=uam_env)

fig, ax, anim = vis.animate_vehicles(
    uam_env=uam_env, show_crash=False)

#plot 

plt.legend()
plt.show()
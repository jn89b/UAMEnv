from uam_env.envs.uam_env import UAMEnv
from uam_env.visualizer.visualizer import Visualizer
import numpy as np
import matplotlib.pyplot as plt

# Create the environment
uam_env = UAMEnv()
uam_env.reset()
print("uam_env: ", uam_env.corridors)

vis = Visualizer()
fig, ax = vis.show_lanes_3d(uam_env.corridors.lane_network)

for i in range(100):
    uam_env.simulate()

data_info = []
# 3D plot
# fig, ax = plt.subplots(subplot_kw={'projection': '3d'})


for i, vehicle in enumerate(uam_env.corridors.vehicles):
    data = vehicle.plane.data_handler
    ax.plot(data.x, data.y, data.z, label=f"Vehicle {i}")
    ax.scatter(data.x[0] , data.y[0], data.z[0], color='red')

ax.legend()
plt.show()
    
    
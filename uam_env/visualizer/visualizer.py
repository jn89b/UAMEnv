from typing import Tuple, Dict
import matplotlib.pyplot as plt
import numpy as np
from uam_env.envs.uam_env import UAMEnv

from uam_env.corridor.corridor import StraightLane, LaneNetwork, Corridor
from uam_env.config import lane_config
from matplotlib.animation import FuncAnimation

import matplotlib.pyplot as plt

class Visualizer(object):
    def __init__(self) -> None:

        # this is for the 3D animation
        self.lines = []
        self.scatters = []
        self.min_x = 1000 
        self.max_x = -1000
        self.min_y = 1000
        self.max_y = -1000
    
    def show_lanes_2D(self, lanes:LaneNetwork) -> Tuple[plt.Figure, plt.Axes]:
        fig, ax = plt.subplots()
        for key, straight_lane in lanes.lanes.items():
            if 'vertical' in key:
                continue
            x = [straight_lane.start[0], straight_lane.end[0]]
            y = [straight_lane.start[1], straight_lane.end[1]]
            ax.plot(x, y, label=key, linestyle='--')
            
            # x_lateral = [straight_lane.start_lateral[0], 
            #              straight_lane.end_lateral[0]]
            # y_lateral = [straight_lane.start_lateral[1],
            #                 straight_lane.end_lateral[1]]
            
        ax.legend()
        
        return fig, ax

    def show_lanes_3d(self, lanes:LaneNetwork,
                      uam_env:UAMEnv=None, 
                      plot_vehicles:bool=False) -> Tuple[plt.Figure, plt.Axes]:
        fig, ax = plt.subplots(subplot_kw={'projection': '3d'},
                               figsize=(10, 10))
        for key, straight_lane in lanes.lanes.items():
            x = [straight_lane.start[0], straight_lane.end[0]]
            y = [straight_lane.start[1], straight_lane.end[1]]
            z = [straight_lane.start[2], straight_lane.end[2]]
            ax.plot(x, y, z, label=key + ' centerline', linestyle='--')
        
        if plot_vehicles:
            self.plot_vehicles(uam_env, ax)
        
        ax.legend()
        scale = 8
        # ax.set_xlim(0, lane_config.LANE_LENGTH_M)
        # ax.set_ylim(-lane_config.LANE_LENGTH_M/scale, 
        #             lane_config.LANE_LENGTH_M/scale)
    
        return fig, ax
    
    def plot_vehicles(self, uam_env:UAMEnv, ax=None) -> None:
        min_x = 1000 
        max_x = -1000
        min_y = 1000
        max_y = -1000
        for i, vehicle in enumerate(uam_env.corridors.vehicles):
            data = vehicle.plane.data_handler
            #choose a random color
            color = np.random.rand(3,)
            ax.plot(data.x, data.y, data.z, label=f"Vehicle {i}", color=color)
            ax.scatter(data.x[0] , data.y[0], data.z[0], color=color, marker='o')
            if min(data.x) < min_x:
                min_x = min(data.x)
            if max(data.x) > max_x:
                max_x = max(data.x)
            if min(data.y) < min_y:
                min_y = min(data.y)
            if max(data.y) > max_y:
                max_y = max(data.y)
        
        buffer = 10
        ax.legend()
        ax.set_xlim(min_x-buffer, max_x+buffer)
        ax.set_ylim(min_y-buffer, max_y+buffer)
        return ax
    
    def animate_vehicles(self, uam_env:UAMEnv) -> None:
        self.uam_env = uam_env
        self.fig, self.ax = self.show_lanes_3d(uam_env.corridors.lane_network)
        # Initialize lines and scatter plots for each vehicle
        for i, vehicle in enumerate(uam_env.corridors.vehicles):
            data = vehicle.plane.data_handler
            color = np.random.rand(3,)
            
            # Initialize a line and a scatter plot
            line, = self.ax.plot([], [], [], label=f"Vehicle {i}", color=color)
            scatter = self.ax.scatter([], [], [], color=color, marker='o')
            
            self.lines.append(line)
            self.scatters.append(scatter)

            # Update min/max values for setting plot limits
            if min(data.x) < self.min_x:
                self.min_x = min(data.x)
            if max(data.x) > self.max_x:
                self.max_x = max(data.x)
            if min(data.y) < self.min_y:
                self.min_y = min(data.y)
            if max(data.y) > self.max_y:
                self.max_y = max(data.y)

        # Set the limits of the plot
        self.ax.set_xlim(self.min_x, self.max_x)
        self.ax.set_ylim(self.min_y, self.max_y)
        # self.ax.set_zlim(0, max(max(data.z) for vehicle in uam_env.corridors.vehicles))

        # Create the animation
        anim = FuncAnimation(self.fig, self.update_plot, frames=len(data.x), interval=10, blit=True)

        plt.legend()
        plt.show()

    def update_plot(self, frame):
        for i, line in enumerate(self.lines):
            data = self.uam_env.corridors.vehicles[i].plane.data_handler
            
            # Update the data of the line and scatter
            line.set_data(data.x[:frame], data.y[:frame])
            line.set_3d_properties(data.z[:frame])
            self.scatters[i]._offsets3d = (data.x[frame:frame+1], data.y[frame:frame+1], data.z[frame:frame+1])

        return self.lines + self.scatters
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
                      plot_vehicles:bool=False,
                      zoom_in:bool=True,
                      show_crash:bool=False) -> Tuple[plt.Figure, plt.Axes]:
        
        fig, ax = plt.subplots(subplot_kw={'projection': '3d'},
                               figsize=(12, 12))
        
        for key, straight_lane in lanes.lanes.items():
            lane_color = lane_config.LANE_COLORS[key]
            x = [straight_lane.start[0], straight_lane.end[0]]
            y = [straight_lane.start[1], straight_lane.end[1]]
            z = [straight_lane.start[2], straight_lane.end[2]]
            ax.plot(x, y, z, linestyle='--',
                    color='black')
        
        #def the draw lanes as a cylinder
        for key, straight_lane in lanes.lanes.items():
            lane: StraightLane = straight_lane
            theta = lane.heading_rad
            lane_color = lane_config.LANE_COLORS[key]
            x_position = lane.start[0]
            y_position = lane.start[1]
            z_position = lane.start[2]
            #direction vector components (cosine and sine of the angle)
            dx = np.cos(theta)
            dy = np.sin(theta)
            
            NUM_POINTS = 50
            #create the cylinder's sides
            z = np.linspace(0, lane_config.LANE_LENGTH_M, NUM_POINTS)
            
            theta_cylinder = np.linspace(0, 2 * np.pi, NUM_POINTS)
            theta_grid, z_grid = np.meshgrid(theta_cylinder, z)
            
            x_grid = lane_config.LANE_WIDTH_M/2 * np.cos(theta_grid)
            y_grid = lane_config.LANE_HEIGHT_M/2 * np.sin(theta_grid)
            
            #rotate the cylinder to align with the heading angle
            x_rotated = x_position + dx * z_grid - dy * x_grid
            y_rotated = y_position + dy * z_grid + dx * x_grid
            z_rotated = z_position + y_grid
            
            #plot the transparent surface of the cylinder
            ax.plot_surface(x_rotated, y_rotated, z_rotated, 
                            color=lane_color, alpha=0.3, rstride=5, cstride=5,
                            label=key)
            ax.plot_wireframe(x_rotated, y_rotated, z_rotated, 
                              color=lane_color, rstride=5, cstride=5,
                              alpha=0.3)
            
            
        if plot_vehicles:
            self.plot_vehicles(uam_env=uam_env, 
                               ax=ax, 
                               show_crash=show_crash)
        
        ax.legend()
        scale = 8
        # ax.set_xlim(0, lane_config.LANE_LENGTH_M)
        # ax.set_ylim(-lane_config.LANE_LENGTH_M/scale, 
        #             lane_config.LANE_LENGTH_M/scale)
    
        return fig, ax
    
    def plot_vehicles(self, uam_env:UAMEnv, ax=None,
                      zoom_in:bool=True,
                      show_crash:bool=False) -> None:
        min_x = 1000 
        max_x = -1000
        min_y = 1000
        max_y = -1000
        for i, vehicle in enumerate(uam_env.corridors.vehicles):
            data = vehicle.plane.data_handler
            if show_crash and vehicle.crashed == False:
                continue 
            #choose a random color
            color = np.random.rand(3,)
            ax.plot(data.x, data.y, data.z, label=f"Vehicle {i}", color=color,
                    linewidth=3)
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
        if zoom_in:
            ax.set_xlim(min_x-buffer, max_x+buffer)
            ax.set_ylim(min_y-buffer, max_y+buffer)
        
        return ax
    
    def animate_vehicles(self, uam_env:UAMEnv,
                         show_crash:bool=False) -> None:
        self.uam_env = uam_env
        self.fig, self.ax = self.show_lanes_3d(uam_env.corridors.lane_network)
        # Initialize lines and scatter plots for each vehicle
        for i, vehicle in enumerate(uam_env.corridors.vehicles):
            if show_crash and vehicle.crashed == False:
                continue 
            data = vehicle.plane.data_handler
            # color = np.random.rand(3,)
            if vehicle.agent:
                color = 'blue'
            else:
                color = 'red'
            
            # Initialize a line and a scatter plot
            line, = self.ax.plot([], [], [], label=f"Vehicle {i}", color=color,
                                 linewidth=3)
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
        # self.ax.set_xlim(self.min_x, self.max_x)
        # self.ax.set_ylim(self.min_y, self.max_y)
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
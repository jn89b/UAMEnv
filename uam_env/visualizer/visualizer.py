from typing import Tuple, Dict
import matplotlib.pyplot as plt
import numpy as np
from uam_env.corridor.corridor import StraightLane, LaneNetwork, Corridor
from uam_env.config import lane_config


import matplotlib.pyplot as plt

class Visualizer(object):
    def __init__(self) -> None:
        pass
    
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

    def show_lanes_3d(self, lanes:LaneNetwork) -> Tuple[plt.Figure, plt.Axes]:
        fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
        for key, straight_lane in lanes.lanes.items():
            x = [straight_lane.start[0], straight_lane.end[0]]
            y = [straight_lane.start[1], straight_lane.end[1]]
            z = [straight_lane.start[2], straight_lane.end[2]]
            ax.plot(x, y, z, label=key + 'centerline', linestyle='--')
        ax.legend()
        ax.set_xlim(0, lane_config.LANE_LENGTH_M)
        ax.set_ylim(0, lane_config.LANE_LENGTH_M/4)
        return fig, ax
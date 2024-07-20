import logging
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple
from uam_env.utils import Vector
from uam_env.config import lane_config
from uam_env.vehicle.kinematics import Kinematics
from uam_env.vehicle.objects import CorridorObject
import numpy as np

"""
A corridor consists of a set of lanes that the agent can navigate through
- To keep it simple a corridor will have the following lanes:
    - A lateral lane
    - A vertical lane
    - A lateral passing lane
    - A vertical passing lane

The agent will make a decision on what lane to navigate through 
based on the current state of the environment
"""

class StraightLane(object):
    """
    A straight lane in the corridor
    TODO: Add vertical and lateral boundaries to the lane from the centerline 
    """
    def __init__(self,
                 start:Vector,
                 end:Vector,
                 width_m:float = lane_config.LANE_WIDTH_M,
                 height_m:float = lane_config.LANE_HEIGHT_M) -> None:
        self.start = start
        self.end = end
        self.width_m = width_m
        self.height_m = height_m
        self.heading_rad = np.arctan2(self.start[1] - self.end[1], 
                                  self.start[0] - self.end[0])
        self.init_boundaries()
        
    def init_boundaries(self) -> Dict[str, Tuple[Vector, Vector]]:
        """
        Initialize the boundaries of the lane
        """
        lateral_boundary = self.create_lateral_boundary()
        vertical_boundary = self.create_vertical_boundary()
        
        return {
            'lateral_boundary': lateral_boundary,
            'vertical_boundary': vertical_boundary
        }
        
    def create_lateral_boundary(self) -> Tuple[Vector, Vector]:
        """
        Create the lateral boundary of the lane
        """
        # get the normal vector
        normal_vector = np.array([-np.sin(self.heading_rad), 
                                  np.cos(self.heading_rad) , 
                                  0])
        # get the lateral boundary points
        start_lateral = self.start + normal_vector * self.width_m
        end_lateral = self.end + normal_vector * self.width_m
        
        return start_lateral, end_lateral
    
    def create_vertical_boundary(self) -> Tuple[Vector, Vector]:
        """
        Create the vertical boundary of the lane
        """
        # get the normal vector
        normal_vector = np.array([np.cos(self.heading_rad), 
                                  np.sin(self.heading_rad),
                                  0])
        # get the vertical boundary points
        start_vertical = self.start + normal_vector * self.height_m
        end_vertical = self.end + normal_vector * self.height_m
        
        return start_vertical, end_vertical
        
class Lanes(object):
    """
    For now lets keep it simple and make 4 lanes
    """
    def __init__(self) -> None:
        self.lanes = {
            'lateral': None,
            'lateral_passing': None,
            'vertical_passing': None
        }
    
    def get_lane_names(self) -> List[str]:
        return list(self.lanes.keys())
    
    def create_straight_lane(self,
                             start:Vector,
                             end:Vector,
                             heading_dg:float) -> StraightLane:
        
        straight_lane = StraightLane(
            start = start,
            end = end,
        )
        
        return straight_lane
    
    def init_lane_points(self, 
        start_vector_m:Vector=np.array([0, 0, 50]),
        heading_dg:float=0,
        is_right_lateral:bool=True,
        width_lane_m:float = lane_config.LANE_WIDTH_M,
        height_lane_m:float = lane_config.LANE_HEIGHT_M) -> Dict[str, Vector]:
        """
        Generates a set of points for the corridor to be created 
        looks something like this 
        
        If you set is_right to True, the corridor will be created to the right
        where the origin point will be on the bottom left of the corridor
        The pivot point will be on the opposite side of the origin point 
        based on how the user defines the heading_dg
        ^ z
        |
        --> x/y
        
                    Vertical Passing
                /                   \
            /                           \
        /                                   \
        origin ---------------------lateral passing
    <-----Width Lane------>     <-----Width Lane------> 
        
        """
        heading_rad = np.deg2rad(heading_dg)
        norm_heading = np.array([np.cos(heading_rad), np.sin(heading_rad)])
        #get the lateral passing point
        
        if is_right_lateral:
            # passing_x = start_vector_m[0] + (width_lane_m * np.cos(heading_rad))
            # passing_y = start_vector_m[1] + (width_lane_m * np.sin(heading_rad))
            # get normal vector to the right
            normal_vector = np.array([-norm_heading[1], norm_heading[0]])
            passing_x = start_vector_m[0] + (width_lane_m * normal_vector[0])
            passing_y = start_vector_m[1] + (width_lane_m * normal_vector[1])
        else:
            normal_vector = np.array([norm_heading[1], -norm_heading[0]])
            passing_x = start_vector_m[0] - (width_lane_m * normal_vector[0])
            passing_y = start_vector_m[1] - (width_lane_m * normal_vector[1])    
        passing_z = start_vector_m[2]
        lateral_passing_point: Vector = np.array([passing_x, passing_y, passing_z])

        #get the midpoint between the origin and the lateral passing point
        vertical_passing_x = (start_vector_m[0] + lateral_passing_point[0]) / 2
        vertical_passing_y = (start_vector_m[1] + lateral_passing_point[1]) / 2
        vertical_passing_z = start_vector_m[2] + height_lane_m 
        
        vertical_passing: Vector = np.array([vertical_passing_x, 
                                             vertical_passing_y, 
                                             vertical_passing_z])
        
        return {
            'lateral': start_vector_m,
            'lateral_passing': lateral_passing_point,
            'vertical_passing': vertical_passing
        }
        
    
    def straight_lanes(
        self, 
        num_lanes:int = 4,
        start_vector_m:Vector=np.array([0, 0, 50]),
        length_m:float = 1000,
        heading_dg:float = 0) -> Tuple:
        """
        Create straight lanes in the corridor
        Each line generated will be 
        """
        heading_rad = np.deg2rad(heading_dg)
        corridor_points = self.init_lane_points(start_vector_m, heading_dg)
        for zone_name, v in self.lanes.items():
            #create a straight lane and add it to the lanes
            origin = corridor_points[zone_name]
            end_x = origin[0] + length_m * np.cos(heading_rad)
            end_y = origin[1] + length_m * np.sin(heading_rad)
            end:Vector = np.array([end_x, end_y, origin[2]])
            straight_lane = self.create_straight_lane(origin, end, heading_dg)
            self.lanes[zone_name] = straight_lane
    
class Corridor(object):
    """
    A corridor is a set of lanes that the agent can navigate through
    TODO: Add more objects to the corridor?
    
    Keep track of which vehicles are in the corridor
    
    """
    def __init__(self,
                 lanes:Lanes = None,
                 vehicles:Kinematics = None,
                 corridor_objects:List[CorridorObject] = None,
                 np_random: np.random.RandomState = None,
                 record_history:bool = None) -> None:
        if lanes == None:
            self.lanes = Lanes()
            self.lanes.straight_lanes()
        else:
            self.lanes = lanes
            
        self.vehicles = vehicles or []
        self.objects = corridor_objects or []
        self.np_random = np_random if np_random else np.random.RandomState()
        self.record_history = record_history
    
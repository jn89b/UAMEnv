import logging
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple
from uam_env.utils import Vector
from uam_env.config import lane_config, kinematics_config
from uam_env.vehicle.kinematics import Vehicle
# from uam_env.vehicle.behavior import DiscreteVehicle
from uam_env.vehicle.objects import CorridorObject
import numpy as np

if TYPE_CHECKING:
    from uam_env.corridor.corridor import Corridor
    from uam_env.vehicle.behavior import DiscreteVehicle
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
    speed_limit = lane_config.SPEED_LIMIT_MS
    def __init__(self,
                 start:Vector,
                 end:Vector,
                 lane_name = None,
                 width_m:float = lane_config.LANE_WIDTH_M,
                 height_m:float = lane_config.LANE_HEIGHT_M) -> None:
        self.start = start
        self.end = end
        self.lane_name = lane_name
        self.width_m = width_m
        self.height_m = height_m
        self.length_m = np.linalg.norm(self.end - self.start)
        self.direction = (self.end - self.start) / self.length_m
        self.direction_lateral = np.array([-self.direction[1], 
                                           self.direction[0], 
                                           0])
        
        self.heading_rad = np.arctan2(self.end[1] - self.start[1], 
                                  self.end[0] - self.start[0])
        self.init_boundaries()
        
    def on_lane(self,
                position: Vector,
                longitudinal: float = None,
                lateral: float = None,
                margin: float = 0,
                vehicle_length_m: float = kinematics_config.LENGTH_m) -> bool:
        """
        Whether a given world position is on the lane.

        :param position: a world position [m]
        :param longitudinal: (optional) the corresponding longitudinal lane coordinate, if known [m]
        :param lateral: (optional) the corresponding lateral lane coordinate, if known [m]
        :param margin: (optional) a supplementary margin around the lane width
        :param vehicle_length_m: (optional) the length of the vehicle [m]
        :return: is the position on the lane?
        """
        # If longitudinal and lateral coordinates are not provided, calculate them
        if longitudinal is None or lateral is None:
            longitudinal, lateral = self.local_coordinates(position)

        # Check if the vehicle's lateral position is within the lane's width plus margin
        if np.abs(lateral) > self.width_m / 2 + margin:
            return False
        
        # Alternatively, check if the vehicle's lateral position plus half its length is within the lane's width
        if np.abs(lateral) > self.width_m / 2 + vehicle_length_m / 2:
            return False
        
        # If neither condition is violated, return True
        return True

    def position(self, longitudinal:float, lateral:float) -> Vector:
        """
        Get the position of the vehicle in the lane
        """
        return (self.start 
                + longitudinal * self.direction 
                + lateral * self.direction_lateral)
        
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
    
    def local_coordinates(self, position:Vector) -> Tuple[float, float]:
        """
        Get the local coordinates of the vehicle in the lane
        Returns the longitudinal and lateral coordinates of the vehicle
        """
        delta = position - self.start
        longitudinal = np.dot(delta, self.direction)
        lateral = np.dot(delta, self.direction_lateral)
        
        return longitudinal, lateral
    
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
        
    def heading_at(self, longitudinal:float=None) -> float:
        """
        Get the heading of the lane at a given longitudinal position
        """
        return self.heading_rad
        
class LaneNetwork(object):
    """
    For now lets keep it simple and make 4 lanes
    """
    def __init__(self) -> None:
        self.lanes = {
            lane_config.LANE_LATERAL_KEY: None,
            lane_config.LANE_LATERAL_PASSING_KEY: None,
            lane_config.LANE_VERTICAL_PASSING_KEY: None
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
            lane_config.LANE_LATERAL_KEY: start_vector_m,
            lane_config.LANE_LATERAL_PASSING_KEY: lateral_passing_point,
            lane_config.LANE_VERTICAL_PASSING_KEY: vertical_passing
        }
        
    
    def closest_position_on_lane(self,
                                    position:Vector,
                                    heading_dg:Optional[float] = None,
                                    ) -> Tuple[str, Vector]:
        """
        Get the closest position on the lane
        :param position: The position of the agent
        :param heading_dg: The heading of the agent
        :return: The name of the lane and the position on the lane
        """
        closest_points = []
        distances = []
        for zone_name, straight_lane in self.lanes.items():
            longitudinal, lateral = straight_lane.local_coordinates(position)
            closest_position = straight_lane.position(longitudinal, lateral)
            closest_points.append(closest_position)
            distances.append(np.linalg.norm(closest_position - position))
        
        min_index = np.argmin(distances)
        return list(self.lanes.keys())[min_index], closest_points[min_index]

    def closest_point_on_lane(self,
                              origin:Vector,
                              direction:Vector,
                              position:Vector) -> Vector:
        
        t = np.dot(position - origin, direction) / np.dot(direction, direction)
        t = max(t, 0)
        Q = origin + t * direction
        return Q

    def get_closest_lane(self, 
                         position:Vector,
                         heading_dg: Optional[float] = None,
                         ) -> Tuple[str, StraightLane]:
        """
        Get the closest lane to the agent 
        :param position: The position of the agent
        :param heading_dg: The heading of the agent
        :return: The name of the lane and the lane object
        """
        # close_lane, min_distance = self.closest_position_on_lane(
        #     position, heading_dg)
        
        closest_points = []
        distances = []
        for zone_name, lane in self.lanes.items():
            start = lane.start
            end = lane.end
            point = self.closest_point_on_lane(
                start, end, position)
            closest_points.append(point)
            distances.append(np.linalg.norm(point - position))
            
        close_lane = list(self.lanes.keys())[np.argmin(distances)]
    
        return close_lane, self.lanes[close_lane]
            
    def straight_lanes(
        self, 
        num_lanes:int = 4,
        start_vector_m:Vector=np.array([0, 0, 50]),
        length_m:float = lane_config.LANE_LENGTH_M,
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
    This is like a board game where the 
    agent can move through the lanes keeps track of the state of 
    the environment
    """
    def __init__(self,
                 lane_network:LaneNetwork = None,
                 vehicles:List[Vehicle] = None,
                 corridor_objects:List[CorridorObject] = None,
                 np_random: np.random.RandomState = None,
                 record_history:bool = None) -> None:
        
        if lane_network == None:
            self.lane_network = LaneNetwork()
            self.lane_network.straight_lanes()
        else:
            self.lane_network = lane_network
        self.vehicles = vehicles or []
        self.objects = corridor_objects or []
        self.np_random = np_random if np_random else np.random.RandomState()
        self.record_history = record_history
    
    #TODO:check typing for vehicles
    def act(self, action:int=None) -> None:
        """
        Act on the corridor
        """
        for vehicle in self.vehicles:
            if vehicle.agent == False:
                vehicle.act()
            #TODO: need to consider how I want to change the action 
            #this will set the discrete action for the vehicle
            else:
                vehicle : DiscreteVehicle 
                vehicle.act(action)
            
    def step(self, dt:float) -> None:
        """
        Step the dynamics of each entity in the corridor
        """
        for vehicle in self.vehicles:
            vehicle.step(dt)
        
        #TODO: handle collisions between vehicles
        for i, vehicle in enumerate(self.vehicles):
            for other in self.vehicles[i + 1:]:
                vehicle.handle_collisions(other, dt)
            for other in self.objects:
                vehicle.handle_collisions(other, dt)
            

    def neighbor_vehicles(self, ego_vehicle:"Vehicle",
                          lane_index:str) -> Tuple[Optional["Vehicle"], Optional["Vehicle"]]:
        """
        Find the preceding and following vehicles of a given vehicle.

        :param vehicle: the vehicle whose neighbours must be found
        :param lane_index: the lane on which to look for preceding and following vehicles.
                     It doesn't have to be the current vehicle lane but can also be another lane, 
                     in which case the vehicle is projected on it 
                     considering its local coordinates in the lane.
        :return: its preceding vehicle, its following vehicle
        """
        lane_index = lane_index or ego_vehicle.lane_index
        if not lane_index:
            return None, None
        lane: StraightLane = self.lane_network.lanes[lane_index]
        s,_ = lane.local_coordinates(ego_vehicle.position)
        s_front = s_rear = None # front and rear vehicle distances
        v_front = v_rear = None # speed of the front and rear vehicle
        
        for v in self.vehicles + self.objects:
            if v is not ego_vehicle: #and not isinstance(v, CorridorObject):
                s_v, lat_v = lane.local_coordinates(v.position)
                if not lane.on_lane(v.position, s_v, lat_v, margin=1):
                    continue
                if s <= s_v and (s_front is None or s_v <= s_front):
                    s_front = s_v
                    v_front = v
                if s_v < s and (s_rear is None or s_v > s_rear):
                    s_rear = s_v
                    v_rear = v
        return v_front, v_rear
        
    def __repr__(self):
        return self.vehicles.__repr__()

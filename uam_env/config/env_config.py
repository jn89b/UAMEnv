"""
These are parameters you can set to change the behavior of the environment
"""
NON_CONTROLLED_VEHICLES = 3
CONTROLLED_VEHICLES = 1
RECORD_HISTORY = True
DURATION = 30
NUM_ACTIONS = 9
DT = 0.1
MAX_NUM_STEPS = 350 #Dt times the number of steps = Total time
DISTANCE_THRESHOLD = 250

##### LANE CONFIG  MAPPING  #####

#TODO: Refactor this to use the lane_config fileS
LANE_LATERAL_KEY = "lateral"
LANE_LATERAL_PASSING_KEY = "lateral_passing"
LANE_VERTICAL_PASSING_KEY = "vertical_passing"

LANE_INDEX_MAPPING = {
    -1: LANE_LATERAL_KEY,
     0: LANE_LATERAL_PASSING_KEY,
     1: LANE_VERTICAL_PASSING_KEY
}

"""
The discrete action mapping is as follows
- Discrete action space:
    
    - 0: lateral lane slow down 
    - 1: lateral lane same speed
    - 2: lateral lane speed up
    
    - 3: lateral passing slow down
    - 4: lateral passing same speed
    - 5: lateral passing speed up
    
    - 6: vertical lane slow down
    - 7: vertical lane same speed
    - 8: vertical lane speed up 

Returns a tuple of (int, int) where the 
first element is the lane index and 
the second element is the acceleration that is
Key: Int = Tuple(lane index, acceleration)
- Feeding this to a PID controller to go to the lane
"""

DISCRETE_ACTION_MAPPING ={
    0: (-1, -1),
    1: (-1, 0),
    2: (-1, 1),
    3: (0, -1),
    4: (0, 0),
    5: (0, 1),
    6: (1, -1),
    7: (1, 0),
    8: (1, 1)
}
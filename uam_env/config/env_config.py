"""
These are parameters you can set to change the behavior of the environment
"""
NON_CONTROLLED_VEHICLES = 2
CONTROLLED_VEHICLES = 1
RECORD_HISTORY = True
DURATION = 30
NUM_ACTIONS = 9
DT = 0.1
MAX_NUM_STEPS = 200 


##### LANE CONFIG  MAPPING  #####
"""
The discrete action mapping is as follows
- Discrete action space:
    - 0: lateral lane speed up 
    - 1: lateral lane slow down
    - 2: lateral lane keep speed
    - 3: lateral passing speed up
    - 4: lateral passing slow down
    - 5: lateral passing keep speed
    - 6: vertical lane speed up
    - 7: vertical lane slow down
    - 8: vertical lane keep speed 

Returns a tuple of (int, int) where the 
first element is the lane index and 
the second element is the acceleration
- Feeding this to a PID controller to go to the lane
"""

#TODO: Refactor this to use the lane_config fileS
LANE_LATERAL_KEY = "lateral"
LANE_LATERAL_PASSING_KEY = "lateral_passing"
LANE_VERTICAL_PASSING_KEY = "vertical_passing"


LANE_INDEX_MAPPING = {
    -1: LANE_LATERAL_KEY,
     0: LANE_LATERAL_PASSING_KEY,
     1: LANE_VERTICAL_PASSING_KEY
}

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
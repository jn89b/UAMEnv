import copy 
from typing import List, Optional, Tuple, Union

import numpy as np

from uam_env.corridor.corridor import Lane, Corridor
from uam_env.utils import Vector
from uam_env.vehicle.kinematics import Vehicle

class ControlledVehicle(Vehicle):
    """
    A vehicle that utilizes a high-level controller to make decisions
    """
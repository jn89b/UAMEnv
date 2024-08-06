import numpy as np

LENGTH_m = 0.5 #Length of the vehicle in meters
WIDTH_m = 0.5  #width of the vehicle in meters 

MAX_SPEED_MS = 25 #Maximum speed of the vehicle in m/s
MIN_SPEED_MS = 12 #Minimum speed of the vehicle in m/s

HISTORY_SIZE = 30
BUFFER_SPACING_M = 25

AGGRESSIVE = 0 #From 0 to 1, where 0 is the most conservative and 1 is the most aggressive
ACC_MAX = 10 #Maximum acceleration in m/s^2

COMFORT_ACC_MAX = 3 #Maximum comfortable acceleration in m/s^2
COMFORT_ACC_MIN = -5 #Minimum comfortable acceleration in m/s^2

TIME_WANTED = 1.5 #Desired time gap to the vehicle in front in seconds


#TODO: Need to move the controller constraints here
state_constraints = {
    'x_min': -2000, #-np.inf,
    'x_max': 2000, #np.inf,
    'y_min': -2000, #-np.inf,
    'y_max': 2000, #np.inf,
    'z_min': 30,
    'z_max': 75,
    'phi_min':  -np.deg2rad(45),
    'phi_max':   np.deg2rad(45),
    'theta_min':-np.deg2rad(20),
    'theta_max': np.deg2rad(20),
    'psi_min':  -np.pi,
    'psi_max':   np.pi,
    'airspeed_min': MIN_SPEED_MS,
    'airspeed_max': MAX_SPEED_MS
}


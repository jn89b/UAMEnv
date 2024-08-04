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
import numpy as np

TAU_ACC = 0.6  # [s]
TAU_HEADING = 0.2  # [s]
TAU_LATERAL = 0.6  # [s]

TAU_PURSUIT = 0.5 * TAU_HEADING  # [s]
KP_LATERAL = 1.0  # [1/s]

KP_HEADING = 1.1
KD_HEADING = 0.4
ROLL_MAX = np.deg2rad(45)  # [rad]
ROLL_MIN = -ROLL_MAX  # [rad]
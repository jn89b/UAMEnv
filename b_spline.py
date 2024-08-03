import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline

# Define control points
control_points = np.array([
    [0, 0],    # P0
    [10, 0],   # P1
    [20, 2],   # P2
    [30, 4],   # P3
    [40, 4],   # P4
    [50, 4]    # P5
])

# Degree of the B-Spline
degree = 3

# Number of control points
n_control_points = len(control_points)

# Uniform knot vector (open)
knot_vector = np.concatenate(([0] * degree, np.arange(1, n_control_points - degree + 1), [n_control_points - degree] * degree))

# Create the B-Spline
b_spline = BSpline(knot_vector, control_points, degree)

# Generate B-Spline points
t_values = np.linspace(0, n_control_points - degree, 100)
curve = b_spline(t_values)

# Plot the curve and control points
plt.plot(curve[:, 0], curve[:, 1], label='B-Spline Curve')
plt.scatter(control_points[:, 0], control_points[:, 1], color='red', label='Control Points')
plt.title('Lane Change B-Spline Curve')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.legend()
plt.grid(True)
plt.show()

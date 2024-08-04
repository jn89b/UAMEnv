import numpy as np

def check_moving_spheres_collision(C1, v1, r1, C2, v2, r2):
    R0 = np.array(C2) - np.array(C1)
    v_r = np.array(v2) - np.array(v1)
    
    a = np.dot(v_r, v_r)
    b = 2 * np.dot(R0, v_r)
    c = np.dot(R0, R0) - (r1 + r2)**2
    
    discriminant = b**2 - 4*a*c
    
    if discriminant < 0:
        return False, None  # No collision
    
    t1 = (-b - np.sqrt(discriminant)) / (2*a)
    t2 = (-b + np.sqrt(discriminant)) / (2*a)
    
    # We want the smallest positive time (if any)
    collision_time = min(t for t in [t1, t2] if t >= 0)
    
    return True, collision_time

# Example usage:
C1 = [0, 0, 0]
v1 = [1, 1, 0]
r1 = 1
C2 = [5, 5, 0]
v2 = [-1, -1, 0]
r2 = 1

collision, time_of_collision = check_moving_spheres_collision(C1, v1, r1, C2, v2, r2)
if collision:
    print(f"Collision will occur at t = {time_of_collision:.2f}")
else:
    print("No collision will occur")

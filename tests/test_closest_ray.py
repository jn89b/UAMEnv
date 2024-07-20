import numpy as np

def closest_point_on_ray(O, D, P):
    """
    Calculate the closest point on a ray to a given position.

    Parameters:
    O (np.ndarray): Origin of the ray.
    D (np.ndarray): Direction vector of the ray.
    P (np.ndarray): The position to which the closest point on the ray is to be found.

    Returns:
    np.ndarray: The closest point on the ray to the given position.
    """
    t = np.dot(P - O, D) / np.dot(D, D)
    t = max(t, 0)  # Ensure the point is on the ray
    Q = O + t * D
    return Q

def find_closest_ray(rays, position):
    """
    Find the ray that is closest to a given position from a list of rays.

    Parameters:
    rays (list of tuples): A list of rays, where each ray is represented as a tuple (O, D).
                           O (np.ndarray): Origin of the ray.
                           D (np.ndarray): Direction vector of the ray.
    position (np.ndarray): The position to which the closest ray is to be found.

    Returns:
    tuple: A tuple containing the closest ray and the closest point on that ray to the given position.
           - Closest ray (tuple): The ray (O, D) that is closest to the given position.
           - Closest point (np.ndarray): The closest point on the closest ray to the given position.
    """
    closest_points = []
    distances = []
    for O, D in rays:
        print("O and D", O, D)
        Q = closest_point_on_ray(O, D, position)
        closest_points.append(Q)
        distances.append(np.linalg.norm(Q - position))
    
    min_index = np.argmin(distances)
    return rays[min_index], closest_points[min_index]

# Example usage:
rays = [
    (np.array([0, 0, 0]), np.array([1, 1, 0])),  # Ray 1: Origin (0,0,0), Direction (1,1,0)
    (np.array([1, 0, 0]), np.array([0, 1, 1])),  # Ray 2: Origin (1,0,0), Direction (0,1,1)
    (np.array([0, 1, 0]), np.array([1, 0, 1]))   # Ray 3: Origin (0,1,0), Direction (1,0,1)
]

position = np.array([1, 2, 3])  # Given position

closest_ray, closest_point = find_closest_ray(rays, position)
print("Closest Ray:", closest_ray)
print("Closest Point on the Closest Ray:", closest_point)

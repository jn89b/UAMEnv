import copy
import importlib
import itertools
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np


# Useful types
Vector = Union[np.ndarray, Sequence[float]]
Matrix = Union[np.ndarray, Sequence[Sequence[float]]]
Interval = Union[
    np.ndarray,
    Tuple[Vector, Vector],
    Tuple[Matrix, Matrix],
    Tuple[float, float],
    List[Vector],
    List[Matrix],
    List[float],
]

def wrap_to_pi(x: float) -> float:
    return ((x + np.pi) % (2 * np.pi)) - np.pi

def not_zero(x: float, eps: float = 1e-2) -> float:
    if abs(x) > eps:
        return x
    elif x >= 0:
        return eps
    else:
        return -eps

# #test the Vector type
# v: Vector = np.array([1, 2, 3])
# print(v)

def are_polygons_intersecting(
    a: Vector, b: Vector, displacement_a: Vector, displacement_b: Vector
) -> Tuple[bool, bool, Optional[np.ndarray]]:
    """
    Checks if the two polygons are intersecting.

    See https://www.codeproject.com/Articles/15573/2D-Polygon-Collision-Detection

    :param a: polygon A, as a list of [x, y] points
    :param b: polygon B, as a list of [x, y] points
    :param displacement_a: velocity of the polygon A
    :param displacement_b: velocity of the polygon B
    :return: are intersecting, will intersect, translation vector
    """
    intersecting = will_intersect = True
    min_distance = np.inf
    translation, translation_axis = None, None
    for polygon in [a, b]:
        for p1, p2 in zip(polygon, polygon[1:]):
            normal = np.array([-p2[1] + p1[1], p2[0] - p1[0]])
            normal /= np.linalg.norm(normal)
            min_a, max_a = project_polygon(a, normal)
            min_b, max_b = project_polygon(b, normal)

            if interval_distance(min_a, max_a, min_b, max_b) > 0:
                intersecting = False

            velocity_projection = normal.dot(displacement_a - displacement_b)
            if velocity_projection < 0:
                min_a += velocity_projection
            else:
                max_a += velocity_projection

            distance = interval_distance(min_a, max_a, min_b, max_b)
            if distance > 0:
                will_intersect = False
            if not intersecting and not will_intersect:
                break
            if abs(distance) < min_distance:
                min_distance = abs(distance)
                d = a[:-1].mean(axis=0) - b[:-1].mean(axis=0)  # center difference
                translation_axis = normal if d.dot(normal) > 0 else -normal

    if will_intersect:
        translation = min_distance * translation_axis
    return intersecting, will_intersect, translation


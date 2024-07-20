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

# #test the Vector type
# v: Vector = np.array([1, 2, 3])
# print(v)

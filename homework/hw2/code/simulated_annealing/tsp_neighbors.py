import numpy as np
from typing import List


def valid_pairs(p1: int, p2: int) -> bool:
    """
    Makes sure that p1 and p2 are not adjacent
    """
    return not (p1 - 1 < p2 < p1 + 1)


def two_opt_swap(current_path: List[int], i1: int, i2: int) -> List[int]:
    """
    Performs the 2-opt swap
    """
    old_path = current_path.copy()
    new_path = old_path[:i1 + 1] + old_path[i2:i1:-1] + old_path[i2 + 1:]
    return new_path


def generate_2opt_neighbor(current_soln: List[int]) -> List[int]:
    i1, i2 = np.random.randint(len(current_soln), size=2)
    while not valid_pairs(i1, i2):
        i1, i2 = np.random.randint(len(current_soln), size=2)
    i1, i2 = sorted([i1, i2])
    return two_opt_swap(current_soln, i1, i2)


def two_swap_swap(current_path: List[int], i1: int, i2: int) -> List[int]:
    """
    Performs the 2-swap swap
    """
    path = current_path.copy()
    path[i1], path[i2] = path[i2], path[i1]
    return path


def generate_2swap_neighbor(current_soln: List[int]) -> List[int]:
    i1, i2 = np.random.randint(len(current_soln), size=2)
    while i1 == i2:
        i1, i2 = np.random.randint(len(current_soln), size=2)
    return two_swap_swap(current_soln, i1, i2)

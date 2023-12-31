from typing import List, Callable

import numpy.random as rd
from more_itertools import distinct_combinations

from tsp_algorithms import TSPResult, logger

TWO_SWAP_DEFAULT_N_ITERS = 100_000


def generate_best_neighbor(current_path: List[int],
                           path_distance: Callable[[List[int]], float]) -> List[int]:
    """
    Generates all neighbors by doing every possible swap and returns the one with the shortest path distance
    """
    combs = list(distinct_combinations([x for x in range(len(current_path))], 2))

    # Do the first valid swap
    i1, i2 = combs[0]
    current_cities = current_path.copy()
    current_distance = path_distance(current_cities)
    current_cities[i1], current_cities[i2] = current_cities[i2], current_cities[i1]

    # Do the rest of the swaps
    for i1, i2 in combs[1:]:
        new_cities = current_cities.copy()
        new_cities[i1], new_cities[i2] = new_cities[i2], new_cities[i1]
        new_distance = path_distance(new_cities)

        # Compares the new distance with existing one
        if new_distance < current_distance:
            current_cities = new_cities
            current_distance = new_distance

    return current_cities


def two_swap_tsp(cities: List[int],
                 path_distance: Callable[[List[int]], float],
                 n_iters: int = TWO_SWAP_DEFAULT_N_ITERS) -> TSPResult:
    """
    Exhaustive 2-swap algorithm
    """
    current_path = cities.copy()
    current_dist = path_distance(current_path)
    progress = [current_dist]

    # Repeating for N iterations
    for iters in range(n_iters):
        # logger(iters)

        # Generates the best neighbor and compares with current best then picks one with lower path distance
        best_neighbor = generate_best_neighbor(current_path, path_distance)
        current_path = min(best_neighbor, current_path, key=path_distance)
        current_dist = path_distance(current_path)

        progress.append(current_dist)

    return TSPResult(
        final_route=current_path + [current_path[0]],
        final_distance=current_dist,
        progress=progress
    )


def rdn_two_swap_tsp(cities: List[int],
                     path_distance: Callable[[List[int]], float],
                     n_iters: int = TWO_SWAP_DEFAULT_N_ITERS) -> TSPResult:
    """
    Randomized 2-swap algorithm
    """
    current_path = cities.copy()
    current_dist = path_distance(current_path)
    progress = [current_dist]

    # Repeating for N iterations
    for iters in range(n_iters):
        # logger(iters)

        # Picks a pair of cities that are not the same
        assert len(current_path) >= 2
        i1, i2 = rd.randint(len(current_path), size=(2,))
        while i1 == i2:
            i1, i2 = rd.randint(len(current_path), size=(2,))

        # Performs the swap and computes new distance
        new_path = current_path.copy()
        new_path[i1], new_path[i2] = new_path[i2], new_path[i1]
        new_dist = path_distance(new_path)

        # Updates the solution accordingly
        if new_dist < current_dist:
            current_path = new_path
            current_dist = new_dist

        progress.append(current_dist)

    return TSPResult(
        final_route=current_path + [current_path[0]],
        final_distance=current_dist,
        progress=progress
    )

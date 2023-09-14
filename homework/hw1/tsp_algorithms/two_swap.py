import pandas as pd
from more_itertools import distinct_combinations
from typing import List, Callable
from tsp_algorithms import TSPResult, logger

TWO_SWAP_DEFAULT_N_ITERS = 100_000


def generate_neighbors(current_path: List[int]) -> List[List[int]]:
    neighbors = []

    for i1, i2 in distinct_combinations(current_path, 2):
        new_cities = current_path.copy()
        new_cities[i1], new_cities[i2] = new_cities[i2], new_cities[i1]
        neighbors.append(new_cities)

    return neighbors


def two_swap_tsp(cities: List[int],
                 path_distance: Callable[[List[int]], float],
                 n_iters: int = TWO_SWAP_DEFAULT_N_ITERS) -> TSPResult:
    progress = []

    current_path = cities.copy()
    current_dist = path_distance(current_path)

    for iters in range(n_iters):
        logger(iters)

        neighbors = generate_neighbors(current_path)
        best_neighbor = min(neighbors, key=path_distance)
        best_neighbor_dist = path_distance(best_neighbor)

        if best_neighbor_dist < current_dist:
            current_path = best_neighbor
            current_dist = best_neighbor_dist

        progress.append(current_dist)

    return TSPResult(
        final_route=current_path + [current_path[0]],
        final_distance=current_dist,
        progress=progress
    )

import numpy.random as rd
import pandas as pd

from typing import Tuple, List, Callable
from tsp_algorithms import TSPResult, logger

TWO_OPT_DEFAULT_N_ITERS = 100_000


def random_adj_idx(length: int) -> Tuple[int, int]:
    assert length >= 4
    pair1 = rd.randint(length - 1)
    pair2 = rd.randint(length - 1)

    while pair1 - 1 < pair2 < pair1 + 1:
        pair2 = rd.randint(length - 1)

    p1, p2 = sorted([pair1, pair2])
    return p1, p2


def two_opt_tsp(cities: List[int],
                path_distance: Callable[[List[int]], float],
                n_iters: int = TWO_OPT_DEFAULT_N_ITERS) -> TSPResult:
    current_path = cities.copy()
    current_dist = path_distance(current_path)
    progress = [current_dist]

    for iters in range(n_iters):
        logger(iters)

        i1, i2 = random_adj_idx(len(current_path))

        old_path = current_path.copy()
        new_path = old_path[:i1 + 1] + old_path[i2:i1:-1] + old_path[i2 + 1:]
        new_distance = path_distance(new_path)

        if new_distance < current_dist:
            current_path = new_path
            current_dist = new_distance

        progress.append(current_dist)

    return TSPResult(
        final_route=current_path + [current_path[0]],
        final_distance=current_dist,
        progress=progress
    )

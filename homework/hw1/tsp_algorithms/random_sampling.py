from typing import List, Callable
from tsp_algorithms import TSPResult, shuffle, logger

RANDOM_SAMPLING_DEFAULT_N_ITERS = 1_000_000


def random_sampling_tsp(cities: List[int],
                        path_distance: Callable[[List[int]], float],
                        n_iters: int = RANDOM_SAMPLING_DEFAULT_N_ITERS) -> TSPResult:
    current_path = cities.copy()
    current_dist = path_distance(current_path)
    progress = [current_dist]

    for iters in range(n_iters):
        # logger(iters)

        new_path = shuffle(cities.copy())
        new_distance = path_distance(new_path)

        if new_distance < current_dist:
            current_dist = new_distance
            current_path = new_path

        progress.append(current_dist)

    return TSPResult(
        final_route=current_path + [current_path[0]],
        final_distance=current_dist,
        progress=progress
    )

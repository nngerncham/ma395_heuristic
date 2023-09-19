from typing import Tuple, List, Callable

import numpy.random as rd
from more_itertools import distinct_combinations

from tsp_algorithms import TSPResult, logger

TWO_OPT_DEFAULT_N_ITERS = 100_000


def valid_pairs(p1, p2):
    """
    Makes sure that p1 and p2 are not adjacent
    """
    return not (p1 - 1 < p2 < p1 + 1)


def generate_valid_pairs(n: int):
    """
    Generates all possible pairs then filters for only the non-adjacent ones
    """
    indices = [x for x in range(n)]
    combs = distinct_combinations(indices, 2)
    return [(p1, p2) for p1, p2 in combs if valid_pairs(p1, p2)]


def two_opt_swap(current_path, i1, i2):
    """
    Performs the 2-opt swap
    Suppose we want to swap edges (A, B) and (C, D), the path splits into: [start...A], [B...C], and [C...end]
    Then, they are combined by concatenating [start...A], reversed([B...C]), and [C...end] together
    """
    old_path = current_path.copy()
    new_path = old_path[:i1 + 1] + old_path[i2:i1:-1] + old_path[i2 + 1:]
    return new_path


def two_opt_tsp(cities: List[int],
                path_distance: Callable[[List[int]], float],
                n_iters: int = TWO_OPT_DEFAULT_N_ITERS) -> TSPResult:
    """
    Exhaustive 2-opt algorithm
    """
    current_path = cities.copy()
    current_dist = path_distance(current_path)
    progress = [current_dist]

    # Repeating N times
    for iters in range(n_iters):
        # logger(iters)

        # Generates all valid locations for 2-opt swaps
        neighbor_pairs = generate_valid_pairs(len(cities))
        for i1, i2 in neighbor_pairs:
            # Performs the swap and computes the path distance
            new_path = two_opt_swap(current_path, i1, i2)
            new_dist = path_distance(new_path)

            # Updates accordingly
            if new_dist < current_dist:
                current_path = new_path
                current_dist = new_dist

        progress.append(current_dist)

    return TSPResult(
        final_route=current_path + [current_path[0]],
        final_distance=current_dist,
        progress=progress
    )


def random_adj_idx(length: int) -> Tuple[int, int]:
    """
    Randomizes a pair of non-adjacent indices that such that they are valid for a 2-opt swap
    """

    # Randomizes a pair of indices
    assert length >= 4
    pair1 = rd.randint(length - 1)
    pair2 = rd.randint(length - 1)

    # Keeps generating a new second half of the pair until the pair is valid
    while not valid_pairs(pair1, pair2):
        pair2 = rd.randint(length - 1)

    # Ensures that the first index shows up before the second
    p1, p2 = sorted([pair1, pair2])
    return p1, p2


def rdn_two_opt_tsp(cities: List[int],
                    path_distance: Callable[[List[int]], float],
                    n_iters: int = TWO_OPT_DEFAULT_N_ITERS) -> TSPResult:
    """
    Randomized 2-opt algorithm
    """
    current_path = cities.copy()
    current_dist = path_distance(current_path)
    progress = [current_dist]

    # Repeating N times
    for iters in range(n_iters):
        # logger(iters)

        # Randomizes edges to swap, performs the swap, then computes the distance
        i1, i2 = random_adj_idx(len(current_path))
        new_path = two_opt_swap(current_path, i1, i2)
        new_distance = path_distance(new_path)

        # Updates accordingly
        if new_distance < current_dist:
            current_path = new_path
            current_dist = new_distance

        progress.append(current_dist)

    return TSPResult(
        final_route=current_path + [current_path[0]],
        final_distance=current_dist,
        progress=progress
    )

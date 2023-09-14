import numpy as np
from typing import List, Callable

import pandas as pd

from tsp_algorithms.two_swap import two_swap_tsp
from tsp_algorithms.two_opt import two_opt_tsp
from tsp_algorithms.random_sampling import random_sampling_tsp

REPS = 10
N_ITERS = 10
SEEDS = [42, 19, 77, 66, 23, 21, 99, 20, 90, 44]


def get_distance(dist_data: np.array, city1: int, city2: int):
    return dist_data[city1, city2]


def generate_pd_computer(dist_data: np.array) -> Callable[[List[int]], float]:
    def pd_computer_with_data(path: List[int]) -> float:
        path_to_use = path + [path[0]]
        return np.sum([get_distance(dist_data, x, y) for x, y in zip(path_to_use[:-1], path_to_use[1:])])

    return pd_computer_with_data


def generate_city_list(no_cities: int) -> List[int]:
    return [x for x in range(no_cities)]


def run_to_progress(data) -> List[float]:
    length, _ = data.shape
    pd_computer = generate_pd_computer(data)
    return two_opt_tsp(generate_city_list(length), pd_computer, N_ITERS).progress


def run_experiment(path_to_data: str, path_to_result: str):
    data = np.loadtxt(path_to_data, dtype=np.int32)

    to_results = np.array([run_to_progress(data) for _ in range(N_ITERS)])
    return pd.DataFrame(np.transpose(to_results))


if __name__ == "__main__":
    run_experiment("data/gr17_d.txt", "results/gr17/")

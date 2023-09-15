from typing import List, Callable

import numpy as np
import pandas as pd

from tsp_algorithms import TSPResult
from tsp_algorithms.random_sampling import random_sampling_tsp
from tsp_algorithms.two_opt import rdn_two_opt_tsp
from tsp_algorithms.two_swap import two_swap_tsp, rdn_two_swap_tsp

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
    xs = [x for x in range(no_cities)]
    np.random.shuffle(xs)
    return xs


def run_progress(data,
                 to_apply: Callable[[List[int], Callable[[List[int]], float], int], TSPResult]) -> List[float]:
    length, _ = data.shape
    pd_computer = generate_pd_computer(data)
    return to_apply(generate_city_list(length), pd_computer, N_ITERS).progress


def prep_progress(progress: List[float], trial_id: int):
    n = len(progress)
    trial_ids = np.repeat(trial_id, n)
    iterations = np.arange(n)

    return list(zip(iterations, trial_ids, progress))


def df_from_progresses(progresses: List[List[float]]):
    total = []
    for i, progress in enumerate(progresses):
        total.extend(prep_progress(progress, i))
    df = pd.DataFrame(total).rename(columns={0: "Iterations", 1: "Trial", 2: "Distance"})
    return df


def run_experiment(path_to_data: str):
    data = np.loadtxt(path_to_data, dtype=np.int32)

    ts_list = []
    rts_list = []
    rto_list = []
    rs_list = []

    for i in range(REPS):
        np.random.seed(SEEDS[i])
        ts_list.append(run_progress(data, two_swap_tsp))

        np.random.seed(SEEDS[i])
        rts_list.append(run_progress(data, rdn_two_swap_tsp))

        np.random.seed(SEEDS[i])
        rto_list.append(run_progress(data, rdn_two_opt_tsp))

        np.random.seed(SEEDS[i])
        rs_list.append(run_progress(data, random_sampling_tsp))

    ts_df = df_from_progresses(ts_list)
    ts_df["Type"] = "Full 2-swap"

    rts_df = df_from_progresses(rts_list)
    rts_df["Type"] = "Randomized 2-swap"

    rto_df = df_from_progresses(rto_list)
    rto_df["Type"] = "Randomized 2-opt"

    rs_df = df_from_progresses(rs_list)
    rs_df["Type"] = "Random sampling"

    return pd.concat([ts_df, rts_df, rto_df, rs_df])


if __name__ == "__main__":
    run_experiment("data/gr17_d.txt")

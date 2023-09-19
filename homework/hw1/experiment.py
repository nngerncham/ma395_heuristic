import multiprocessing as mp
from typing import List, Callable

import numpy as np
import pandas as pd
import time

from tsp_algorithms import TSPResult
from tsp_algorithms.random_sampling import random_sampling_tsp
from tsp_algorithms.two_opt import two_opt_tsp, rdn_two_opt_tsp
from tsp_algorithms.two_swap import two_swap_tsp, rdn_two_swap_tsp

REPS = 30
SEEDS = np.random.randint(1000, size=REPS)


def get_distance(dist_data: np.array, city1: int, city2: int):
    """
    Gets the distance from the distance matrix
    """
    return dist_data[city1, city2]


def generate_pd_computer(dist_data: np.array) -> Callable[[List[int]], float]:
    """
    Generates a function that computes the total path distance based on the data set
    This is so that the experiments can be run in parallel and thus saves time
    """

    def pd_computer_with_data(path: List[int]) -> float:
        path_to_use = path + [path[0]]
        return np.sum([get_distance(dist_data, x, y) for x, y in zip(path_to_use[:-1], path_to_use[1:])])

    return pd_computer_with_data


def generate_city_list(no_cities: int) -> List[int]:
    """
    Generates a randomized initial path
    """
    xs = [x for x in range(no_cities)]
    np.random.shuffle(xs)
    return xs


def run_progress(data,
                 to_apply: Callable[[List[int], Callable[[List[int]], float], int], TSPResult],
                 n_iters: int) -> List[float]:
    """
    Runs one trial of experiment on the algorithm passed in as `to_apply` on the data set for N iterations
    """
    length, _ = data.shape
    pd_computer = generate_pd_computer(data)
    return to_apply(generate_city_list(length), pd_computer, n_iters).progress


def prep_progress(progress: List[float], trial_id: int):
    """
    Transforms the progress from List of distances to a List of Tuple(iteration, trial, distance progress)
    """
    n = len(progress)
    trial_ids = np.repeat(trial_id, n)
    iterations = np.arange(n)

    return list(zip(iterations, trial_ids, progress))


def df_from_progresses(progresses: List[List[float]]):
    """
    Concatenates the progress from different trials and makes a Pandas DataFrame
    so that it is more suitable for plotting
    """
    total = []
    for i, progress in enumerate(progresses):
        total.extend(prep_progress(progress, i))
    df = pd.DataFrame(total).rename(columns={0: "Iterations", 1: "Trial", 2: "Distance"})
    return df


def run_with_seed(seed, data, function, n_iters):
    """
    Runs an experiment starting from a certain seed on given data set, algorithm, and number of iterations
    """
    np.random.seed(seed)
    return run_progress(data, function, n_iters)


def run_experiment(path_to_data: str, to_exclude: List[str], n_iters: int):
    """
    Takes in a path to the data set (distance matrix) file and runs all algorithms
    (except ones in `to_exclude`) for N iterations
    """

    # Prepping for parallelization
    data = np.loadtxt(path_to_data, dtype=np.int32)
    total = []
    datas = [data for _ in range(REPS)]
    n_iters_lst = [n_iters for _ in range(REPS)]

    # Runs each algorithm 30 times for `n_iters` iterations in parallel
    with mp.Pool(mp.cpu_count()) as pool:
        if "full_ts" not in to_exclude:
            fs = [two_swap_tsp for _ in range(REPS)]
            ts_list = pool.starmap(run_with_seed, zip(SEEDS, datas, fs, n_iters_lst))

            ts_df = df_from_progresses(ts_list)
            ts_df["Type"] = "Full 2-swap"
            total.append(ts_df)

        if "random_ts" not in to_exclude:
            fs = [rdn_two_swap_tsp for _ in range(REPS)]
            rts_list = pool.starmap(run_with_seed, zip(SEEDS, datas, fs, n_iters_lst))

            rts_df = df_from_progresses(rts_list)
            rts_df["Type"] = "Randomized 2-swap"
            total.append(rts_df)

        if "full_to" not in to_exclude:
            fs = [two_opt_tsp for _ in range(REPS)]
            to_list = pool.starmap(run_with_seed, zip(SEEDS, datas, fs, n_iters_lst))

            to_df = df_from_progresses(to_list)
            to_df["Type"] = "Full 2-opt"
            total.append(to_df)

        if "random_to" not in to_exclude:
            fs = [rdn_two_opt_tsp for _ in range(REPS)]
            rto_list = pool.starmap(run_with_seed, zip(SEEDS, datas, fs, n_iters_lst))

            rto_df = df_from_progresses(rto_list)
            rto_df["Type"] = "Randomized 2-opt"
            total.append(rto_df)

        if "random_sample" not in to_exclude:
            fs = [random_sampling_tsp for _ in range(REPS)]
            rs_list = pool.starmap(run_with_seed, zip(SEEDS, datas, fs, n_iters_lst))

            rs_df = df_from_progresses(rs_list)
            rs_df["Type"] = "Random sampling"
            total.append(rs_df)

    return pd.concat(total)


def run_timer(path_to_data: str, to_exclude: List[str], n_iters: int):
    """
    Times the amount of time a function takes and prints the average per iteration and total time
    given some number of iterations
    """
    data = np.loadtxt(path_to_data, dtype=np.int32)
    n, _ = data.shape
    path = generate_city_list(n)
    dist_computer = generate_pd_computer(data)

    def timer_helper(name: str, algo):
        start = time.time()
        _ = algo(path, dist_computer, n_iters)
        end = time.time()
        print(f"=== {name} ===")
        print(f"Total for {n_iters} iters: {end - start}s")
        print(f"Avg: {(end - start) / n_iters}s per iteration\n")

    if "full_ts" not in to_exclude:
        timer_helper("Exhaustive 2-swap", two_swap_tsp)

    if "random_ts" not in to_exclude:
        timer_helper("Randomized 2-swap", rdn_two_swap_tsp)

    if "full_to" not in to_exclude:
        timer_helper("Exhaustive 2-opt", two_opt_tsp)

    if "random_to" not in to_exclude:
        timer_helper("Randomized 2-opt", rdn_two_opt_tsp)

    if "random_sample" not in to_exclude:
        timer_helper("Random sampling", random_sampling_tsp)


if __name__ == "__main__":
    run_experiment("data/gr17_d.txt", [], 100)

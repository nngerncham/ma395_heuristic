from typing import List
import numpy as np
from tsp_algorithms.two_swap import two_swap_tsp
from tsp_algorithms.two_opt import two_opt_tsp
from tsp_algorithms.random_sampling import random_sampling_tsp


def in_class_2opt_distance(x: int, y: int):
    dist_matrix = [
        [0, 7, 7, 3, 8],
        [7, 0, 7, 5, 3],
        [7, 7, 0, 8, 5],
        [3, 5, 8, 0, 5],
        [8, 3, 5, 5, 0]
    ]

    return dist_matrix[x][y]


def in_class_path_distance(path: List[int]):
    path_to_use = path + [path[0]]
    return np.sum([in_class_2opt_distance(x, y) for x, y in zip(path_to_use[:-1], path_to_use[1:])])


if __name__ == "__main__":
    rd_res = random_sampling_tsp([i for i in range(5)], in_class_path_distance, n_iters=50)
    ts_res = two_swap_tsp([i for i in range(5)], in_class_path_distance, n_iters=50)
    to_res = two_opt_tsp([i for i in range(5)], in_class_path_distance, n_iters=50)

    print(rd_res)
    print(ts_res)
    print(to_res)

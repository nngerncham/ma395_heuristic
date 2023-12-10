import os
from dataclasses import dataclass
from time import time_ns
from typing import List

import diskannpy
import numpy as np
from bayes_opt import BayesianOptimization

from algorithms import utils
from algorithms.moo import Scaler, NUM_THREADS


@dataclass
class BOTracker:
    build_times: List[float]
    search_times: List[float]
    recalls: List[float]


def bo_obj_factory(data_set, queries, gts, factory_weight):
    scaler = Scaler(None, None, None, None)
    bo_tracker = BOTracker([], [], [])

    def bo_objective(M, C, S, alpha):
        idx_path = "../index_scaling2/"
        os.system("rm -rf " + idx_path + "*")
        M = int(M * 1024)
        if M == 0:
            M = 1
        C = int(C * 1024)
        S = int(S * 1024)

        # build time
        start_time = time_ns()
        diskannpy.build_memory_index(data_set, distance_metric="l2", index_directory=idx_path,
                                     complexity=C, graph_degree=M,
                                     alpha=alpha, num_threads=NUM_THREADS)
        end_time = time_ns()
        build_time = (end_time - start_time) / 1e9  # seconds
        bo_tracker.build_times.append(build_time)

        index = diskannpy.StaticMemoryIndex(idx_path, distance_metric="l2", num_threads=NUM_THREADS,
                                            initial_search_complexity=S,
                                            vector_dtype=np.float32)

        # search time
        start_time = time_ns()
        results, _ = index.batch_search(queries, 100, S, NUM_THREADS)
        end_time = time_ns()
        search_time = (end_time - start_time) / 1e9  # seconds
        bo_tracker.search_times.append(search_time)

        recall_error = 1 - utils.evaluate_knn(results, gts)
        bo_tracker.recalls.append(1 - recall_error)

        build_was_none = False
        if scaler.build_min is None:
            build_was_none = True
            scaler.build_min = build_time
            scaler.build_max = build_time
        if build_time < scaler.build_min:
            scaler.build_min = build_time
        if build_time > scaler.build_max:
            scaler.build_max = build_time
        if build_was_none:
            build_time = 1
        else:
            build_time = (build_time - scaler.build_min) / (scaler.build_max - scaler.build_min)

        search_was_none = False
        if scaler.query_min is None:
            search_was_none = True
            scaler.query_min = search_time
            scaler.query_max = search_time
        if search_time < scaler.query_min:
            scaler.query_min = search_time
        if search_time > scaler.query_max:
            scaler.query_max = search_time
        if search_was_none:
            search_time = 1
        else:
            search_time = (search_time - scaler.query_min) / (scaler.query_max - scaler.query_min)

        obj_value = np.dot([build_time, search_time, recall_error], factory_weight)
        return 2 / (1 + obj_value)

    return bo_objective, scaler, bo_tracker


def bayesian_optimization(data_set, queries, gts, bo_weight, n_iters=100):
    bo_objective, _, bo_tracker = bo_obj_factory(data_set, queries, gts, bo_weight)
    bo = BayesianOptimization(
        bo_objective,
        {
            'M': (0.001, 1),
            'C': (0.1, 1),
            'S': (0.1, 1),
            'alpha': (1, 2)
        },
        allow_duplicate_points=True,
    )

    bo.maximize(init_points=1, n_iter=n_iters)
    return bo.res, bo_tracker


if __name__ == '__main__':
    data = utils.load_data("/home/nawat/muic/ma395_heuristic/project/algorithms/data/siftsmall/siftsmall_base.fvecs",
                           128)
    query = utils.load_data("/home/nawat/muic/ma395_heuristic/project/algorithms/data/siftsmall/siftsmall_query.fvecs",
                            128)
    gt = utils.load_data(
        "/home/nawat/muic/ma395_heuristic/project/algorithms/data/siftsmall/siftsmall_groundtruth.ivecs",
        100, np.int32)
    weights = [
        [1, 1, 1],
        [1, 2, 2],
        [2, 1, 1],
        [1, 2, 3],
    ]
    for weight in weights:
        results_name = f"../bo-small-{''.join(map(str, weight))}-unscaled.csv"
        with open(results_name, "w") as f:
            f.write("trials,iter,M,C,S,alpha,build_time,search_time,recall,ws\n")
        for trial in range(20):
            bo_result, tracker = bayesian_optimization(data, query, gt, weight, 20)
            with open(results_name, "a") as f:
                for gen_iter, generation in enumerate(bo_result):
                    entry = (f"{trial},"
                             f"{gen_iter},"
                             f"{int(generation['params']['M'] * 1024)},"
                             f"{int(generation['params']['C'] * 1024)},"
                             f"{int(generation['params']['S'] * 1024)},"
                             f"{generation['params']['alpha']},"
                             f"{tracker.build_times[gen_iter]},"
                             f"{tracker.search_times[gen_iter]},"
                             f"{tracker.recalls[gen_iter]},"
                             f"{generation['target']}\n")
                    f.write(entry)

import os
from time import time_ns

import diskannpy
import numpy as np
from bayes_opt import BayesianOptimization

from algorithms import utils
from algorithms.moo import Scaler, NUM_THREADS


def bo_obj_factory(data_set, queries, gts):
    scaler = Scaler(None, None, None, None)

    def bo_objective(M, C, S, alpha):
        idx_path = "../index_non_scaling/"
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

        index = diskannpy.StaticMemoryIndex(idx_path, distance_metric="l2", num_threads=NUM_THREADS,
                                            initial_search_complexity=S,
                                            vector_dtype=np.float32)

        # search time
        start_time = time_ns()
        results, _ = index.batch_search(queries, 100, S, NUM_THREADS)
        end_time = time_ns()
        search_time = (end_time - start_time) / 1e9  # seconds

        recall = 1 - utils.evaluate_knn(results, gts)

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

        return 1 / np.dot([build_time, search_time, recall], [1, 1, 1])

    return bo_objective, scaler


def bayesian_optimization(data_set, queries, gts, n_iters=100):
    bo_objective, scaler = bo_obj_factory(data_set, queries, gts)
    bo = BayesianOptimization(
        bo_objective,
        {
            'M': (0.00097, 1),
            'C': (0.098, 1),
            'S': (0.098, 1),
            'alpha': (1, 2)
        },
    )

    bo.maximize(init_points=10, n_iter=n_iters)
    return bo.res


if __name__ == '__main__':
    data = utils.load_data("/home/nawat/muic/ma395_heuristic/project/algorithms/data/siftsmall/siftsmall_base.fvecs",
                           128)
    query = utils.load_data("/home/nawat/muic/ma395_heuristic/project/algorithms/data/siftsmall/siftsmall_query.fvecs",
                            128)
    gt = utils.load_data(
        "/home/nawat/muic/ma395_heuristic/project/algorithms/data/siftsmall/siftsmall_groundtruth.ivecs",
        100, np.int32)
    data_apply_function, _scaler = bo_obj_factory(data, query, gt)
    bo_result = bayesian_optimization(data, query, gt, 100)

    results_name = "../bo-small-111.csv"
    with open(results_name, "w") as f:
        f.write("iter,M,C,S,alpha,ws\n")
    with open(results_name, "a") as f:
        for gen_iter, generation in enumerate(bo_result):
            entry = (f"{gen_iter},"
                     f"{int(generation['params']['M'] * 1024)},"
                     f"{int(generation['params']['C'] * 1024)},"
                     f"{int(generation['params']['S'] * 1024)},"
                     f"{generation['params']['alpha']},"
                     f"{1 / generation['target']}\n")
            f.write(entry)

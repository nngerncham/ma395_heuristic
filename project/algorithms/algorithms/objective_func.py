import os
from os import getpid
from time import time_ns
from typing import Any, Set, List

import diskannpy
import numpy as np
from psutil import Process

import nsga2
import utils

NUM_THREADS = 24


class BuildParams(nsga2.Individual):
    def __init__(self, v):
        super().__init__(v)
        self.function_values = [0, 0, 0, 0]

    def max_degree(self):
        return utils.bin_to_int(self.v[0])

    def size_construction(self):
        return 100 + utils.bin_to_int(self.v[1])

    def size_search(self):
        return 100 + utils.bin_to_int(self.v[2])

    def alpha(self):
        top = utils.bin_to_int(self.v[3])
        return 1 + (top / 1024)

    def set_build_time(self, build_time):
        self.function_values[0] = build_time

    def set_memory(self, memory):
        self.function_values[1] = memory

    def set_search_time(self, search_time):
        self.function_values[2] = search_time

    def set_recall(self, recall):
        self.function_values[3] = 1 / (recall + 1e-6)  # since want to maximize

    def __hash__(self):
        return hash(tuple(self.v))

    def __repr__(self):
        return f"""
        max_degree: {self.max_degree()}
        size_construction: {self.size_construction()}
        size_search: {self.size_search()}
        alpha: {self.alpha()}
        """

    def __str__(self):
        return (f"{self.max_degree()},{self.size_construction()},{self.size_search()},{self.alpha()},"
                f"{self.function_values[0]},{self.function_values[1]},{self.function_values[2]},{1 / self.function_values[3] - 1e-6}\n")


def moo_factory(data_set: np.ndarray[np.ndarray[Any]],
                queries: np.ndarray[np.ndarray[Any]],
                gts: np.ndarray[np.ndarray[Any]]):
    def apply_function(bps: nsga2.Population):  # modifies the bp object in-place
        idx_path = "../index/"
        for bp in bps:
            os.system("rm -rf " + idx_path + "*")

            # build time
            start_time = time_ns()
            diskannpy.build_memory_index(data_set, distance_metric="l2", index_directory=idx_path,
                                         complexity=bp.size_construction(), graph_degree=bp.max_degree(),
                                         alpha=bp.alpha(), num_threads=NUM_THREADS)
            end_time = time_ns()
            build_time = (end_time - start_time) / 1e9  # seconds
            bp.set_build_time(build_time)

            # memory usage
            proc = Process(getpid())
            start_mem = proc.memory_info().rss
            index = diskannpy.StaticMemoryIndex(idx_path, distance_metric="l2", num_threads=NUM_THREADS,
                                                initial_search_complexity=bp.size_search(),
                                                vector_dtype=np.float32)
            end_mem = proc.memory_info().rss
            index_size = (end_mem - start_mem) / 1e9
            # index_size = (start_mem)
            bp.set_memory(index_size)

            # search time
            start_time = time_ns()
            results, _ = index.batch_search(queries, 100, bp.size_search(), NUM_THREADS)
            end_time = time_ns()
            search_time = (end_time - start_time) / 1e9  # seconds
            bp.set_search_time(search_time)

            # recall
            recall = utils.evaluate_knn(results, gts)
            bp.set_recall(recall)

    return apply_function


def cut_concat(individual1: BuildParams, individual2: BuildParams):
    # mutation is built into this function
    new_v: List[str] = ["", "", "", ""]
    for i in range(4):
        cut_point = np.random.randint(10)
        new_v[i] = individual1.v[i][:cut_point] + individual2.v[i][cut_point:]
        flip = {'0': '1', '1': '0'}
        for j in range(len(new_v[i])):
            if np.random.rand() < 0.05:
                new_v[i] = new_v[i][:j] + flip[new_v[i][j]] + new_v[i][j + 1:]
        if utils.bin_to_int(new_v[i]) <= 0:
            new_v[i] = utils.int_to_bin(1)
    return BuildParams(new_v)


def choice_from_set(population: Set):
    return np.random.choice(list(population))


def cut_concat_new_pop(population: nsga2.Population):
    n = len(population)
    new_pop = set()
    while len(new_pop) < n:
        ind1 = choice_from_set(population)
        ind2 = choice_from_set(population)
        new_pop.add(cut_concat(ind1, ind2))

    return new_pop


def randomize_individual():
    v = [utils.int_to_bin(np.random.randint(1, 1024)),
         utils.int_to_bin(np.random.randint(924)),
         utils.int_to_bin(np.random.randint(924)),
         utils.int_to_bin(np.random.randint(1, 1024))]
    return BuildParams(v)


def generate_p0(n: int):
    vs = set()
    while len(vs) < n:
        vs.add(randomize_individual())
    return vs


if __name__ == '__main__':
    data = utils.load_data("/home/nawat/muic/ma395_heuristic/project/algorithms/data/siftsmall/siftsmall_base.fvecs",
                           128)
    query = utils.load_data("/home/nawat/muic/ma395_heuristic/project/algorithms/data/siftsmall/siftsmall_query.fvecs",
                            128)
    gt = utils.load_data(
        "/home/nawat/muic/ma395_heuristic/project/algorithms/data/siftsmall/siftsmall_groundtruth.ivecs",
        100, np.int32)
    data_apply_function = moo_factory(data, query, gt)

    p0 = generate_p0(10)
    nsga2_result = nsga2.nsga2(data_apply_function, p0, cut_concat_new_pop, 4, 10)
    with open("../result-small.csv", "w") as f:
        f.write("generation,max_deg,size_construction,size_search,alpha,build_time,memory,search_time,recall\n")
        for gen_iter, generation in enumerate(nsga2_result.populations):
            for individual in generation:
                f.write(f"{gen_iter}," + str(individual))

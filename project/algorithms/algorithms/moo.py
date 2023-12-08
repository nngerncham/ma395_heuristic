import dataclasses
import os
from time import time_ns
from typing import Any, List, Callable, Tuple

import diskannpy
import numpy as np

import nsga2
import utils

NUM_THREADS = 10
P_MU = 0.05


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
        self.function_values[3] = 1 - recall  # since want to maximize

    def __hash__(self):
        return hash(tuple(self.v))

    def __repr__(self):
        # return f"""
        # max_degree: {self.max_degree()}
        # size_construction: {self.size_construction()}
        # size_search: {self.size_search()}
        # alpha: {self.alpha()}
        # """
        return str(self.v)

    def __str__(self):
        return (f"{self.max_degree()},{self.size_construction()},{self.size_search()},{self.alpha()},"
                f"{self.function_values[0]},{self.function_values[1]},{self.function_values[2]},"
                f"{1 - self.function_values[3]}\n")


@dataclasses.dataclass
class Scaler:
    build_min: float | None
    build_max: float | None
    query_min: float | None
    query_max: float | None


def moo_factory(data_set: np.ndarray[np.ndarray[Any]],
                queries: np.ndarray[np.ndarray[Any]],
                gts: np.ndarray[np.ndarray[Any]]):
    scaler = Scaler(None, None, None, None)  # scaler is created along with apply_function

    def apply_function(bps: nsga2.Population):  # modifies the bp object in-place
        idx_path = "../index_scaling/"
        build_times = []
        search_times = []
        recalls = []
        for bp in bps:
            os.system("rm -rf " + idx_path + "*")

            # build time
            start_time = time_ns()
            diskannpy.build_memory_index(data_set, distance_metric="l2", index_directory=idx_path,
                                         complexity=bp.size_construction(), graph_degree=bp.max_degree(),
                                         alpha=bp.alpha(), num_threads=NUM_THREADS)
            end_time = time_ns()
            build_time = (end_time - start_time) / 1e9  # seconds
            # bp.set_build_time(build_time)
            build_times.append(build_time)

            # memory usage
            index = diskannpy.StaticMemoryIndex(idx_path, distance_metric="l2", num_threads=NUM_THREADS,
                                                initial_search_complexity=bp.size_search(),
                                                vector_dtype=np.float32)

            # search time
            start_time = time_ns()
            results, _ = index.batch_search(queries, 100, bp.size_search(), NUM_THREADS)
            end_time = time_ns()
            search_time = (end_time - start_time) / 1e9  # seconds
            # bp.set_search_time(search_time)
            search_times.append(search_time)

            # recall
            recall = utils.evaluate_knn(results, gts)
            # bp.set_recall(recall)
            recalls.append(recall)

        if scaler.build_min is None:
            scaler.build_min = min(build_times)
            scaler.build_max = max(build_times)
        if min(build_times) < scaler.build_min:
            scaler.build_min = min(build_times)
        if max(build_times) > scaler.build_max:
            scaler.build_max = max(build_times)

        if scaler.query_min is None:
            scaler.query_min = min(search_times)
            scaler.query_max = max(search_times)
        if min(search_times) < scaler.query_min:
            scaler.query_min = min(search_times)
        if max(search_times) > scaler.query_max:
            scaler.query_max = max(search_times)

        for i, bp in enumerate(bps):
            bp.set_build_time((build_times[i] - scaler.build_min) / (scaler.build_max - scaler.build_min))
            bp.set_search_time((search_times[i] - scaler.query_min) / (scaler.query_max - scaler.query_min))
            bp.set_recall(recalls[i])

    return apply_function, scaler  # returns scaler as well, so it doesn't get garbage collected
    # return apply_function


def single_cutcatenate(individual1: BuildParams, individual2: BuildParams):
    # mutation is built into this function
    new_v: List[str] = ["", "", "", ""]
    flip = {'0': '1', '1': '0'}
    for i in range(4):
        cut_point = np.random.randint(10)
        new_v[i] = individual1.v[i][:cut_point] + individual2.v[i][cut_point:]
        for j in range(len(new_v[i])):
            if np.random.rand() < P_MU:
                new_v[i] = new_v[i][:j] + flip[new_v[i][j]] + new_v[i][j + 1:]
        if utils.bin_to_int(new_v[i]) <= 0:
            new_v[i] = utils.int_to_bin(1)
    return BuildParams(new_v)


def multi_cutcatenate(individual1: BuildParams, individual2: BuildParams):
    # mutation is built into this function
    new_v: List[str] = ["", "", "", ""]
    flip = {'0': '1', '1': '0'}
    for i in range(4):
        num_cut_points = np.random.randint(1, 4)  # 1 <= num_cut_points <= 3
        cut_points = [0]
        cut_points.extend(sorted(np.random.randint(10, size=num_cut_points)))
        cut_points.append(10)
        inds = [individual1, individual2]
        sections = [inds[j % 2].v[i][cut_points[j - 1]:cut_points[j]] for j in range(1, len(cut_points))]
        new_v[i] = "".join(sections)

        # mutation
        for j in range(len(new_v[i])):
            if np.random.rand() < P_MU:
                new_v[i] = new_v[i][:j] + flip[new_v[i][j]] + new_v[i][j + 1:]
        if utils.bin_to_int(new_v[i]) <= 0:
            new_v[i] = utils.int_to_bin(1)
    return BuildParams(new_v)


def uniform_random_selection(population: nsga2.Population) -> (BuildParams, BuildParams):
    return np.random.choice(list(population), size=2)


def tournament_selection(population: nsga2.Population) -> (BuildParams, BuildParams):
    frontiers = nsga2.fast_non_dominated_sort(population.copy())
    pop_copy = []
    for front in frontiers:
        if len(pop_copy) >= 4:
            break
        pop_copy.extend(front)
    np.random.shuffle(pop_copy)

    mid_point = len(pop_copy) // 2
    match1 = pop_copy[:mid_point]
    match2 = pop_copy[mid_point:]

    return min(match1), min(match2)


def make_new_pop_factory(crossover: Callable[[BuildParams, BuildParams], BuildParams],
                         select: Callable[[nsga2.Population], Tuple[BuildParams, BuildParams]]) \
        -> Callable[[nsga2.Population], nsga2.Population]:
    def new_pop(population: nsga2.Population):
        n = len(population)
        next_pop = set()
        while len(next_pop) < n:
            ind1, ind2 = select(population)
            next_pop.add(crossover(ind1, ind2))

        return next_pop

    return new_pop


def generate_p0(n: int):
    def randomize_individual():
        v = [utils.int_to_bin(np.random.randint(1, 1024)),
             utils.int_to_bin(np.random.randint(924)),
             utils.int_to_bin(np.random.randint(924)),
             utils.int_to_bin(np.random.randint(1, 1024))]
        return BuildParams(v)

    vs = set()
    while len(vs) < n:
        vs.add(randomize_individual())

    nsga2.fast_non_dominated_sort(vs)
    nsga2.crowding_distance_assignment(vs, 4)
    return vs


if __name__ == '__main__':
    data = utils.load_data("/home/nawat/muic/ma395_heuristic/project/algorithms/data/siftsmall/siftsmall_base.fvecs",
                           128)
    query = utils.load_data("/home/nawat/muic/ma395_heuristic/project/algorithms/data/siftsmall/siftsmall_query.fvecs",
                            128)
    gt = utils.load_data(
        "/home/nawat/muic/ma395_heuristic/project/algorithms/data/siftsmall/siftsmall_groundtruth.ivecs",
        100, np.int32)
    data_apply_function, _scaler = moo_factory(data, query, gt)
    # data_apply_function = moo_factory(data, query, gt)

    crossover_methods = {
        "single-cutcat-unif": make_new_pop_factory(single_cutcatenate, uniform_random_selection),
        "multi-cutcat-unif": make_new_pop_factory(multi_cutcatenate, uniform_random_selection),
        "single-cutcat-tour": make_new_pop_factory(single_cutcatenate, tournament_selection),
        "multi-cutcat-tour": make_new_pop_factory(multi_cutcatenate, tournament_selection)
    }

    n_trials = 20
    file_target = "../result-scaling-frontier.csv"
    with open(file_target, "w") as f:
        f.write(
            "trial,generation,method,max_deg,size_construction,size_search,alpha,build_time,memory,search_time,recall\n")
    for trial in range(n_trials):
        for method_key in crossover_methods.keys():
            p0 = generate_p0(10)
            nsga2_result = nsga2.nsga2(data_apply_function, p0, crossover_methods[method_key], 4, 10)
            with open(file_target, "a") as f:
                for gen_iter, generation in enumerate(nsga2_result.populations):
                    for individual in generation:
                        entry = f"{trial},{gen_iter},{method_key}," + str(individual)
                        f.write(entry)

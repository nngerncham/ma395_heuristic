from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def load_data(source_path: str | Path, dims: int, tp=np.float32) -> np.ndarray[Any]:
    """
    Loads the data for ANNS
    :param source_path: Path to the data file
    :param dims: Dimensionality of data
    :param tp: type
    :return: numpy array of shape (n, dims)
    """
    loaded_data = np.fromfile(source_path, dtype=tp)
    return np.reshape(loaded_data, (-1, dims + 1))[:, 1:]


def evaluate_knn(results: np.ndarray[np.ndarray[int]], gts: np.ndarray[np.ndarray[int]]) -> float:
    """
    Calculates the recall of the result of knn search
    :param results: Result from an algorithm
    :param gts: Ground truths
    :return: Average recall
    """
    sets_of_results = [set(result) for result in results]
    sets_of_gts = [set(gt) for gt in gts]
    recalls = np.array([len(a & b) / len(b) for a, b in zip(sets_of_results, sets_of_gts)])
    return np.mean(recalls)


def int_to_bin(int_value: int) -> str:
    """
    Converts an integer to a binary string
    :param int_value: integer
    :return: binary string
    """
    return bin(int_value & 0x7ff)[2:].zfill(10)


def bin_to_int(bin_str: str) -> int:
    """
    Converts a binary string to an integer
    :param bin_str: binary string
    :return: integer
    """
    return int(bin_str, 2)


def csv_to_non_dominated_frontier(csv_path: str | Path) -> np.ndarray[np.ndarray[float]]:
    split_path = csv_path.split("-")
    front = split_path[:3]
    back = split_path[-1]
    target_file = f"../{'-'.join(front)}-frontier-{back}"

    methods = [
        "single-cutcat-unif",
        "multi-cutcat-unif",
        "single-cutcat-tour",
        "multi-cutcat-tour"
    ]
    data = pd.read_csv(csv_path)
    for method in methods:
        to_use = data[data["method"] == method]

        def pd_to_individual(row):
            from moo import BuildParams
            ind = BuildParams([bin_to_int(row["max_deg"]),
                               bin_to_int(row["size_construction"]),
                               bin_to_int(row["size_search"]),
                               bin_to_int(row["alpha"])])
            ind.set_build_time(row["build_time"])
            ind.set_search_time(row["search_time"])
            ind.set_recall(row["recall"])

        to_use.apply(pd_to_individual)


if __name__ == '__main__':
    data_path = "../result-scaling.csv"
    print(csv_to_non_dominated_frontier(data_path))

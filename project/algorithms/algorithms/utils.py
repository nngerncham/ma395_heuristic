from pathlib import Path
from typing import Any

import numpy as np


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


if __name__ == '__main__':
    data = load_data("/home/nawat/muic/ma395_heuristic/project/algorithms/data/siftsmall/siftsmall_base.fvecs", 128)
    quality = evaluate_knn(data, data)
    print(quality)

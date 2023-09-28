import numpy as np
from scipy.stats import norm, uniform
from typing import List, Tuple


def gaussian_perturbation(x: np.ndarray, bounds: List[Tuple[float, float]], sigma_ratio: float = 0.1) -> np.ndarray:
    assert len(bounds) == len(x)
    v = np.repeat(0, len(x))
    for i, bound in enumerate(bounds):
        lb, ub = bound
        scale = (ub - lb) * sigma_ratio
        vi = x[i] + scale * norm.rvs()
        while not (lb <= vi <= ub):
            vi = x[i] + scale * norm.rvs()
        v[i] = vi
    return np.array(v)


def uniform_perturbation(x: np.ndarray, bounds: List[Tuple[float, float]], sigma_ratio: float = 0.1) -> np.ndarray:
    assert len(bounds) == len(x)
    v = np.repeat(0, len(x))
    for i, bound in enumerate(bounds):
        lb, ub = bound
        scale = (ub - lb) * sigma_ratio
        if x[i] - scale >= lb:
            v[i] = uniform.rvs(x[i] - scale, 2 * scale)
        else:
            v[i] = uniform.rvs(lb, min(ub, x[i] + scale - lb))
    return v

from typing import List, Tuple, TypeVar

import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import seaborn as sns
from numpy.random import default_rng

from simulated_annealing.algorithm import SAResult

T = TypeVar("T")


def shape_progress(result: SAResult, trial: int) -> pd.DataFrame:
    return pd.DataFrame({
        "Iteration": np.arange(len(result.b_progress)),
        "Best cost": result.b_progress,
        "Current cost": result.c_progress,
        "Acceptance rate": result.acceptance_rate,
        "Trial": trial,
    })


def shuffle(lst: List[T]) -> List[T]:
    rng = default_rng()
    lst_cp = lst.copy()
    rng.shuffle(lst_cp)
    return lst_cp

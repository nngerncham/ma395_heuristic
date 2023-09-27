from typing import List, Tuple, TypeVar

import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import seaborn as sns
from numpy.random import default_rng

from simulated_annealing.algorithm import SAResult

T = TypeVar("T")


def shape_progress(result: SAResult) -> pd.DataFrame:
    return pd.DataFrame({
        "Iteration": np.arange(len(result.b_progress)),
        "Best cost": result.b_progress,
        "Current cost": result.c_progress,
        "Acceptance rate": result.acceptance_rate,
    })


def shuffle(lst: List[T]) -> List[T]:
    rng = default_rng()
    lst_cp = lst.copy()
    rng.shuffle(lst_cp)
    return lst_cp


def plot(df: pd.DataFrame, optimal: float) -> Tuple[plt.Figure, Tuple]:
    fig, (a1, a2, a3) = plt.subplots(1, 3, width_ratios=[0.3, 0.35, 0.35], figsize=(15, 3))
    sns.lineplot(df, x="Iteration", y="Acceptance rate", ax=a1)
    a2.axhline(optimal, color="black")
    sns.lineplot(df, x="Iteration", y="Best cost", hue="Type", ax=a2)
    a3.axhline(optimal, color="black")
    sns.lineplot(df, x="Iteration", y="Current cost", hue="Type", ax=a3)

    return fig, (a1, a2, a3)

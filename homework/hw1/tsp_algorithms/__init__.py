from dataclasses import dataclass
from typing import List, TypeVar
from numpy.random import default_rng


@dataclass
class TSPResult:
    final_route: List[int]
    final_distance: float
    progress: List[float]

    def __repr__(self):
        return "Final distance: " + str(self.final_distance) + "\nRoute: " + str(self.final_route)


T = TypeVar("T")


def shuffle(lst: List[T]) -> List[T]:
    rng = default_rng()
    lst_cp = lst.copy()
    rng.shuffle(lst_cp)
    return lst_cp


SUBDIVISION = 10_000


def logger(iters: int):
    if iters % SUBDIVISION == 0 and iters > 0:
        print(f"{iters} iterations done")

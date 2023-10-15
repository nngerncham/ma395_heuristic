from dataclasses import dataclass
from typing import Callable, Collection, Dict, Tuple, Any


class NeighborsTabuListInterface:
    def __init__(self, tenure: int, nbrs_size: int, cost: Callable[[Collection], float]):
        self.tenure = tenure
        self.nbrs_size = nbrs_size
        self.cost = cost

    def add(self, attr: Any, i: int):
        pass

    def neighbors(self, s_current: Collection) -> Tuple[Collection, Any]:
        pass

    def evaluate(self, s_current: Collection, i: int) -> float:
        pass

    def is_tabu(self, attr: Any, i: int) -> bool:
        pass


@dataclass
class TabuSearchResult:
    solution: Collection
    cost: float
    progress_current: Collection[float]
    progress_best: Collection[float]


def tabu_search(cost: Callable[[Collection], float],
                s0: Collection,
                tabu_neighbor: NeighborsTabuListInterface,
                max_iters: int = 200,
                ) -> TabuSearchResult:
    # initializing stuff
    s_current = s0
    s_best = s0
    cost_current = cost(s_current)
    cost_best = cost_current

    currents = [cost_current]
    bests = [cost_best]

    for i in range(max_iters):
        # generates neighbors and their respective attributes and sort by cost
        nbrs, attrs = tabu_neighbor.neighbors(s_current)
        sorted_nbrs = sorted(zip(nbrs, attrs), key=lambda x: tabu_neighbor.evaluate(x[0], i))

        # checking neighbor for tabu-ness, skip that iteration if all is tabu
        for nbr, attr in sorted_nbrs:
            # if is not tabu, accept immediately
            if not tabu_neighbor.is_tabu(attr, i):
                s_current = nbr
                cost_current = cost(s_current)
                tabu_neighbor.add(attr, i)
                break
            # is tabu but cost < AL
            elif tabu_neighbor.evaluate(nbr, i) < cost_best:
                s_current = nbr
                cost_current = cost(s_current)
                tabu_neighbor.add(attr, i)
                break

        # update best costs
        if cost_current <= cost_best:
            s_best = s_current
            cost_best = cost_current

        currents.append(cost_current)
        bests.append(cost_best)

    return TabuSearchResult(s_best, cost_best, currents, bests)

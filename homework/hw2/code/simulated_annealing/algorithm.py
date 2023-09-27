import numpy as np
from dataclasses import dataclass
from numpy import ndarray
from typing import Callable, List

ACCEPTANCE_RATE_INTERVALS = 20


@dataclass
class SAResult:
    solution: ndarray
    cost: float

    c_progress: List[float]
    b_progress: List[float]
    acceptance_rate: List[float]

    def __repr__(self):
        return (f"soln: {self.solution}\ncost: {self.cost}\n"
                f"(cur, best, acc): "
                f"{len(self.c_progress)}, {len(self.b_progress)}, {len(self.acceptance_rate)}")


def simple_simulated_annealing(objective_f: Callable[[ndarray or List], float],
                               generate_neighbor: Callable[[ndarray or List], ndarray or List],
                               initial_solution: ndarray,
                               initial_temperature: float = 1_000,
                               max_iterations: int = 10_000,
                               temp_factor: float = 0.95) -> SAResult:
    c_progress = []
    b_progress = []
    acceptance = []

    current_soln = initial_solution
    current_cost = objective_f(current_soln)
    current_temp = initial_temperature

    best_soln = current_soln
    best_cost = current_cost

    iters = 0
    while current_temp > 0 and iters < max_iterations:
        new_candidate = generate_neighbor(current_soln)
        new_cost = objective_f(new_candidate)

        delta_cost = new_cost - current_cost
        if delta_cost < 0:
            acceptance.append(1)
            current_soln = new_candidate
            current_cost = new_cost

            if current_cost < best_cost:
                best_soln = current_soln
                best_cost = current_cost

        elif np.random.uniform() < np.exp(- delta_cost / current_temp):
            current_soln = new_candidate
            current_cost = new_cost
            acceptance.append(1)
        else:
            acceptance.append(0)

        iters += 1
        current_temp *= temp_factor

        c_progress.append(current_cost)
        b_progress.append(best_cost)

    acceptance_rate = list(np.cumsum(acceptance) / (np.arange(len(acceptance)) + 1))

    return SAResult(
        solution=best_soln,
        cost=best_cost,
        c_progress=c_progress,
        b_progress=b_progress,
        acceptance_rate=acceptance_rate,
    )

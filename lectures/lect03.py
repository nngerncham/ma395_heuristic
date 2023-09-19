import numpy as np
from dataclasses import dataclass
from numpy import ndarray
from typing import Callable, List, Tuple


@dataclass
class SAResult:
    solution: ndarray
    cost: float
    c_progress: List[float]
    b_progress: List[float]


def simulated_annealing(objective_f: Callable[[ndarray], float],
                        generate_neighbor: Callable[[ndarray], ndarray],
                        initial_solution: ndarray,
                        initial_temperature: float = 10_000,
                        metropolis_iters: int = 500,
                        final_temperature: float = 1e-10,
                        max_iterations: int = 100_000,
                        alpha: float = 0.95,
                        beta: float = 1) -> SAResult:
    c_progress = []
    b_progress = []

    def metropolis(current_metro_solution: ndarray,
                   current_metro_cost: float,
                   best_metro_cost: float,
                   metro_temperature: float,
                   metro_iters: int) -> Tuple[ndarray, float]:  # solution and cost
        for _ in range(metro_iters):
            new_metro_solution = generate_neighbor(current_metro_solution)
            new_metro_cost = objective_f(new_metro_solution)
            delta_cost = new_metro_cost - current_metro_cost

            if delta_cost < 0:
                current_metro_solution = new_metro_solution
                current_metro_cost = new_metro_cost
                if current_metro_cost < best_metro_cost:
                    best_metro_cost = current_metro_cost
            else:
                if np.random.uniform() < np.exp(- delta_cost / metro_temperature):
                    current_metro_solution = new_metro_solution
                    current_metro_cost = new_metro_cost

            c_progress.append(current_metro_cost)
            b_progress.append(best_metro_cost)

        return current_metro_solution, current_metro_cost

    temperature = initial_temperature
    current_solution = initial_solution
    current_cost = objective_f(current_solution)
    c_progress.append(current_cost)

    best_solution = current_solution
    best_cost = current_cost
    b_progress.append(best_cost)

    iters = 0
    while temperature > final_temperature and iters < max_iterations:
        current_solution, current_cost = metropolis(current_solution,
                                                    current_cost,
                                                    best_cost,
                                                    temperature,
                                                    int(metropolis_iters))

        if current_cost < best_cost:
            best_solution = current_solution
            best_cost = current_cost

        iters = iters + metropolis_iters
        temperature = temperature * alpha
        metropolis_iters = metropolis_iters * beta

        c_progress.append(current_cost)
        b_progress.append(best_cost)

    return SAResult(
        best_solution,
        best_cost,
        c_progress,
        b_progress
    )

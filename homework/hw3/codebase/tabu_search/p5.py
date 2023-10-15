import numpy as np

from tabu_search.algorithm import TabuSearchResult


def p5_freq_tabu_search(s0, cost,
                        tenure_length=4,
                        max_iters=200,
                        explore_interval=30,
                        freq_multiplier=5,
                        ) -> TabuSearchResult:
    best_soln = current_soln = s0
    best_cost = current_cost = cost(s0)
    tabu_table = np.zeros(8)
    freq_table = np.zeros(8)

    currents = [current_cost]
    bests = [best_cost]

    for i in range(max_iters):
        def score(nbr):  # nbr is a single index
            nbr_path = current_soln.copy()
            nbr_path[nbr] = (nbr_path[nbr] + 1) % 2  # flip the index

            nbr_cost = cost(nbr_path)
            if nbr_cost <= best_cost or i % explore_interval != 0:
                return nbr_cost
            else:
                return nbr_cost + freq_multiplier * freq_table[nbr]

        nbrs = np.arange(8)
        min_nbr = min(nbrs, key=score)
        mutated_soln = current_soln.copy()
        mutated_soln[min_nbr] = (mutated_soln[min_nbr] + 1) % 2

        if tabu_table[min_nbr] < i:  # is not tabu
            current_soln = mutated_soln
            current_cost = cost(current_soln)
            tabu_table[min_nbr] = i + tenure_length
            freq_table[min_nbr] += 1
        elif score(min_nbr) <= best_cost:  # tabu but better than best
            current_soln = mutated_soln
            current_cost = cost(current_soln)
            tabu_table[min_nbr] = i + tenure_length
            freq_table[min_nbr] += 1

        if current_cost <= best_cost:
            best_soln = current_soln
            best_cost = current_cost

        currents.append(current_cost)
        bests.append(best_cost)

    return TabuSearchResult(
        solution=best_soln,
        cost=best_cost,
        progress_current=currents,
        progress_best=bests,
    )

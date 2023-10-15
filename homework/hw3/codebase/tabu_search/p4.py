import numpy as np

from tabu_search.algorithm import TabuSearchResult


def p4_freq_tabu_search(s0, cost,
                        tenure_length=4,
                        nbr_sample_size=3,
                        max_iters=200,
                        explore_interval=30,
                        freq_multiplier=5,
                        ) -> TabuSearchResult:
    best_soln = current_soln = s0
    best_cost = current_cost = cost(s0)
    tabu_table = np.zeros((9, 9), dtype=int)

    currents = [current_cost]
    bests = [best_cost]

    for i in range(max_iters):
        def score(nbr):
            i1, i2 = sorted(nbr)
            nbr_path = current_soln.copy()
            nbr_path[i1], nbr_path[i2] = nbr_path[i2], nbr_path[i1]

            nbr_cost = cost(nbr_path)
            if nbr_cost <= best_cost or i % explore_interval != 0:
                return nbr_cost
            else:
                return nbr_cost + freq_multiplier * tabu_table[i2, i1]

        nbrs = [np.random.choice(9, 2, replace=False) for _ in range(nbr_sample_size)]
        sorted_nbrs = sorted(nbrs, key=score)

        for nbr in sorted_nbrs:
            i1, i2 = sorted(nbr)
            mutated_soln = current_soln.copy()
            mutated_soln[i1], mutated_soln[i2] = mutated_soln[i2], mutated_soln[i1]

            if tabu_table[i1, i2] > i:  # is tabu
                current_soln = mutated_soln
                current_cost = cost(current_soln)
                tabu_table[i1, i2] = i + tenure_length
                break
            elif score(nbr) <= best_cost:  # not tabu but better than best
                current_soln = mutated_soln
                current_cost = cost(current_soln)
                tabu_table[i1, i2] = i + tenure_length
                break

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

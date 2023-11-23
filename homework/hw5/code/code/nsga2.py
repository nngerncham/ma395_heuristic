import dataclasses
from functools import total_ordering
from typing import Set, List, Callable, TypeVar

from ordered_set import OrderedSet


@total_ordering
class Individual:
    def __init__(self, v):
        self.v: List[str] = v  # decision variables
        self.subordinates = set()  # individuals dominated by this individual
        self.dominated_count = 0  # number of individuals dominating this individual
        self.crowding_distance = 0
        self.function_values: List[float] | None = None
        self.rank = None

    def __lt__(self, other):
        return self.rank < other.rank or (self.rank == other.rank and self.crowding_distance > other.crowding_distance)

    def __le__(self, other):
        return self.rank < other.rank or (self.rank == other.rank and self.crowding_distance >= other.crowding_distance)

    def __eq__(self, other):
        return self.v == other.v

    def __gt__(self, other):
        return not (self <= other)

    def __ge__(self, other):
        return not (self < other)


IndividualDerived = TypeVar("IndividualDerived", bound=Individual)
Population = Set[IndividualDerived] | OrderedSet[IndividualDerived]
ObjectiveFunctionApplier = Callable[[Population], None]
ReproductionFunction = Callable[[Population], Population]


@dataclasses.dataclass
class NSGA2Result:
    populations: List[Population]  # big storage
    best_individuals: List[Individual]  # best individuals


def dominates(individual1: Individual, individual2: Individual):
    # i1 dominates i2 means f(i1) <= f(i2)
    m = len(individual1.function_values)
    for i in range(m):
        if individual1.function_values[i] > individual2.function_values[i]:
            return False
    return True


def fast_non_dominated_sort(population: Population):
    frontiers: List[Set[Individual]] = [set()]
    for individual in population:
        individual.subordinates = set()
        individual.dominated_count = 0

        for other_individual in population:
            if dominates(individual, other_individual):
                individual.subordinates.add(other_individual)
            elif dominates(other_individual, individual):
                individual.dominated_count += 1

        if individual.dominated_count == 0:
            individual.rank = 0
            frontiers[0].add(individual)

    i = 0
    current_frontier = frontiers[0]
    while len(current_frontier) > 0:
        next_frontier = set()
        for individual in current_frontier:
            for other_individual in individual.subordinates:
                other_individual.dominated_count -= 1
                if other_individual.dominated_count == 0:
                    other_individual.rank = i + 1
                    next_frontier.add(other_individual)
        frontiers.append(next_frontier)
        current_frontier = next_frontier
        i += 1

    return frontiers


def crowding_distance_assignment(individuals: Population, m_size: int):
    for m in range(m_size):
        sorted_inds = sorted(individuals, key=lambda ind: ind.function_values[m])
        sorted_inds[0].crowding_distance = float('inf')
        sorted_inds[-1].crowding_distance = float('inf')
        fm_max = sorted_inds[-1].function_values[m]
        fm_min = sorted_inds[0].function_values[m]

        for i in range(1, len(sorted_inds) - 1):
            nbr_diff = sorted_inds[i + 1].function_values[m] - sorted_inds[i - 1].function_values[m]
            f_diff = fm_max - fm_min + 1e-6
            sorted_inds[i].crowding_distance += nbr_diff / f_diff


def nsga2(compute_obj: ObjectiveFunctionApplier, pop0: Population, make_new_population: ReproductionFunction, nf: int,
          max_generations: int = 500) -> NSGA2Result:
    population: OrderedSet = OrderedSet(pop0.copy())

    populations = [population]
    best_pops: List[Individual] = [population[0]]

    n = len(population)
    for iters in range(max_generations):
        if len(population) < n:  # too many dupes
            print(iters)
            return NSGA2Result(populations, best_pops)

        offsprings = make_new_population(population)
        total_candidates = population.union(offsprings)  # R_t
        compute_obj(total_candidates)
        frontiers = fast_non_dominated_sort(total_candidates)

        next_population = set()
        i = 0
        while len(next_population) + len(frontiers[i]) <= n:
            if len(frontiers[i]) <= 0:
                break
            crowding_distance_assignment(frontiers[i], nf)
            next_population = next_population.union(frontiers[i])
            i += 1
        sorted_frontier = sorted(frontiers[i])
        next_population = next_population.union(sorted_frontier)
        population = OrderedSet(next_population)[:n]

        populations.append(population)
        best_pops.append(population[0])

    return NSGA2Result(populations, best_pops)

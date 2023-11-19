import dataclasses
from functools import total_ordering
from typing import Set, List, Dict, Callable


@total_ordering
class Individual:
    def __init__(self, v):
        self.v: List[float] = v  # decision variables
        self.dominates = set()  # individuals dominated by this individual
        self.dominated_count = 0  # number of individuals dominating this individual
        self.crowding_distance = None
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


Population = Set[Individual]
ObjectiveFunction = Callable[[Individual], float]
ReproductionFunction = Callable[[Population], Population]
CrossOverFunction = Callable[[Individual, Individual], Individual]


@dataclasses.dataclass
class NSGA2Result:
    population: Set[Individual]


def dominates(fs: List[ObjectiveFunction], individual1: Individual, individual2: Individual):
    # i1 dominates i2 means f(i1) <= f(i2)
    for f in fs:
        if f(individual1) > f(individual2):
            return False
    return True


def fast_non_dominated_sort(fs: List[ObjectiveFunction], population: Population):
    frontiers: Dict[int, Population] = {}  # not the most efficient way to store frontiers but whatever
    for individual in population:
        individual.dominates = set()
        individual.dominated_count = 0

        for other_individual in population:
            if dominates(fs, individual, other_individual):
                individual.dominates.add(other_individual)
            elif dominates(fs, other_individual, individual):
                individual.dominated_count += 1

        if individual.dominated_count == 0:
            individual.rank = 0
            frontiers[0].add(individual)

    i = 0
    while frontiers[i]:
        next_frontier = set()
        for individual in frontiers[i]:
            for other_individual in individual.dominates:
                other_individual.dominated_count -= 1
                if other_individual.dominated_count == 0:
                    other_individual.rank = i + 1
                    next_frontier.add(other_individual)
        frontiers[i] = next_frontier
        i += 1

    return frontiers


def crowding_distance_assignment(fs: List[ObjectiveFunction], individuals: Population):
    for individual in individuals:
        individual.crowding_distance = 0

    for m in range(len(fs)):
        # sort individuals by f
        f = fs[m]
        sorted_individuals = sorted(individuals, key=f)
        sorted_individuals[0].crowding_distance = float('inf')
        sorted_individuals[-1].crowding_distance = float('inf')
        fm_max = sorted_individuals[-1].function_values[m]
        fm_min = sorted_individuals[0].function_values[m]
        for i in range(1, len(sorted_individuals) - 1):
            sorted_individuals[i].crowding_distance += (
                    (f(sorted_individuals[i + 1]) - f(sorted_individuals[i - 1])) / (fm_max - fm_min)
            )


def crowded_comparison(individual1: Individual, individual2: Individual):
    if individual1.rank < individual2.rank:
        return True
    elif individual1.rank == individual2.rank:
        return individual1.crowding_distance > individual2.crowding_distance
    return False


def nsga2(fs: List[ObjectiveFunction], pop0: Population, make_new_population: ReproductionFunction,
          max_generations: int = 500) -> NSGA2Result:
    population = pop0.copy()
    n = len(population)
    for _ in range(max_generations):
        offsprings = make_new_population(population)
        population = population.union(offsprings)  # R_t
        frontiers = fast_non_dominated_sort(fs, population)

        next_population = set()
        i = 0
        while len(next_population) + len(frontiers[i]) <= len(population):
            crowding_distance_assignment(fs, frontiers[i])
            next_population = next_population.union(frontiers[i])
            i += 1

        sorted_frontier = sorted(frontiers[i])
        next_population.union(sorted_frontier[:n - len(next_population)])
    return NSGA2Result(population)

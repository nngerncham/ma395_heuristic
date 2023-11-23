from codebase import nsga2


class P1Point(nsga2.Individual):
    def __init__(self, v, fvs):
        super().__init__(v)
        self.function_values = fvs

    def __hash__(self):
        return hash(self.v[0])

    def __repr__(self):
        return f"{self.v[0]}: {self.function_values}"


p1_population = [
    P1Point(["A"], [1 / 2, 5]),
    P1Point(["B"], [1 / 1, 8]),
    P1Point(["C"], [1 / 2, 9]),
    P1Point(["D"], [1 / 2.5, 4.7]),
    P1Point(["E"], [1 / 2.8, 1.0]),
    P1Point(["F"], [1 / 1.5, 0.2]),
    P1Point(["G"], [1 / 5.6, 9.3]),
    P1Point(["H"], [1 / 4.7, 6.3]),
    P1Point(["I"], [1 / 9.2, 4.5]),
    P1Point(["J"], [1 / 2.7, 8.8]),
]

frontiers = nsga2.fast_non_dominated_sort(p1_population)
for frontier in frontiers:
    print(frontier)

nsga2.crowding_distance_assignment_frontier(frontiers, 2)
for frontier in frontiers:
    print([f"cd({p.v[0]}): {p.crowding_distance}" for p in frontier])


class P2Point(nsga2.Individual):
    def __init__(self, v, fvs):
        super().__init__(v)
        self.function_values = fvs

    def __hash__(self):
        return hash(self.v[0])

    def __repr__(self):
        return f"{self.v[0]}: {self.function_values}"


p1_population = [
    P2Point(["A"], [0.22, 2.35]),
    P2Point(["B"], [3.01, 0.07]),
    P2Point(["C"], [0.67, 1.4]),
    P2Point(["D"], [0.17, 5.83]),
    P2Point(["E"], [10.31, 1.47]),
    P2Point(["F"], [1.62, 10.71]),
    P2Point(["G"], [2.27, 12.31]),
    P2Point(["H"], [3.36, 14.68]),
    P2Point(["I"], [4.67, 17.31]),
    P2Point(["J"], [16.85, 37.27]),
]

frontiers = nsga2.fast_non_dominated_sort(p1_population)
for frontier in frontiers:
    print(frontier)

nsga2.crowding_distance_assignment_frontier(frontiers, 2)
for frontier in frontiers:
    print([f"cd({p.v[0]}): {p.crowding_distance}" for p in frontier])

print()

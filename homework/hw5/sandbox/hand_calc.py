import numpy as np
from numpyarray_to_latex import to_ltx

points = {
    "A": (2, 5),
    "B": (1, 8),
    "C": (2, 9),
    "D": (2.5, 4.7),
    "E": (2.8, 1.0),
    "F": (1.5, 0.2),
    "G": (5.6, 9.3),
    "H": (4.7, 6.3),
    "I": (9.2, 4.5),
    "J": (2.7, 8.8),
}
keys = list(points.keys())

print(sorted(keys, key=lambda x: points[x][0]))
print(sorted(keys, key=lambda x: points[x][1]))

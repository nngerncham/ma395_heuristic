import numpy as np


def gaussian_perturbation(x: np.ndarray, sigma: float = 1) -> np.ndarray:
    d = len(x)
    perturbation = sigma * np.random.normal(size=d)
    return x + perturbation


def uniform_perturbation(x: np.ndarray, sigma: float = 1) -> np.ndarray:
    d = len(x)
    perturbation = np.random.uniform(-sigma, sigma, size=d)
    return x + perturbation

import numpy as np


def sigmoid(x: np.ndarray):
    return 1.0 / (1 + np.exp(-x))


def sigmoid_derivative(x: np.ndarray):
    return x * (1.0 - x)

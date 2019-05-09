import numpy as np


class MathFunctions:

    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
        return x * (1.0 - x)

    @staticmethod
    def transform_into_discrete_values(input: np.ndarray) -> np.ndarray:
        return input / 2 - 1

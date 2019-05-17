import numpy as np
from sklearn.metrics import confusion_matrix


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

    @staticmethod
    def are_values_the_same(predicted_value: float, true_value: float) -> bool:
        a = round(predicted_value)
        b = round(true_value)
        return a == b

    @staticmethod
    def accuracy_total(predicted_values: [], true_values: []) -> float:
        con_mat = confusion_matrix(true_values, predicted_values)
        total_accuracy = (con_mat[0, 0] + con_mat[1, 1]) / float(np.sum(con_mat))
        return total_accuracy

    @staticmethod
    def accuracy_class1(predicted_values: [], true_values: []) -> float:
        con_mat = confusion_matrix(true_values, predicted_values)
        class1_accuracy = (con_mat[0, 0] / float(np.sum(con_mat[0, :])))
        return class1_accuracy

    @staticmethod
    def accuracy_class2(predicted_values: [], true_values: []) -> float:
        con_mat = confusion_matrix(true_values, predicted_values)
        class2_accuracy = (con_mat[1, 1] / float(np.sum(con_mat[1, :])))
        return class2_accuracy




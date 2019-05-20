from src.shared.math_functions import MathFunctions as mf
import numpy as np


class Accuracy:
    def __init__(self, predicted_values: np.ndarray, true_values: np.ndarray):
        self.predicted_values = predicted_values
        self.true_values = true_values

    def accuracy_total(self) -> float:
        accuracy_properly = self.__accuracy_properly_total()
        return accuracy_properly

    def accuracy_class1(self) -> float:
        accuracy_properly = self.__accuracy_properly_class1()
        return accuracy_properly

    def accuracy_class2(self) -> float:
        accuracy_properly = self.__accuracy_properly_class2()
        return accuracy_properly

    def __accuracy_properly_total(self) -> float:
        return mf.accuracy_total(self.predicted_values, self.true_values)

    def __accuracy_properly_class1(self) -> float:
        return mf.accuracy_class1(self.predicted_values, self.true_values)

    def __accuracy_properly_class2(self) -> float:
        return mf.accuracy_class2(self.predicted_values, self.true_values)

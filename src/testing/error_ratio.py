from typing import Tuple
from src.shared.math_functions import MathFunctions as mf
import numpy as np


class ErrorRatio:
    def __init__(self):
        self.__error_number = 0
        self.__all_number = 0

    def update_error_ratio(self, predicted_value: float, true_value: float):
        is_predicted_properly = self.__is_predicted_properly(predicted_value, true_value)
        if not is_predicted_properly:
            self.__error_number += 1
        self.__all_number += 1

    def get_ratio(self) -> Tuple[any, any]:
        return self.__error_number, self.__all_number

    def get_error_ratio(self) -> float:
        return self.__error_number / self.__all_number

    def get_error_number(self) -> float:
        return self.__error_number

    def get_all_number(self) -> int:
        return self.__all_number

    @staticmethod
    def __is_predicted_properly(predicted_value: float, true_value: float) -> bool:
        return mf.are_values_the_same(predicted_value, true_value)

    def accuracy_total(self, predicted_values: np.ndarray, true_values: np.ndarray) -> float:
        accuracy_properly = self.__accuracy_properly_total(predicted_values, true_values)
        return accuracy_properly

    def accuracy_class1(self, predicted_values: np.ndarray, true_values: np.ndarray) -> float:
        accuracy_properly = self.__accuracy_properly_class1(predicted_values, true_values)
        return accuracy_properly

    def accuracy_class2(self, predicted_values: np.ndarray, true_values: np.ndarray) -> float:
        accuracy_properly = self.__accuracy_properly_class2(predicted_values, true_values)
        return accuracy_properly

    @staticmethod
    def __accuracy_properly_total(predicted_values: [], true_values: []) -> float:
        return mf.accuracy_total(predicted_values, true_values)

    @staticmethod
    def __accuracy_properly_class1(predicted_values: [], true_values: []) -> float:
        return mf.accuracy_class1(predicted_values, true_values)

    @staticmethod
    def __accuracy_properly_class2(predicted_values: [], true_values: []) -> float:
        return mf.accuracy_class2(predicted_values, true_values)

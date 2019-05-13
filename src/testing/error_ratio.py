from typing import Tuple
from src.shared.math_functions import MathFunctions as mf


class ErrorRatio:
    def __init__(self):
        self.__error_number = 0
        self.__all_number = 0

    def update_error_ratio(self, predicted_value: float, true_value: float):
        is_predicted_properly = self.__is_predicted_properly(predicted_value, true_value)
        if not is_predicted_properly:
            self.__error_number += 1
        self.__all_number += 1

    def get_ratio(self) -> Tuple[any,any]:
        return self.__error_number, self.__all_number

    def get_error_ratio(self):
        return self.__error_number

    def get_all_number(self):
        return self.__all_number

    @staticmethod
    def __is_predicted_properly(predicted_value: float, true_value: float) -> bool:
        return mf.are_values_the_same(predicted_value, true_value)

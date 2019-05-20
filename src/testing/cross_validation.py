from typing import List

import numpy as np
import math

from src.shared.import_data import ImportData
from src.shared.math_functions import MathFunctions as mf
from src.testing.nn_performance import NNPerformance


class CrossValidation:
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = x
        self.y = y
        self.repeat_time = 5
        self.performances: np.ndarray = np.zeros(self.repeat_time)
        self.variances: np.ndarray = np.zeros(self.repeat_time)
        self.t_value = 0

    def compute_t_value(self):
        self.__prepare_data()
        p = self.performances[0]
        self.t_value = p / math.sqrt(self.variances.sum())
        print('Performances: ', self.performances)
        print('Variances: ', self.variances)
        print('T Value: ', self.t_value)

    def __prepare_data(self):
        for i in range(self.repeat_time):
            print('HELLO')
            nn_per = NNPerformance(self.x, self.y)

            nn_per.compute_performance_difference()
            nn_per.compute_variance_squared()
            self.performances[i] = nn_per.get_performance_diff()
            self.variances[i] = nn_per.get_variance()


if __name__ == "__main__":
    test = ImportData()
    X1: np.ndarray = test.import_all_data()
    y1: np.ndarray = test.import_data(np.array(['Class']))
    y1 = mf.transform_into_discrete_values(y1)
    cv = CrossValidation(X1, y1)
    cv.compute_t_value()

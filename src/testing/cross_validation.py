from typing import Tuple

import numpy as np
import pandas as pd
from random import shuffle

from src.shared.import_data import ImportData
from src.shared.math_functions import MathFunctions as mf
from src.shared.neural_network import NeuralNetwork
from src.testing.error_ratio import ErrorRatio


class CrossValidation:
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = x
        self.y = y

    def compute_performance(self):
        length = self.x.shape[0]
        indexes_test, indexes_training = self.get_indexes(length)
        x_test, x_training = self.split_data(self.x, indexes_test, indexes_training)
        y_test, y_training = self.split_data(self.y, indexes_test, indexes_training)
        error_ratio = ErrorRatio()

        temp_X = np.array([x_training[0]])
        temp_y = np.array([y_training[0]])
        nn = NeuralNetwork(temp_X, temp_y)

        nn.train_network(x_training, y_training)

        for i in range(0, x_test.shape[0]):
            temp_X = np.array([x_test[i]])
            temp_y = np.array([y_test[i]])
            predicted_value, true_value = nn.predict_value(temp_X, temp_y)
            error_ratio.update_error_ratio(predicted_value, true_value)
            print('Iteracja \t', i, 'Prawdziwa wartosc:\t', true_value, 'Estymowana: \t', predicted_value)
        print('Rozpoznano niepoprawnie ', error_ratio.get_error_ratio(), ' na ', error_ratio.get_all_number())


    @staticmethod
    def get_indexes(length: int) -> Tuple[any, any]:
        indexes = list(range(0, length))
        shuffle(indexes)
        indexes_test, indexes_training = CrossValidation.split_list(indexes)
        return indexes_test, indexes_training

    @staticmethod
    def split_list(x: []):
        half = len(x) // 2
        return x[:half], x[half:]

    @staticmethod
    def split_data(data: np.ndarray, indexes_test: [], indexes_training: []) -> Tuple[np.ndarray, np.ndarray]:
        data_test = np.take(data, indexes_test, axis=0)
        data_training = np.take(data, indexes_training, axis=0)
        return data_test, data_training

if __name__ == "__main__":
    test = ImportData()
    X1: np.ndarray = test.import_all_data()
    y1: np.ndarray = test.import_data(np.array(['Class']))
    y1 = mf.transform_into_discrete_values(y1)
    cv = CrossValidation(X1, y1)
    cv.compute_performance()
from random import shuffle
from typing import Tuple
import numpy as np
import math

from src.shared.neural_network import NeuralNetwork
from src.testing.error_ratio import ErrorRatio


class NNPerformance:
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = x
        self.y = y
        self.length = self.x.shape[0]
        self.performance_difference = 0
        self.performance_variance = 0
        self.p1 = self.p2 = 0
        self.__setup_data()

    def compute_performance_difference(self):
        self.p1 = self.__compute_half_performance_diff()
        self.__switch_training_testing()
        self.p2 = self.__compute_half_performance_diff()
        self.performance_difference = (self.p1 + self.p2) / 2.0
        print(self.performance_difference)

    def compute_variance_squared(self):
        square_1 = math.pow(self.p1 - self.performance_difference, 2)
        square_2 = math.pow(self.p2 - self.performance_difference, 2)
        self.performance_variance = square_1 + square_2

    def get_variance(self):
        return self.performance_variance

    def get_performance_diff(self):
        return self.performance_difference

    def __compute_half_performance_diff(self) -> float:
        self.__setup_neural_network()
        self.nn.train_network(self.x_training, self.y_training)

        p1 = self.__compute_performance(self.x_training, self.y_training)
        p2 = self.__compute_performance(self.x_test, self.y_test)

        return abs(p1 - p2)

    def __compute_performance(self, x: np.ndarray, y: np.ndarray):
        error_ratio = ErrorRatio()

        for i in range(0, x.shape[0]):
            temp_X = np.array([x[i]])
            temp_y = np.array([y[i]])
            predicted_value, true_value = self.nn.predict_value(temp_X, temp_y)
            error_ratio.update_error_ratio(predicted_value, true_value)
        return error_ratio.get_error_ratio()

    def __setup_data(self):
        self.indexes_test, self.indexes_training = self.__get_indexes(self.length)
        self.x_test, self.x_training = self.__split_data(self.x, self.indexes_test, self.indexes_training)
        self.y_test, self.y_training = self.__split_data(self.y, self.indexes_test, self.indexes_training)

    def __setup_neural_network(self):
        temp_X = np.array([self.x_training[0]])
        temp_y = np.array([self.y_training[0]])
        self.nn = NeuralNetwork(temp_X, temp_y, add_bias=True)

    def __switch_training_testing(self):
        x_temp: np.ndarray = self.x_test
        self.x_test = self.x_training
        self.x_training = x_temp

        y_temp: np.ndarray = self.y_test
        self.y_test = self.y_training
        self.y_training = y_temp

    def __get_indexes(self, length: int) -> Tuple[any, any]:
        indexes = list(range(0, length))
        shuffle(indexes)
        indexes_test, indexes_training = self.__split_list(indexes)
        return indexes_test, indexes_training

    def __split_list(self, x: []):
        half = len(x) // 2
        return x[:half], x[half:]

    @staticmethod
    def __split_data(data: np.ndarray, indexes_test: [], indexes_training: []) -> Tuple[np.ndarray, np.ndarray]:
        data_test = np.take(data, indexes_test, axis=0)
        data_training = np.take(data, indexes_training, axis=0)
        return data_test, data_training

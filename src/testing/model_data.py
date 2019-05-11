from typing import Tuple
from random import shuffle

import numpy as np


class ModelData:
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = x
        self.y = y

    @staticmethod
    def get_indexes(length: int) -> Tuple[any, any]:
        indexes = list(range(0, length))
        shuffle(indexes)
        indexes_test, indexes_training = ModelData.split_list(indexes)
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

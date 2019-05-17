from typing import Tuple

import numpy as np
import pandas as pd
from random import shuffle
from src.shared.import_data import ImportData
from src.shared.math_functions import MathFunctions as mf
from src.shared.neural_network import NeuralNetwork
from src.testing.error_ratio import ErrorRatio
from src.testing.model_data import ModelData


class CrossValidation:
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = x
        self.y = y

    def compute_performance(self):
        length = self.x.shape[0]
        indexes_test, indexes_training = ModelData.get_indexes(length)
        x_test, x_training = ModelData.split_data(self.x, indexes_test, indexes_training)
        y_test, y_training = ModelData.split_data(self.y, indexes_test, indexes_training)
        error_ratio = ErrorRatio()

        temp_X = np.array([x_training[0]])
        temp_y = np.array([y_training[0]])
        nn = NeuralNetwork(temp_X, temp_y, add_bias=True)

        nn.train_network(x_training, y_training)

        list_true = []
        list_predicted = []

        for i in range(0, x_test.shape[0]):
            temp_X = np.array([x_test[i]])
            temp_y = np.array([y_test[i]])
            predicted_value, true_value = nn.predict_value(temp_X, temp_y)
            error_ratio.update_error_ratio(predicted_value, true_value)
            print(round(true_value))
            print(round(predicted_value))
            list_true.append(round(true_value))
            list_predicted.append(round(predicted_value))


            print('Iteracja \t', i, 'Prawdziwa wartosc:\t', true_value, 'Estymowana: \t', predicted_value)
        print('Rozpoznano niepoprawnie ', error_ratio.get_error_ratio(), ' na ', error_ratio.get_all_number())
        print('Accuracy total %.8f' % error_ratio.accuracy_total(list_predicted, list_true))
        print('Accuracy class1 %.8f' % error_ratio.accuracy_class1(list_predicted, list_true))
        print('Accuracy class2 %.8f' % error_ratio.accuracy_class2(list_predicted, list_true))



if __name__ == "__main__":
    test = ImportData()
    X1: np.ndarray = test.import_all_data()
    y1: np.ndarray = test.import_data(np.array(['Class']))
    y1 = mf.transform_into_discrete_values(y1)
    cv = CrossValidation(X1, y1)
    cv.compute_performance()
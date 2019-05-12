from import_data import ImportData
import numpy as np
from neural_network import NeuralNetwork
from src.shared.math_functions import MathFunctions as mf

if __name__ == '__main__':
    importer = ImportData()
    X1: np.ndarray = importer.cut_columns_from_data(['Mitoses'])
    y1: np.ndarray = importer.import_data(np.array(['Class']))

    y1 = mf.transform_into_discrete_values(y1)

    length = X1.shape[0]

    temp_X = np.array([X1[0]])
    temp_y = np.array([y1[0]])
    print(temp_X.shape, temp_X)
    nn = NeuralNetwork(temp_X, temp_y)

    nn.train_network(X1, y1)

    good_values = 0
    values = 0

    for i in range(length):
        temp_X = np.array([X1[i]])
        temp_y = np.array([y1[i]])
        predicted_value, true_value = nn.predict_value(temp_X, temp_y)
        a = round(predicted_value)
        b = round(true_value)
        if a == b:
            good_values += 1
        values += 1

    print('Rozpoznano poprawnie ', good_values, ' na ', values)


from typing import Tuple
import math

import numpy as np

from src.shared.import_data import ImportData
from src.shared.math_functions import MathFunctions as mf
from src.testing.accuracy import Accuracy


class NeuralNetwork:
    def __init__(self, x: np.ndarray, y: np.ndarray, hidden_layer_inputs=6, add_bias=False):
        self.add_bias = add_bias
        self.hidden_layer_inputs = hidden_layer_inputs

        self.input = x
        self.y = y
        self.output = np.zeros(y.shape)

        print(self.input.shape)

        self.__set_weights()

    def set_new_x_y(self, x: np.ndarray, y: np.ndarray):
        self.input = x
        self.y = y

    def __set_weights(self):
        if self.add_bias:
            add_bias_input = 1
        else:
            add_bias_input = 0

        limit_input_to_layer = 1 / math.sqrt(self.input.shape[1])
        limit_layer_to_output = 1 / math.sqrt(self.hidden_layer_inputs)

        self.weights_input_to_layer = \
            np.random.rand(
                self.input.shape[1] + add_bias_input,
                self.hidden_layer_inputs) * \
            limit_input_to_layer * 2 - limit_input_to_layer

        self.weights_layer_to_output = \
            np.random.rand(
                self.hidden_layer_inputs + add_bias_input,
                1) * \
            limit_layer_to_output * 2 - limit_layer_to_output

    def train_network(self, x_data: np.ndarray, y_data: np.ndarray, repeat_time=200, tries=10):
        for j in range(tries):
            self.__set_weights()
            self.__training_process(x_data, y_data, repeat_time)
            list_predicted, list_true = self.predict_values(x_data, y_data)
            accuracy = Accuracy(list_predicted, list_true)
            if accuracy.is_accuracy_better_than():
                # powtarzamy proces uczenia
                self.__training_process(x_data, y_data, int(repeat_time / 2))
                return
            else:
                self.input = np.array([x_data[0]])
                print('Proba nieudana nr: ', j)
        print('WYLICZONE WAGI NIEZDATNE DO UZYTKU, ZALECANE POWTORZENIE BADAN')

    def __training_process(self, x_data: np.ndarray, y_data: np.ndarray, repeat_time):
        length = x_data.shape[0]
        for j in range(repeat_time):
            for i in range(length):
                temp_x = np.array([x_data[i]])
                temp_yy = np.array([y_data[i]])
                self.set_new_x_y(temp_x, temp_yy)
                self.__feed_forward()
                self.__back_propagation()

    def predict_value(self, x_data: np.ndarray, y_data: np.ndarray) -> Tuple[float, float]:
        self.set_new_x_y(x_data, y_data)
        self.__feed_forward()
        predicted_value = self.output[0, 0]
        true_value = self.y[0, 0]
        return predicted_value, true_value

    def predict_values(self, x_data: np.ndarray, y_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        list_true: np.ndarray = np.empty(x_data.shape[0])
        list_predicted: np.ndarray = np.empty(x_data.shape[0])

        for i in range(0, x_data.shape[0]):
            temp_X = np.array([x_data[i]])
            temp_y = np.array([y_data[i]])
            predicted_value, true_value = self.predict_value(temp_X, temp_y)
            list_true[i] = round(true_value)
            list_predicted[i] = round(predicted_value)

        return list_predicted, list_true

    def __feed_forward(self):
        if self.add_bias:
            self.__feed_forward_bias()
        else:
            self.__feed_forward_no_bias()

    def __feed_forward_no_bias(self):
        self.layer1 = mf.sigmoid(np.dot(self.input, self.weights_input_to_layer))
        self.output = mf.sigmoid(np.dot(self.layer1, self.weights_layer_to_output))

    def __feed_forward_bias(self):

        length = self.input.shape[0]
        one_array = np.ones((length, 1))

        temp_input = np.concatenate((self.input, one_array), axis=1)
        self.input = temp_input
        temp_layer1 = mf.sigmoid(np.dot(temp_input, self.weights_input_to_layer))
        self.layer1 = np.concatenate((temp_layer1, one_array), axis=1)

        self.output = mf.sigmoid(np.dot(self.layer1, self.weights_layer_to_output))

    def __back_propagation(self):
        if self.add_bias:
            self.__back_propagation_bias()
        else:
            self.__back_propagation_no_bias()

    def __back_propagation_no_bias(self):
        diff = self.y - self.output
        loss_derivative_output = 2 * diff * mf.sigmoid_derivative(self.output)
        d_weights_layer_to_output = np.dot(self.layer1.T, loss_derivative_output)

        loss_derivative_layer = np.dot(loss_derivative_output, self.weights_layer_to_output.T) * mf.sigmoid_derivative(
            self.layer1)
        d_weights_input_to_layer = np.dot(self.input.T, loss_derivative_layer)

        self.__update_weights(d_weights_input_to_layer, d_weights_layer_to_output)

    def __back_propagation_bias(self):
        diff = self.y - self.output
        loss_derivative_output = 2 * diff * mf.sigmoid_derivative(self.output)
        d_weights_layer_to_output = np.dot(self.layer1.T, loss_derivative_output)

        loss_derivative_layer = np.dot(loss_derivative_output, self.weights_layer_to_output.T) * mf.sigmoid_derivative(
            self.layer1)
        loss_derivative_layer = np.delete(loss_derivative_layer, loss_derivative_layer.shape[1] - 1, 1)
        d_weights_input_to_layer = np.dot(self.input.T, loss_derivative_layer)

        self.__update_weights(d_weights_input_to_layer, d_weights_layer_to_output)

    def __update_weights(self, d_weights_input_to_layer, d_weights_layer_to_output):
        self.weights_input_to_layer += d_weights_input_to_layer
        self.weights_layer_to_output += d_weights_layer_to_output


if __name__ == "__main__":
    test = ImportData()
    X1: np.ndarray = test.import_all_data()
    y1: np.ndarray = test.import_data(np.array(['Class']))

    y1 = mf.transform_into_discrete_values(y1)

    length = X1.shape[0]

    temp_X = np.array([X1[0]])
    temp_y = np.array([y1[0]])
    print(temp_X.shape, temp_X)
    nn = NeuralNetwork(temp_X, temp_y, add_bias=True)

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
        print('Iteracja \t', i, 'Prawdziwa wartosc:\t', true_value, 'Estymowana: \t', predicted_value)
    print('Rozpoznano poprawnie ', good_values, ' na ', values)

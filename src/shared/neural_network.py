import numpy as np

from src.shared.import_data import ImportData
from src.shared.math_functions import MathFunctions as mf


class NeuralNetwork:
    def __init__(self, x: np.ndarray, y: np.ndarray, hidden_layer_inputs=4, add_bias=False):
        self.input = x
        self.length = self.input.shape[0]
        self.hidden_layer_inputs = hidden_layer_inputs
        print(self.input.shape)
        self.weights_input_to_layer = np.random.rand(self.input.shape[1], self.hidden_layer_inputs)
        self.weights_layer_to_output = np.random.rand(self.hidden_layer_inputs, 1)
        self.y = y
        self.output = np.zeros(y.shape)

    def train_network(self, x_data: np.ndarray, y_data: np.ndarray, repeat_time=300):
        for j in range(repeat_time):
            for i in range(self.length):
                temp_x = np.array([x_data[i]])
                temp_y = np.array([y_data[i]])
                self.set_new_x_y(temp_x, temp_y)
                self.feed_forward()
                self.back_propagation()

    def feed_forward(self):
        self.layer1 = mf.sigmoid(np.dot(self.input, self.weights_input_to_layer))
        self.output = mf.sigmoid(np.dot(self.layer1, self.weights_layer_to_output))

    def back_propagation(self):
        diff = self.y - self.output
        loss_derivative_output = 2 * diff * mf.sigmoid_derivative(self.output)
        d_weights_layer_to_output = np.dot(self.layer1.T, loss_derivative_output)

        loss_derivative_layer = np.dot(loss_derivative_output, self.weights_layer_to_output.T) * mf.sigmoid_derivative(
            self.layer1)
        d_weights_input_to_layer = np.dot(self.input.T, loss_derivative_layer)

        self.__update_weights(d_weights_input_to_layer, d_weights_layer_to_output)

    def set_new_x_y(self, x: np.ndarray, y: np.ndarray):
        self.input = x
        self.y = y

    def __update_weights(self, d_weights_input_to_layer, d_weights_layer_to_output):
        self.weights_input_to_layer += d_weights_input_to_layer
        self.weights_layer_to_output += d_weights_layer_to_output


if __name__ == "__main__":
    # X = np.array([[0,0,1],
    #               [0,1,1],
    #               [1,0,1],
    #               [1,1,1]])
    # y = np.array([[0],[1],[1],[0]])
    # nn = NeuralNetwork(X,y)

    test = ImportData()
    X1: np.ndarray = test.import_all_data()
    y1: np.ndarray = test.import_data(np.array(['Class']))

    # todo rafal problem z zastsosowaniem sigmoidu do obszaru wykraczajacego zbior 0-1
    # potrzebna pomoc

    y1 = mf.transform_into_discrete_values(y1)
    print(y1)

    length = X1.shape[0]

    temp_X = np.array([X1[0]])
    temp_y = np.array([y1[0]])
    print(temp_X.shape, temp_X)
    nn = NeuralNetwork(temp_X, temp_y)

    nn.train_network(X1, y1)

    for i in range(length):
        temp_X = np.array([X1[i]])
        temp_y = np.array([y1[i]])
        nn.set_new_x_y(temp_X, temp_y)
        nn.feed_forward()
        nn.back_propagation()
        print('Iteracja \t', i, 'Prawdziwa wartosc:\t', nn.y[0, 0], 'Estymowana: \t', nn.output[0, 0])

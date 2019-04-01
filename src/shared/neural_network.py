import numpy as np

from src.shared.math_functions import sigmoid, sigmoid_derivative


class NeuralNetwork:
    def __init__(self, x: np.ndarray, y: np.ndarray, hidden_layer_inputs=4, add_bias=False):
        self.input = x
        self.hidden_layer_inputs = hidden_layer_inputs
        self.weights_input_to_layer = np.random.rand(self.input.shape[1], self.hidden_layer_inputs)
        self.weights_layer_to_output = np.random.rand(self.hidden_layer_inputs, 1)
        self.y = y
        self.output = np.zeros(y.shape)

    def feed_forward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights_input_to_layer))
        self.output = sigmoid(np.dot(self.layer1, self.weights_layer_to_output))

    def back_propagation(self):
        diff = self.y - self.output
        loss_derivative_output = 2 * diff * sigmoid_derivative(self.output)
        d_weights_layer_to_output = np.dot(self.layer1.T, loss_derivative_output)

        loss_derivative_layer = np.dot(loss_derivative_output, self.weights_layer_to_output.T) * sigmoid_derivative(self.layer1)
        d_weights_input_to_layer = np.dot(self.input.T, loss_derivative_layer)

        self.__update_weights(d_weights_input_to_layer, d_weights_layer_to_output)

    def set_new_x_y(self, x, y):
        self.input = x
        self.y = y

    def __update_weights(self, d_weights_input_to_layer, d_weights_layer_to_output):
        self.weights_input_to_layer += d_weights_input_to_layer
        self.weights_layer_to_output += d_weights_layer_to_output




if __name__ == "__main__":
    X = np.array([[0,0,1],
                  [0,1,1],
                  [1,0,1],
                  [1,1,1]])
    y = np.array([[0],[1],[1],[0]])
    nn = NeuralNetwork(X,y)

    for i in range(1500):
        nn.feed_forward()
        nn.back_propagation()
        if (i % 45 == 0):
            print('Odczytano: ', i/1500.0*100, '%')

    print(nn.output)
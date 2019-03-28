import numpy as np


class NeuralNetwork:
    def __init__(self, x, y):
        self.input = x
        self.weights = np.random.rand(self.input.shape[1], 4)
        self.y = y
        self.output = np.zeros.shape(y.shape)

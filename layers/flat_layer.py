import numpy as np

class Flatten:
    def __init__(self, input_shape=None):
        self.input_shape=None
        self.output_shape = None
        self.input_data= None
        self.output = None
        self.isbias = False
        self.activation = None
        self.parameters = 0
        self.delta = 0
        self.weights = 0
        self.bias = 0
        self.delta_weights = 0
        self.delta_biases = 0
    def set_output_shape(self):
        self.output_shape = (self.input_shape[0] * self.input_shape[1] * self.input_shape[2])
        self.weights = 0
    def apply_activation(self, x):
        self.input_data = x
        self.output = np.array(self.input_data).flatten()
        return self.output
    def activation_dfn(self, x):
        return x
    def backpropagate(self, nx_layer):
        self.error = np.dot(nx_layer.weights, nx_layer.delta)
        self.delta = self.error * self.activation_dfn(self.out)
        self.delta = self.delta.reshape(self.input_shape)

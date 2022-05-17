import numpy as np

class FFL():
    def __init__(self, input_shape=None, neurons=1, bias=None, weights=None, activation=None, is_bias=True):
        np.random.seed(100)
        self.input_shape = input_shape
        self.neurons = neurons
        self.isbias = is_bias
        self.name = ""
        self.w = weights
        self.b = bias
        if input_shape != None:
            self.output_shape = neurons
        if self.input_shape != None:
            self.weights = weights if weights != None else np.random.randn(self.input_shape, neurons)
            self.parameters = self.input_shape * self.neurons + self.neurons if self.isbias else 0
        if (is_bias):
            self.biases = bias if bias != None else np.random.randn(neurons)
        else:
            self.biases = 0
        self.out = None
        self.input = None
        self.error = None
        self.delta = None
        activations = ["relu", "sigmoid", "tanh", "softmax"]
        self.delta_weights = 0
        self.delta_biases = 0
        self.pdelta_weights = 0
        self.pdelta_biases = 0
        if activation not in activations and activation != None:
            raise ValueError(f"Activation function not recognised. Use one of {activations} instead.")
        else:
            self.activation = activation

    def activation_dfn(self, r):
        """
            A method of FFL to find derivative of given activation function.
        """
        if self.activation is None:
            return np.ones(r.shape)
        if self.activation == 'tanh':
            return 1 - r ** 2
        if self.activation == 'sigmoid':
            r = self.activation_fn(r)
            return r * (1 - r)
        if self.activation == "softmax":
            soft = self.activation_fn(r)
            diag_soft = soft * (1 - soft)
            return diag_soft
        if self.activation == 'relu':
            r[r < 0] = 0
            return r
        return r

    def activation_fn(self, r):
        """
        A method of FFL which contains the operation and defination of given activation function.
        """
        if self.activation == 'relu':
            r[r < 0] = 0
            return r
        if self.activation == None or self.activation == "linear":
            return r
        if self.activation == 'tanh':
            return np.tanh(r)
        if self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-r))
        if self.activation == "softmax":
            r = r - np.max(r)
            s = np.exp(r)
            return s / np.sum(s)

    def apply_activation(self, x):
        soma = np.dot(x, self.weights) + self.biases
        self.out = self.activation_fn(soma)
        return self.out

    def set_n_input(self):
        self.weights = self.w if self.w != None else np.random.normal(size=(self.input_shape, self.neurons))

    def backpropagate(self, nx_layer):
        self.error = np.dot(nx_layer.weights, nx_layer.delta)
        self.delta = self.error * self.activation_dfn(self.out)
        self.delta_weights += self.delta * np.atleast_2d(self.input).T
        self.delta_biases += self.delta

    def set_output_shape(self):
        self.set_n_input()
        self.output_shape = self.neurons
        self.get_parameters()

    def get_parameters(self):
        self.parameters = self.input_shape * self.neurons + self.neurons if self.isbias else 0
        return self.parameters
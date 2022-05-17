import numpy as np

class Conv2d():
    def __init__(self, input_shape=None, filters=1, kernel_size = (3, 3), isbias=True, activation=None, stride=(1, 1), padding="zero", kernel=None, bias=None):
        self.input_shape = input_shape
        self.filters = filters
        self.isbias = isbias
        self.activation = activation
        self.stride = stride
        self.padding = padding
        self.p = 1 if padding != None else 0
        self.bias = bias
        self.kernel = kernel
        if input_shape != None:
            self.kernel_size = (kernel_size[0], kernel_size[1], input_shape[2], filters)
            self.output_shape = (int((input_shape[0] - kernel_size[0] + 2 * self.p) / stride[0]) + 1,
                                int((input_shape[1] - kernel_size[1] + 2 * self.p) / stride[1]) + 1, filters)
            self.set_variables()
            self.out = np.zeros(self.output_shape)
        else:
            self.kernel_size = (kernel_size[0], kernel_size[1])

    def set_variables(self):
        self.weights = self.init_param(self.kernel_size)
        self.biases = self.init_param((self.filters, 1))
        self.parameters = np.multiply.reduce(self.kernel_size) + self.filters if self.isbias else 1
        self.delta_weights = np.zeros(self.kernel_size)
        self.delta_biases = np.zeros(self.biases.shape)

    def init_param(self, size):
        stddev = 1 / np.sqrt(np.prod(size))
        return np.random.normal(loc=0, scale=stddev, size=size)

    def activation_fn(self, r):
        """
        A method of FFL which contains the operation and defination of given activation function.
        """
        if self.activation == None or self.activation == "linear":
            return r
        if self.activation == 'tanh':  # tanh
            return np.tanh(r)
        if self.activation == 'sigmoid':  # sigmoid
            return 1 / (1 + np.exp(-r))
        if self.activation == "softmax":  # stable softmax
            r = r - np.max(r)
            s = np.exp(r)
            return s / np.sum(s)

    def activation_dfn(self, r):
        """
            A method of FFL to find derivative of given activation function.
        """
        if self.activation is None:
            return np.ones(r.shape)
        if self.activation == 'tanh':
            return 1 - r ** 2
        if self.activation == 'sigmoid':
            return r * (1 - r)
        if self.activation == 'softmax':
            soft = self.activation_fn(r)
            return soft * (1 - soft)
        if self.activation == 'relu':
            r[r < 0] = 0
            return r

    def apply_activation(self, image):
        for f in range(self.filters):
            image = self.input
            kshape = self.kernel_size
            if kshape[0] % 2 != 1 or kshape[1] % 2 != 1:
                raise ValueError("Please provide odd length of 2d kernel.")
            if type(self.stride) == int:
                stride = (stride, stride)
            else:
                stride = self.stride
            shape = image.shape
            if self.padding == "zero":
                zeros_h = np.zeros((shape[1], shape[2])).reshape(-1, shape[1], shape[2])
                zeros_v = np.zeros((shape[0] + 2, shape[2])).reshape(shape[0] + 2, -1, shape[2])
                padded_img = np.vstack((zeros_h, image, zeros_h))  # add rows
                padded_img = np.hstack((zeros_v, padded_img, zeros_v))  # add cols
                image = padded_img
                shape = image.shape
            elif self.padding == "same":
                h1 = image[0].reshape(-1, shape[1], shape[2])
                h2 = image[-1].reshape(-1, shape[1], shape[2])
                padded_img = np.vstack((h1, image, h2))  # add rows
                v1 = padded_img[:, 0].reshape(padded_img.shape[0], -1, shape[2])
                v2 = padded_img[:, -1].reshape(padded_img.shape[0], -1, shape[2])
                padded_img = np.hstack((v1, padded_img, v2))  # add cols
                image = padded_img
                shape = image.shape
            elif self.padding == None:
                pass
            rv = 0
            cimg = []
            for r in range(kshape[0], shape[0] + 1, stride[0]):
                cv = 0
                for c in range(kshape[1], shape[1] + 1, stride[1]):
                    chunk = image[rv:r, cv:c]
                    soma = (np.multiply(chunk, self.weights[:, :, :, f]))
                    summa = soma.sum() + self.biases[f]
                    cimg.append(summa)
                    cv += stride[1]
                rv += stride[0]
            cimg = np.array(cimg).reshape(int(rv / stride[0]), int(cv / stride[1]))
            self.out[:, :, f] = cimg
        self.out = self.activation_fn(self.out)
        return self.out

    def backpropagate(self, nx_layer):
        layer = self
        layer.delta = np.zeros((layer.input_shape[0], layer.input_shape[1], layer.input_shape[2]))
        image = layer.input
        for f in range(layer.filters):
            kshape = layer.kernel_size
            shape = layer.input_shape
            stride = layer.stride
            rv = 0
            i = 0
            for r in range(kshape[0], shape[0] + 1, stride[0]):
                cv = 0
                j = 0
                for c in range(kshape[1], shape[1] + 1, stride[1]):
                    chunk = image[rv:r, cv:c]
                    layer.delta_weights[:, :, :, f] += chunk * nx_layer.delta[i, j, f]
                    layer.delta[rv:r, cv:c, :] += nx_layer.delta[i, j, f] * layer.weights[:, :, :, f]
                    j += 1
                    cv += stride[1]
                rv += stride[0]
                i += 1
            layer.delta_biases[f] = np.sum(nx_layer.delta[:, :, f])
        layer.delta = layer.activation_dfn(layer.delta)

    def set_output_shape(self):
        self.output_shape = (int((self.input_shape[0] - self.kernel_size[0] + 2 * self.p) / self.stride[0] + 1),
                             int((self.input_shape[1] - self.kernel_size[1] + 2 * self.p) / self.stride[1] + 1),
                             self.input_shape[2])

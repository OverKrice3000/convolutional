import numpy as np

class Pool2d:
    def __init__(self, kernel_size = (2, 2), stride=None, kind="max", padding=None):
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
        self.padding = padding
        self.p = 1 if padding != None else 0
        self.kernel_size = kernel_size
        if type(stride) == int:
                 stride = (stride, stride)
        self.stride = stride
        if self.stride == None:
            self.stride = self.kernel_size
        self.pools = ['max', "average", 'min']
        if kind not in self.pools:
            raise ValueError("Pool kind not understoood.")
        self.kind = kind

    def set_output_shape(self):
        self.output_shape = (int((self.input_shape[0] - self.kernel_size[0] + 2 * self.p) / self.stride[0] + 1),
                             int((self.input_shape[1] - self.kernel_size[1] + 2 * self.p) / self.stride[1] + 1),
                             self.input_shape[2])

    def apply_activation(self, image):
        stride = self.stride
        kshape = self.kernel_size
        shape = image.shape
        self.input_shape = shape
        self.set_output_shape()
        self.out = np.zeros((self.output_shape))
        for nc in range(shape[2]):
            cimg = []
            rv = 0
            for r in range(kshape[0], shape[0] + 1, stride[0]):
                cv = 0
                for c in range(kshape[1], shape[1] + 1, stride[1]):
                    chunk = image[rv:r, cv:c, nc]
                    if len(chunk) > 0:
                        if self.kind == "max":
                            chunk = np.max(chunk)
                        if self.kind == "min":
                            chunk = np.min(chunk)
                        if self.kind == "average":
                            chunk = np.mean(chunk)
                        cimg.append(chunk)
                    else:
                        cv -= stride[1]
                    cv += stride[1]
                rv += stride[0]
            cimg = np.array(cimg).reshape(int(rv / stride[0]), int(cv / stride[1]))
            self.out[:, :, nc] = cimg
        return self.out

    def backpropagate(self, nx_layer):
        """
            Gradients are passed through index of latest output value .
        """
        layer = self
        stride = layer.stride
        kshape = layer.kernel_size
        image = layer.input
        shape = image.shape
        layer.delta = np.zeros(shape)
        cimg = []
        rstep = stride[0]
        cstep = stride[1]
        for f in range(shape[2]):
            i = 0
            rv = 0
            for r in range(kshape[0], shape[0] + 1, rstep):
                cv = 0
                j = 0
                for c in range(kshape[1], shape[1] + 1, cstep):
                    chunk = image[rv:r, cv:c, f]
                    dout = nx_layer.delta[i, j, f]
                    if layer.kind == "max":
                        p = np.max(chunk)
                        index = np.argwhere(chunk == p)[0]
                        layer.delta[rv + index[0], cv + index[1], f] = dout
                    if layer.kind == "min":
                        p = np.min(chunk)
                        index = np.argwhere(chunk == p)[0]
                        layer.delta[rv + index[0], cv + index[1], f] = dout
                    if layer.kind == "average":
                        p = np.mean(chunk)
                        layer.delta[rv:r, cv:c, f] = dout
                    j += 1
                    cv += cstep
                rv += rstep
                i += 1
import pandas as pd
from optimiser.optimiser import Optimizer
import numpy as np
import time


class CNN():
    def __init__(self):
        self.layers = []
        self.info_df = {}
        self.column = ["LName", "Input Shape", "Output Shape", "Activation", "Bias"]
        self.parameters = []
        self.optimizer = ""
        self.loss = "mse"
        self.lr = 0.01
        self.mr = 0.0001
        self.metrics = []
        self.av_optimizers = ["sgd", "momentum", "adam"]
        self.av_metrics = ["mse", "accuracy", "cse"]
        self.av_loss = ["mse", "cse"]
        self.iscompiled = False
        self.model_dict = None
        self.out = []
        self.eps = 1e-15
        self.train_loss = {}
        self.val_loss = {}
        self.train_acc = {}
        self.val_acc = {}

    def add(self, layer):
        if (len(self.layers) > 0):
            prev_layer = self.layers[-1]
            if prev_layer.name != "Input Layer":
                prev_layer.name = f"{type(prev_layer).__name__}{len(self.layers) - 1}"
            if layer.input_shape == None:
                if type(layer).__name__ == "Flatten":
                    ops = prev_layer.output_shape[:]
                    if type(prev_layer).__name__ == "Pool2d":
                        ops = prev_layer.output_shape[:]
                elif type(layer).__name__ == "Conv2d":
                    ops = prev_layer.output_shape[:]
                    if type(prev_layer).__name__ == "Pool2d":
                        ops = prev_layer.output_shape
                elif type(layer).__name__ == "Pool2d":
                    ops = prev_layer.output_shape[:]
                    if type(prev_layer).__name__ == "Pool2d":
                        ops = prev_layer.output_shape[:]
                else:
                    ops = prev_layer.output_shape
                layer.input_shape = ops
                layer.set_output_shape()
            layer.name = f"Out Layer({type(layer).__name__})"
        else:
            layer.name = "Input Layer"
        if type(layer).__name__ == "Conv2d":
            if (layer.output_shape[0] <= 0 or layer.output_shape[1] <= 0):
                raise ValueError(
                    f"The output shape became invalid [i.e. {layer.output_shape}]. Reduce filter size or increase image size.")
        self.layers.append(layer)
        self.parameters.append(layer.parameters)

    def summary(self):
        lname = []
        linput = []
        loutput = []
        lactivation = []
        lisbias = []
        lparam = []
        for layer in self.layers:
            lname.append(layer.name)
            linput.append(layer.input_shape)
            loutput.append(layer.output_shape)
            lactivation.append(layer.activation)
            lisbias.append(layer.isbias)
            lparam.append(layer.parameters)
        model_dict = {"Layer Name": lname, "Input": linput, "Output Shape": loutput,
                      "Activation": lactivation, "Bias": lisbias, "Parameters": lparam}
        model_df = pd.DataFrame(model_dict).set_index("Layer Name")
        print(model_df)
        print(f"Total Parameters: {sum(lparam)}")

    def train(self, X, Y, epochs, show_every=1, batch_size=32, shuffle=True, val_split=0.1, val_x=None, val_y=None):
        self.check_trainnable(X, Y)
        self.batch_size = batch_size
        t1 = time.time()
        curr_ind = np.arange(0, len(X), dtype=np.int32)
        if shuffle:
            np.random.shuffle(curr_ind)
        if type(val_x) != type(None) and type(val_y) != type(None):
            self.check_trainnable(val_x, val_y)
            print("\nValidation data found.\n")
        else:
            val_ex = int(len(X) * val_split)
            val_exs = []
            while len(val_exs) != val_ex:
                rand_ind = np.random.randint(0, len(X))
                if rand_ind not in val_exs:
                    val_exs.append(rand_ind)
            val_ex = np.array(val_exs)
            val_x, val_y = X[val_ex], Y[val_ex]
            curr_ind = np.array([v for v in curr_ind if v not in val_ex])
        print(f"\nTotal {len(X)} samples.\nTraining samples: {len(curr_ind)} Validation samples: {len(val_x)}.")
        out_activation = self.layers[-1].activation
        batches = []
        len_batch = int(len(curr_ind) / batch_size)
        if len(curr_ind) % batch_size != 0:
            len_batch += 1
        batches = np.array_split(curr_ind, len_batch)
        print(f"Total {len_batch} batches, most batch has {batch_size} samples.\n")
        for e in range(epochs):
            print(e)
            err = []
            for batch in batches:
                a = []
                curr_x, curr_y = X[batch], Y[batch]
                b = 0
                batch_loss = 0
                for x, y in zip(curr_x, curr_y):
                    out = self.feedforward(x)
                    loss, error = self.apply_loss(y, out)
                    batch_loss += loss
                    err.append(error)
                    update = False
                    if b == batch_size - 1:
                        update = True
                        loss = batch_loss / batch_size
                    self.backpropagate(loss, update)
                    b += 1
            if e % show_every == 0:
                train_out = self.predict(X[curr_ind])
                train_loss, train_error = self.apply_loss(Y[curr_ind], train_out)
                val_out = self.predict(val_x)
                val_loss, val_error = self.apply_loss(val_y, val_out)
                if out_activation == "softmax":
                    train_acc = train_out.argmax(axis=1) == Y[curr_ind].argmax(axis=1)
                    val_acc = val_out.argmax(axis=1) == val_y.argmax(axis=1)
                elif out_activation == "sigmoid":
                    train_acc = train_out > 0.7
                    val_acc = val_out > 0.7
                elif out_activation == None:
                    train_acc = abs(Y[curr_ind] - train_out) < 0.000001
                    val_acc = abs(Y[val_ex] - val_out) < 0.000001
                self.train_loss[e] = round(train_error.mean(), 4)
                self.train_acc[e] = round(train_acc.mean() * 100, 4)
                self.val_loss[e] = round(val_error.mean(), 4)
                self.val_acc[e] = round(val_acc.mean() * 100, 4)
                print(f"Epoch: {e}:")
                print(f"Time: {round(time.time() - t1, 3)}sec")
                print(f"Train Loss: {round(train_error.mean(), 4)} Train Accuracy: {round(train_acc.mean() * 100, 4)}%")
                print(f'Val Loss: {(round(val_error.mean(), 4))} Val Accuracy: {round(val_acc.mean() * 100, 4)}% \n')
                t1 = time.time()

    def check_trainnable(self, X, Y):
        if self.iscompiled == False:
            raise ValueError("Model is not compiled.")
        if len(X) != len(Y):
            raise ValueError("Length of training input and label is not equal.")
        if X[0].shape != self.layers[0].input_shape:
            layer = self.layers[0]
            raise ValueError(f"'{layer.name}' expects input of {layer.input_shape} while {X[0].shape[0]} is given.")
        if Y.shape[-1] != self.layers[-1].neurons:
            op_layer = self.layers[-1]
            raise ValueError(f"'{op_layer.name}' expects input of {op_layer.neurons} while {Y.shape[-1]} is given.")

    def compile_model(self, lr=0.01, mr=0.001, opt="adam", loss="mse", metrics=['mse']):
        if opt not in self.av_optimizers:
            raise ValueError(f"Optimizer is not understood, use one of {self.av_optimizers}.")
        for m in metrics:
            if m not in self.av_metrics:
                raise ValueError(f"Metrics is not understood, use one of {self.av_metrics}.")
        if loss not in self.av_loss:
            raise ValueError(f"Loss function is not understood, use one of {self.av_loss}.")
        self.optimizer = opt
        self.loss = loss
        self.lr = lr
        self.mr = mr
        self.metrics = metrics
        self.iscompiled = True
        self.optimizer = Optimizer(layers=self.layers, name=opt, learning_rate=lr, mr=mr)
        self.optimizer = self.optimizer.opt_dict[opt]

    def feedforward(self, x, train=True):
        if train:
            for l in self.layers:
                l.input = x
                x = np.nan_to_num(l.apply_activation(x))
                l.out = x
            return x
        else:
            for l in self.layers:
                l.input = x
                if type(l).__name__ == "Dropout":
                    x = np.nan_to_num(l.apply_activation(x, train=train))
                else:
                    x = np.nan_to_num(l.apply_activation(x))
                l.out = x
            return x

    def apply_loss(self, y, out):
        if self.loss == "mse":
            loss = y - out
            mse = np.mean(np.square(loss))
            return loss, mse
        if self.loss == 'cse':
            """ Requires out to be probability values. """
            if len(out) == len(y) == 1:  # print("Using Binary CSE.")
                cse = -(y * np.log(out) + (1 - y) * np.log(1 - out))
                loss = -(y / out - (1 - y) / (1 - out))
            else:  # print("Using Categorical CSE.")
                if self.layers[-1].activation == "softmax":
                    """if o/p layer's fxn is softmax then loss is y - out
                    check the derivation of softmax and crossentropy with derivative"""
                    loss = y - out
                    loss = loss / self.layers[-1].activation_dfn(out)
                else:
                    y = np.float64(y)
                    out += self.eps
                    loss = -(np.nan_to_num(y / out) - np.nan_to_num((1 - y) / (1 - out)))
                cse = -np.sum((y * np.nan_to_num(np.log(out)) + (1 - y) * np.nan_to_num(np.log(1 - out))))
            return loss, cse

    def backpropagate(self, loss, update):
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            if layer == self.layers[-1]:
                if (type(layer).__name__ == "FFL"):
                    layer.error = loss
                    layer.delta = layer.error * layer.activation_dfn(layer.out)
                    layer.delta_weights += layer.delta * np.atleast_2d(layer.input).T
                    layer.delta_biases += layer.delta
            else:
                nx_layer = self.layers[i + 1]
                layer.backpropagate(nx_layer)
            if update:
                layer.delta_weights /= self.batch_size
                layer.delta_biases /= self.batch_size
        if update:
            self.optimizer(self.layers)
            self.zerograd()

    def zerograd(self):
        for l in self.layers:
            try:
                l.delta_weights = np.zeros(l.delta_weights.shape)
                l.delta_biases = np.zeros(l.delta_biases.shape)
            except:
                pass

    def predict(self, X):
        out = []
        if X.shape != self.layers[0].input_shape:
            for x in X:
                out.append(self.feedforward(x, train=False))
        else:
            out.append(self.feedforward(X, train=False))
        return np.array(out)
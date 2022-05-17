from keras.datasets import mnist
import pandas as pd
from network.network import CNN
from layers.flat_layer import Flatten
from layers.convolutional import Conv2d
from layers.fully_connected import FFL
from layers.pooling import Pool2d

def create_datasets():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x = x_train.reshape(-1, 28 * 28)
    x = (x - x.mean(axis=1).reshape(-1, 1)) / x.std(axis=1).reshape(-1, 1)
    x = x.reshape(-1, 28, 28, 1)
    y = pd.get_dummies(y_train).to_numpy()
    xt = x_test.reshape(-1, 28 * 28)
    xt = (xt - xt.mean(axis=1).reshape(-1, 1)) / xt.std(axis=1).reshape(-1, 1)
    xt = xt.reshape(-1, 28, 28, 1)
    yt = pd.get_dummies(y_test).to_numpy()
    return x, y, xt, yt


if __name__ == '__main__':
    x, y, xt, yt = create_datasets()
    m = CNN()
    print(x)
    m.add(Conv2d(input_shape=(28, 28, 1), filters=8, stride=(5, 5), padding=None, kernel_size=(3, 3), activation=None))
    m.add(Flatten())
    m.add(FFL(neurons=10, activation='softmax'))
    m.compile_model(lr=0.01, opt="adam", loss="cse", mr=0.001)
    m.summary()
    m.train(x[:1000], y[:1000], epochs=100, batch_size=32, val_x=xt[1000:1100], val_y=yt[1000:1100])


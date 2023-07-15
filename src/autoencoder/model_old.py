from keras.layers import Dense, Flatten, Reshape, Input, InputLayer, Conv2D, Conv2DTranspose, MaxPool2D
from keras.models import Sequential, Model
import numpy as np


def build_autoencoder(img_shape, code_size):
    encoder = Sequential()
    encoder.add(InputLayer(img_shape))
    encoder.add(Flatten())
    encoder.add(Dense(code_size))

    decoder = Sequential()
    decoder.add(InputLayer((code_size,)))
    decoder.add(Dense(np.prod(img_shape)))
    decoder.add(Reshape(img_shape))


def build_autoencoder_v3(img_shape, code_size):
    encoder = Sequential()
    encoder.add(InputLayer(img_shape))
    encoder.add(Conv2D(code_size//4, 3, padding="same"))
    encoder.add(MaxPool2D(2))
    encoder.add(Conv2D(code_size//2, 3, padding="same"))
    encoder.add(MaxPool2D(2))
    encoder.add(Conv2D(code_size, 3, padding="same"))
    encoder.add(MaxPool2D(2))

    decoder = Sequential()
    decoder.add(Conv2DTranspose(code_size, 5, padding="same", strides=2))
    decoder.add(Conv2DTranspose(code_size//2, 3, padding="same", strides=2))
    decoder.add(Conv2DTranspose(code_size//4, 2, padding="same", strides=2))
    decoder.add(Conv2DTranspose(3, 3, padding="same"))
    return encoder, decoder


def build_autoencoder_dense(img_shape, code_size):

    # 8 = 2*2*2 da MaxPool2D
    pre_flatten_shape = (img_shape[0] // 8, img_shape[1] // 8, code_size)

    encoder = Sequential()
    encoder.add(InputLayer(img_shape))
    encoder.add(Conv2D(code_size//4, 3, padding="same"))
    encoder.add(MaxPool2D(2))
    encoder.add(Conv2D(code_size//2, 3, padding="same"))
    encoder.add(MaxPool2D(2))
    encoder.add(Conv2D(code_size, 3, padding="same"))
    encoder.add(MaxPool2D(2))

    encoder.add(Flatten())
    encoder.add(Dense(1024, activation='relu'))

    decoder = Sequential()
    decoder.add(InputLayer((1024,)))
    # Mappa lo spazio latente di 1024 alla dimensione pre_flatten
    decoder.add(Dense(np.prod(pre_flatten_shape), activation='relu'))
    # Riorganizza i dati nella forma originale
    decoder.add(Reshape(pre_flatten_shape))

    decoder.add(Conv2DTranspose(code_size, 5, padding="same", strides=2))
    decoder.add(Conv2DTranspose(code_size//2, 3, padding="same", strides=2))
    decoder.add(Conv2DTranspose(code_size//4, 2, padding="same", strides=2))
    decoder.add(Conv2DTranspose(3, 3, activation='sigmoid', padding="same"))
    return encoder, decoder

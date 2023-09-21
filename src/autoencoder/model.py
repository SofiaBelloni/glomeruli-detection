import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Reshape, Conv2D, Conv2DTranspose, MaxPool2D, LeakyReLU, UpSampling2D, BatchNormalization
import numpy as np


def build_simple_autoencoder(img_shape, code_size):
    encoder = tf.keras.Sequential([
        Flatten(input_shape=img_shape),
        Dense(code_size)
    ])

    decoder = tf.keras.Sequential([
        Dense(np.prod(img_shape), input_shape=(code_size,)),
        Reshape(img_shape)
    ])

    return encoder, decoder


def build_cnn_autoencoder(img_shape, code_size):
    encoder = tf.keras.Sequential([
        Conv2D(code_size//4, 3, padding="same", input_shape=img_shape),
        MaxPool2D(2),
        Conv2D(code_size//2, 3, padding="same"),
        MaxPool2D(2),
        Conv2D(code_size, 3, padding="same"),
        MaxPool2D(2)
    ])

    decoder = tf.keras.Sequential([
        Conv2DTranspose(code_size, 5, padding="same", strides=2, input_shape=(
            img_shape[0]//8, img_shape[1]//8, code_size)),
        Conv2DTranspose(code_size//2, 3, padding="same", strides=2),
        Conv2DTranspose(code_size//4, 2, padding="same", strides=2),
        Conv2DTranspose(3, 3, activation='sigmoid', padding="same")
    ])

    return encoder, decoder


def build_cnn_dense_autoencoder(img_shape, code_size=1024):
    pre_flatten_shape = (img_shape[0] // 8, img_shape[1] // 8, code_size)

    encoder = tf.keras.Sequential([
        Conv2D(code_size//4, 3, padding="same",
               input_shape=img_shape, activation='relu'),
        MaxPool2D(2),
        Conv2D(code_size//2, 3, padding="same", activation='relu'),
        MaxPool2D(2),
        Conv2D(code_size, 3, padding="same", activation='relu'),
        MaxPool2D(2),
        Flatten(),
        Dense(code_size, activation='relu')
    ])

    decoder = tf.keras.Sequential([
        Dense(np.prod(pre_flatten_shape),
              activation='relu', input_shape=(code_size,)),
        Reshape(pre_flatten_shape),
        Conv2DTranspose(code_size, 5, padding="same",
                        strides=2, activation='relu'),
        Conv2DTranspose(code_size//2, 3, padding="same",
                        strides=2, activation='relu'),
        Conv2DTranspose(code_size//4, 2, padding="same",
                        strides=2, activation='relu'),
        Conv2DTranspose(3, 3, activation='sigmoid', padding="same")
    ])

    return encoder, decoder




def build_cnn_v6_autoencoder(img_shape, code_size=1024):
    pre_flatten_shape = (img_shape[0] // 8, img_shape[1] // 8, code_size)
    
    # Encoder
    encoder = tf.keras.Sequential([
        Conv2D(code_size//4, 3, padding="same", input_shape=img_shape),
        LeakyReLU(alpha=0.1),
        MaxPool2D(2),
        Conv2D(code_size//2, 3, padding="same"),
        LeakyReLU(alpha=0.1),
        MaxPool2D(2),
        Conv2D(code_size, 3, padding="same"),
        LeakyReLU(alpha=0.1),
        MaxPool2D(2),
        Flatten(),
        Dense(code_size),
        LeakyReLU(alpha=0.1)
    ])

    # Decoder
    decoder = tf.keras.Sequential([
        Dense(np.prod(pre_flatten_shape), input_shape=(code_size,)),
        LeakyReLU(alpha=0.1),
        Reshape(pre_flatten_shape),
        UpSampling2D((2,2)),
        Conv2D(code_size, 3, padding='same'),
        LeakyReLU(alpha=0.1),
        UpSampling2D((2,2)),
        Conv2D(code_size//2, 3, padding='same'),
        LeakyReLU(alpha=0.1),
        UpSampling2D((2,2)),
        Conv2D(code_size//4, 3, padding='same'),
        LeakyReLU(alpha=0.1),
        Conv2D(3, 3, activation='sigmoid', padding="same")
    ])

    return encoder, decoder


def build_autoencoder(img_shape, code_size=1024):
    pre_flatten_shape = (img_shape[0] // 8, img_shape[1] // 8, code_size)
    leaky_relu = LeakyReLU()

    encoder = tf.keras.Sequential([
        Conv2D(code_size//4, 3, padding="same", input_shape=img_shape),
        BatchNormalization(),
        leaky_relu,
        MaxPool2D(2),

        Conv2D(code_size//2, 3, padding="same"),
        BatchNormalization(),
        leaky_relu,
        MaxPool2D(2),

        Conv2D(code_size, 3, padding="same"),
        BatchNormalization(),
        leaky_relu,
        MaxPool2D(2),

        Flatten(),
        Dense(code_size),
        leaky_relu
    ])

    decoder = tf.keras.Sequential([
        Dense(np.prod(pre_flatten_shape), input_shape=(code_size,)),
        leaky_relu,
        Reshape(pre_flatten_shape),

        Conv2DTranspose(code_size, 5, padding="same", strides=2),
        BatchNormalization(),
        leaky_relu,

        Conv2DTranspose(code_size//2, 3, padding="same", strides=2),
        BatchNormalization(),
        leaky_relu,

        Conv2DTranspose(code_size//4, 2, padding="same", strides=2),
        BatchNormalization(),
        leaky_relu,

        Conv2DTranspose(3, 3, activation='sigmoid', padding="same")
    ])

    return encoder, decoder

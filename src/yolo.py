import tensorflow as tf
from keras.models import Model
from keras.layers.convolutional import Conv2D
from tensorflow.keras.layers import BatchNormalization, GlobalAveragePooling2D, Dense, LeakyReLU
from layers import SkipConnection

NUM_CLASSES = 2
INPUT_SHAPE = (512, 512, 3)
IMAGE_SIZE = (512, 512)


class Yolo(Model):
    def __init__(self, num_class=NUM_CLASSES, input_shape=INPUT_SHAPE):
        super().__init__()
        self.conv1 = Conv2D(32, padding='same', kernel_size=(3, 3), strides=1)
        self.conv2 = Conv2D(64, padding='same', kernel_size=(3, 3), strides=2)
        self.res3 = SkipConnection(filt=32)
        self.conv4 = Conv2D(128, padding='same', kernel_size=(3, 3), strides=2)
        self.res5 = SkipConnection(filt=64)
        self.conv6 = Conv2D(256, padding='same', kernel_size=(3, 3), strides=2)
        self.re7 = SkipConnection(filt=128)
        self.conv8 = Conv2D(512, padding='same', kernel_size=(3, 3), strides=2)
        self.res9 = SkipConnection(filt=256)
        self.conv10 = Conv2D(1024, padding='same',
                             kernel_size=(3, 3), strides=2)
        self.res11 = SkipConnection(filt=512)
        self.avgp12 = GlobalAveragePooling2D()
        self.dense = Dense(1024)
        self.act = LeakyReLU(alpha=0.1)
        self.bathc = BatchNormalization()
        self.out = Dense(NUM_CLASSES, activation="sigmoid")
        
    def call(self, input):
        pass

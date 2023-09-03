import tensorflow as tf
from keras.models import Model
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.merging import concatenate


class Unet(Model):

    def __init__(self, dropout_value=0.3, num_classes=2, input_shape=(512, 512, 3)):
        super().__init__()
        vgg19 = tf.keras.applications.vgg19.VGG19(
            include_top=False,   # Exclusion of the last 3 layers
            weights='imagenet',
            # input_tensor=None,
            input_shape=input_shape,
            pooling='max',
            classes=num_classes,
            classifier_activation='relu'
        )
        # for layer in vgg19.layers:
        #  layer.trainable = False
        # Block 1
        self.b1c1 = vgg19.get_layer('block1_conv1')
        self.b1c2 = vgg19.get_layer('block1_conv2')
        # Block 2
        self.b2p = vgg19.get_layer('block1_pool')
        self.b2c1 = vgg19.get_layer('block2_conv1')
        self.b2c2 = vgg19.get_layer('block2_conv2')
        # Block 3
        self.b3p = vgg19.get_layer('block2_pool')
        self.b3c1 = vgg19.get_layer('block3_conv1')
        self.b3c2 = vgg19.get_layer('block3_conv2')
        # Block 4
        self.b4p = vgg19.get_layer('block3_pool')
        self.b4c1 = vgg19.get_layer('block4_conv1')
        self.b4c2 = vgg19.get_layer('block4_conv2')
        # Block 5
        self.b5p = vgg19.get_layer('block4_pool')
        self.b5c1 = vgg19.get_layer('block5_conv1')
        self.b5c2 = vgg19.get_layer('block5_conv2')
        # Block 6
        # self.b6d1 = Dropout(dropout_value) #Controllando meglio su internet, sembrerebbe che il dropout non è presente nativamente su unet, ma è possibile inserirlo qualora si osservi dell'overfitting
        self.b6p = vgg19.get_layer('block5_pool')
        self.b6c1 = Conv2D(filters=1024, activation='relu', kernel_size=(
            3, 3), kernel_initializer='he_normal', padding='same')
        self.b6c2 = Conv2D(filters=1024, activation='relu', kernel_size=(
            3, 3), kernel_initializer='he_normal', padding='same')
        # self.b6d2 = Dropout(dropout_value)
        self.b6u = Conv2DTranspose(
            512, (3, 3), activation="relu", strides=(2, 2), padding='same')
        # Block 7
        # After concatenate
        self.b7c1 = Conv2D(filters=512, activation='relu', kernel_size=(
            3, 3), kernel_initializer='he_normal', padding='same')
        self.b7c2 = Conv2D(filters=512, activation='relu', kernel_size=(
            3, 3), kernel_initializer='he_normal', padding='same')
        self.b7u = Conv2DTranspose(
            512, (3, 3), activation="relu", strides=(2, 2), padding='same')
        # Block 8
        # After concatenate
        self.b8c1 = Conv2D(filters=512, activation='relu', kernel_size=(
            3, 3), kernel_initializer='he_normal', padding='same')
        self.b8c2 = Conv2D(filters=512, activation='relu', kernel_size=(
            3, 3), kernel_initializer='he_normal', padding='same')
        self.b8u = Conv2DTranspose(
            256, (3, 3), activation="relu", strides=(2, 2), padding='same')
        # Block 9
        # After concatenate
        self.b9c1 = Conv2D(filters=256, activation='relu', kernel_size=(
            3, 3), kernel_initializer='he_normal', padding='same')
        self.b9c2 = Conv2D(filters=256, activation='relu', kernel_size=(
            3, 3), kernel_initializer='he_normal', padding='same')
        self.b9u = Conv2DTranspose(
            128, (3, 3), activation="relu", strides=(2, 2), padding='same')
        # Block 10
        # After concatenate
        self.b10c1 = Conv2D(filters=128, activation='relu', kernel_size=(
            3, 3), kernel_initializer='he_normal', padding='same')
        self.b10c2 = Conv2D(filters=128, activation='relu', kernel_size=(
            3, 3), kernel_initializer='he_normal', padding='same')
        self.b10u = Conv2DTranspose(
            64, (3, 3), activation="relu", strides=(2, 2), padding='same')
        # Block 11
        # After concatenate
        self.b11c1 = Conv2D(filters=64, activation='relu', kernel_size=(
            3, 3), kernel_initializer='he_normal', padding='same')
        self.b11c2 = Conv2D(filters=64, activation='relu', kernel_size=(
            3, 3), kernel_initializer='he_normal', padding='same')
        self.b11c3 = Conv2D(filters=64, activation='relu', kernel_size=(
            3, 3), kernel_initializer='he_normal', padding='same')
        self.b11s = Conv2D(2, (1, 1), activation='softmax')

    def call(self, input):
        # Block 1
        r1 = self.b1c1(input)
        r1 = self.b1c2(r1)
        # Block 2
        r2 = self.b2p(r1)
        r2 = self.b2c1(r2)
        r2 = self.b2c2(r2)
        # Block 3
        r3 = self.b3p(r2)
        r3 = self.b3c1(r3)
        r3 = self.b3c2(r3)
        # Block 4
        r4 = self.b4p(r3)
        r4 = self.b4c1(r4)
        r4 = self.b4c2(r4)
        # Block 5
        r5 = self.b5p(r4)
        r5 = self.b5c1(r5)
        r5 = self.b5c2(r5)
        # Block 6
        # r6 = self.b6d1(r5)
        r6 = self.b6p(r5)
        r6 = self.b6c1(r6)
        r6 = self.b6c2(r6)
        # r6 = self.b6d2(r6)
        r6 = self.b6u(r6)
        # Block 7
        r7 = concatenate([r6, r5])
        r7 = self.b7c1(r7)
        r7 = self.b7c2(r7)
        r7 = self.b7u(r7)
        # Block 8
        r8 = concatenate([r7, r4])
        r8 = self.b8c1(r8)
        r8 = self.b8c2(r8)
        r8 = self.b8u(r8)
        # Block 9
        r9 = concatenate([r8, r3])
        r9 = self.b9c1(r9)
        r9 = self.b9c2(r9)
        r9 = self.b9u(r9)
        # Block 10
        r10 = concatenate([r9, r2])
        r10 = self.b10c1(r10)
        r10 = self.b10c2(r10)
        r10 = self.b10u(r10)
        # Block 11
        r11 = concatenate([r10, r1])
        r11 = self.b11c1(r11)
        r11 = self.b11c2(r11)
        r11 = self.b11c3(r11)
        out = self.b11s(r11)
        return out

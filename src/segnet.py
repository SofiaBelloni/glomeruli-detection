import tensorflow as tf
from data_augmentation import data_augment as da
from tensorflow.keras.layers import BatchNormalization, MaxUnpooling2D, BatchNormalization, Conv2D

NUM_CLASSES = 2
INPUT_SHAPE = (400, 400, 3)


class SegNet(tf.keras.models.Model):
    def __init__(self):
        self.da = da()
        vgg19 = tf.keras.applications.vgg19.VGG19(
            include_top=False,   # Exclusion of the last 3 layers
            weights='imagenet',
            input_tensor=None,
            input_shape=INPUT_SHAPE,
            pooling='max',
            classes=NUM_CLASSES,
            classifier_activation='relu'
        )
        self.encoder =  tf.keras.Sequential([
            vgg19.get_layer('input_1'),
            BatchNormalization(),
            vgg19.get_layer('block1_conv1'),
            BatchNormalization(),
            vgg19.get_layer('block1_conv2'),
            BatchNormalization(),
            vgg19.get_layer('block1_pool'),
            vgg19.get_layer('block2_conv1'),
            BatchNormalization(),
            vgg19.get_layer('block2_conv2'),
            BatchNormalization(),
            vgg19.get_layer('block2_pool'),
            vgg19.get_layer('block3_conv1'),
            BatchNormalization(),
            vgg19.get_layer('block3_conv2'),
            BatchNormalization(),
            vgg19.get_layer('block3_conv3'),
            BatchNormalization(),
            vgg19.get_layer('block3_conv4'),
            BatchNormalization(),
            vgg19.get_layer('block3_pool'),
            vgg19.get_layer('block4_conv1'),
            BatchNormalization(),
            vgg19.get_layer('block4_conv2'),
            BatchNormalization(),
            vgg19.get_layer('block4_conv3'),
            BatchNormalization(),
            vgg19.get_layer('block4_conv4'),
            BatchNormalization(),
            vgg19.get_layer('block4_pool')
        ])
        
        self.decoder = tf.keras.Sequential([
            # Block 5
            MaxUnpooling2D(size=(2, 2), stride = 2),
            Conv2D(512, (3, 3), activation='relu', padding='same'), #TODO check padding
            BatchNormalization(),
            Conv2D(512, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(512, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(512, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            #Block 4
            MaxUnpooling2D(size=(2, 2), stride = 2),
            Conv2D(512, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(512, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(512, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(512, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            #Block 3
            MaxUnpooling2D(size=(2, 2), stride = 2),
            Conv2D(256, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(256, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(256, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(256, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            #Block 2
            MaxUnpooling2D(size=(2, 2), stride = 2),
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            #Block 1
            MaxUnpooling2D(size=(2, 2), stride = 2),
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            #Softmax
            Conv2D(NUM_CLASSES, (1, 1), activation='softmax', padding='same'),
        ])

    def call(self, input, training=False):
        if training:
            # Qui dovrei separare image e label da inputs per fara data augment con imgaug
            dataset, labels = da(
                image=self.dataset, segmentation_maps=self.labels)
        pass

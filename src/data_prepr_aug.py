import imgaug.augmenters as iaa
import tensorflow as tf
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from sklearn.model_selection import train_test_split
import numpy as np


def data_augment():
    return iaa.Sequential([
        iaa.Dropout((0, 0.05)),  # Remove random pixel
        iaa.Affine(rotate=(-30, 30)),  # Rotate between -30 and 30 degreed
        iaa.Fliplr(0.5),  # Flip with 0.5 probability
        iaa.Crop(percent=(0, 0.2), keep_size=True),  # Random crop
        # Add -50 to 50 to the brightness-related channels of each image
        iaa.WithBrightnessChannels(iaa.Add((-50, 50))),
        # Change images to grayscale and overlay them with the original image by varying strengths, effectively removing 0 to 50% of the color
        iaa.Grayscale(alpha=(0.0, 0.5)),
        # Add random value to each pixel
        iaa.GammaContrast((0.5, 2.0), per_channel=True),
        # Local distortions of images by moving points around
        iaa.PiecewiseAffine(scale=(0.01, 0.1)),
    ], random_order=True)


def process_data(image, label):
    return tf.cast(image, tf.float32)/255, tf.one_hot(label, 2, name="label", axis=-1)


def data_aug_impl(dataset, image_train, label_train):
    da = data_augment()
    segmented_label_train = [SegmentationMapsOnImage(
        label, shape=dataset[1].shape) for label in label_train]
    image_train_copy = image_train.copy()
    for _ in range(1):
        augmented_images, augmented_labels = da(
            images=image_train_copy, segmentation_maps=segmented_label_train)
        image_train = np.append(image_train, augmented_images, axis=0)
        label_train = np.append(label_train, np.array(
            [label.get_arr() for label in augmented_labels]), axis=0)

    return image_train, label_train


def generate_train_data_tensor(image_train, label_train):
    train_data = tf.data.Dataset.from_tensor_slices((image_train, label_train))
    train_data = train_data.map(
        process_data, num_parallel_calls=tf.data.AUTOTUNE)
    train_data = train_data.cache()
    train_data = train_data.shuffle(100)
    train_data = train_data.batch(128)
    train_data = train_data.prefetch(tf.data.AUTOTUNE)
    return train_data

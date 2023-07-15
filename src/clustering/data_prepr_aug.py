import cv2
import imgaug.augmenters as iaa
import tensorflow as tf
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from sklearn.model_selection import train_test_split
import numpy as np
from extrapolate_glomeruli import get_glomeruli, get_glomeruli_labels, get_no_glomeruli, get_no_glomeruli_labels


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


def data_aug_impl(shape_dataset, image_train, label_train):
    da = data_augment()
    segmented_label_train = [SegmentationMapsOnImage(
        label, shape=shape_dataset) for label in label_train]
    image_train_copy = image_train.copy()
    augmented_images, augmented_labels = da(
        images=image_train_copy, segmentation_maps=segmented_label_train)
    image_train = np.append(image_train, augmented_images, axis=0)
    label_train = np.append(label_train, np.array(
        [label.get_arr() for label in augmented_labels]), axis=0)
    return image_train, label_train


def data_aug_no_impl(shape_dataset, image_train, label_train):
    da = data_augment()
    segmented_label_train = [SegmentationMapsOnImage(
        label, shape=shape_dataset) for label in label_train]
    image_train_copy = image_train.copy()
    augmented_images, augmented_labels = da(
        images=image_train_copy, segmentation_maps=segmented_label_train)
    return augmented_images, np.array([label.get_arr() for label in augmented_labels])


# def process_data(image, label):
#     return tf.cast(image, tf.float32)/255, tf.one_hot(label, 2, name="label", axis=-1)


def resize_images(images):
    return np.array([cv2.resize(img, dsize=(200, 200), interpolation=cv2.INTER_CUBIC) for img in images])


def augment_and_label_data(images, labels, label_value):
    augmented_images, _ = data_aug_no_impl(labels[0].shape, images, labels)
    return augmented_images, np.ones(augmented_images.shape[0]) * label_value


def process_data(image, label):
    return tf.cast(image, tf.float32)/255, label


def create_tf_data(images, labels, buffer_size, batch_size):
    data = tf.data.Dataset.from_tensor_slices((images, labels))
    data = data.map(process_data, num_parallel_calls=tf.data.AUTOTUNE)
    # return data.shuffle(buffer_size=buffer_size).batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)
    return data


def generate_train_data(dataset, labels, buffer_size=1024, batch_size=32):
    # Generate and preprocess glomeruli data
    glomeruli = get_glomeruli(dataset, labels)
    glomeruli_labels = get_glomeruli_labels(dataset, labels)
    resized_glomeruli = resize_images(glomeruli)
    resized_glomeruli_labels = resize_images(glomeruli_labels)
    train_glomeruli, train_glomeruli_labels = augment_and_label_data(
        resized_glomeruli, resized_glomeruli_labels, label_value=1)

    # Generate and preprocess no_glomeruli data
    no_glomeruli = get_no_glomeruli(dataset, labels)
    no_glomeruli_labels = get_no_glomeruli_labels(dataset, labels)
    resized_no_glomeruli = resize_images(no_glomeruli)
    resized_no_glomeruli_labels = resize_images(no_glomeruli_labels)
    train_no_glomeruli, train_no_glomeruli_labels = augment_and_label_data(
        resized_no_glomeruli, resized_no_glomeruli_labels, label_value=0)

    # Combine data and labels
    data_train = np.append(train_glomeruli, train_no_glomeruli, axis=0)
    label_train = np.append(train_glomeruli_labels,
                            train_no_glomeruli_labels, axis=0)
    # Return a TensorFlow Dataset
    # return create_tf_data(data_train, label_train, buffer_size, batch_size)
    return data_train, label_train

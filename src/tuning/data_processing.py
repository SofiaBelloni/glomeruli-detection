import numpy as np
import cv2


def get_image_with_glomeruli(dataset, labels):
    return np.array([dataset[i] for i in range(len(dataset)) if (1 in labels[i])])


def get_image_with_glomeruli_labels(dataset, labels):
    return np.array([labels[i] for i in range(len(dataset)) if (1 in labels[i])])


def get_image_with_no_glomeruli(dataset, labels):
    return np.array([dataset[i] for i in range(len(dataset)) if not (1 in labels[i])])


def get_image_with_no_glomeruli_labels(dataset, labels):
    return np.array([labels[i] for i in range(len(dataset)) if not (1 in labels[i])])


def shuffle(images, labels):
    shuffled_indices = np.arange(len(images))
    np.random.shuffle(shuffled_indices)
    images = images[shuffled_indices]
    labels = labels[shuffled_indices]

    return images, labels


def resize_images(images, size=(200, 200)):
    return np.array([cv2.resize(img, dsize=size, interpolation=cv2.INTER_CUBIC) for img in images])

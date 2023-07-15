import cv2
import numpy as np
import tensorflow as tf
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage


def glomeruli_crop(glomeruli, glomeruli_labels):
    result = []
    original_images = []
    original_labels = []
    result_labels = []

    for j in range(0, len(glomeruli)):
        num_labels, labels = cv2.connectedComponents(
            glomeruli_labels[j], connectivity=8)
        for i in range(1, num_labels):
            image = glomeruli[j]

            # Find row and column indices of pixels with value 1
            yrow, xcol = np.where(labels == i)

            # Calculate minimum and maximum row indices
            yrowmin = np.min(yrow)
            yrowmax = np.max(yrow)

            # Calculate minimum and maximum column indices
            xcolmin = np.min(xcol)
            xcolmax = np.max(xcol)

            # Calculate the size of the side of the square as the max between the height and the width
            side = max(yrowmax - yrowmin + 1, xcolmax - xcolmin + 1)
            if (side >= 20):
                # Adjust the bounding box to be square
                yrowmin = max(
                    0, yrowmin - (side - (yrowmax - yrowmin + 1)) // 2)
                yrowmax = yrowmin + side
                if yrowmax > image.shape[0]:
                    yrowmax = image.shape[0]
                    yrowmin = yrowmax - side
                xcolmin = max(
                    0, xcolmin - (side - (xcolmax - xcolmin + 1)) // 2)
                xcolmax = xcolmin + side
                if xcolmax > image.shape[1]:
                    xcolmax = image.shape[1]
                    xcolmin = xcolmax - side

                # Extract the square image from the original image
                cropped_image = image[yrowmin:yrowmax, xcolmin:xcolmax]

                # Calculate the proportion of glomerulus pixels in the cropped image
                cropped_label = labels[yrowmin:yrowmax, xcolmin:xcolmax] == i
                glomerulus_proportion = np.mean(cropped_label)

                # Only keep the image if the glomerulus proportion is above the threshold
                if glomerulus_proportion >= 0.6:
                    # Resize image
                    resized_image = cv2.resize(
                        cropped_image, (200, 200), interpolation=cv2.INTER_AREA)
                    cropped_label = labels[yrowmin:yrowmax, xcolmin:xcolmax]
                    resized_label = cv2.resize(
                        cropped_label, (200, 200), interpolation=cv2.INTER_NEAREST)

                    result.append(resized_image)
                    result_labels.append(resized_label)
                    original_images.append(image)
                    original_labels.append(labels)

    return np.array(result), np.array(result_labels), np.array(original_images), np.array(original_labels)


def process_data(image):
    return tf.cast(image, tf.float32)/255, tf.cast(image, tf.float32)/255


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


def data_aug_impl_no_label(image_train, n=1):
    da = data_augment()
    image_train_copy = image_train.copy()
    for i in range(n):
        augmented_images = da(
            images=image_train_copy)
        image_train = np.append(image_train, augmented_images, axis=0)
    return image_train

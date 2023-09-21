import numpy as np
import cv2

dataset_path = '../../dataset512x512/image_train_augmented_balanced.npy'
label_path = '../../dataset512x512/label_train_augmented_balanced.npy'

path_to_save_dataset = '../../dataset512x512/image_train_augmented_balanced_cropped.npy'
path_to_save_label = '../../dataset512x512/label_train_augmented_balanced_cropped.npy'

print("Test set")

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

def glomeruli_crop_dark_backgroung_super_cool(glomeruli, glomeruli_labels):
    result = []
    result_labels = []

    glomeruli_labels[glomeruli_labels != 0] = 1

    for i in range(0, len(glomeruli)):
        # temp_label = np.where(glomeruli_labels[i] == i, 1, 0)
        # Apply the mask to the image
        result_image = cv2.merge([channel * glomeruli_labels[i]
                                 for channel in cv2.split(glomeruli[i])])
        result.append(result_image)
        result_labels.append(glomeruli_labels[i])

    return np.array(result), np.array(result_labels)

ds = np.load(dataset_path)
lb = np.load(label_path)

#ds = (ds * 255).astype(np.uint8)
#lb = np.argmax(np.eye(lb.shape[-1])[np.argmax(lb, axis=-1)], axis=-1).astype(np.uint8)

print(f"Dataset shape {ds.shape} \nLabels shape {lb.shape}")

ds_c, lb_c, _, _ = glomeruli_crop(ds, lb)
print(f"Dataset shape cropped {ds_c.shape} \nLabels shape cropped {lb_c.shape}")

#ds_d, lb_d = glomeruli_crop_dark_backgroung_super_cool(ds_c, lb_c)
#print(f"Dataset shape cropped dark {ds_d.shape} \nLabels shape cropped dark {lb_d.shape}")

np.save(path_to_save_dataset, ds_c)
np.save(path_to_save_label, lb_c)


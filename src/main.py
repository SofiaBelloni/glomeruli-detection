import numpy as np
from sklearn.model_selection import train_test_split
from data_prepr_aug import data_aug_impl, generate_train_data_tensor
from segnet import SegNet
import tensorflow as tf
from utils import plt_history

# The path can also be read from a config file, etc.
import os
if hasattr(os, 'add_dll_directory'):
    # Python >= 3.8 on Windows
    OPENSLIDE_PATH = r'C:\Users\sofia\openslide-win64-20230414\openslide-win64-20230414\bin'
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide

tf.compat.v1.disable_eager_execution()

path_to_dataset = "../dataset/slides/dataset.npy"
path_to_labels = "../dataset/annotations/labels.npy"

print("Loading dataset and labels")

dataset = np.load(path_to_dataset)[0:10]
labels = np.load(path_to_labels)[0:10]

print(
    f"Dataset and labels loaded\nDataset shape {dataset.shape} \nLabels shape {labels.shape}")

image_train, image_test, label_train, label_test = train_test_split(
    dataset, labels, test_size=0.25, random_state=42)

print("Dataset and labels splitted in train and test set\n" +
      f"image_train shape {image_train.shape} - label_train shape {label_train.shape}" +
      f"image_test shape {image_test.shape} - image_test shape {label_test.shape}")

image_train, label_train = data_aug_impl(dataset, image_train, label_train)

print("Applied data agumentation to train set\n" +
      f"image_train augmented shape {image_train.shape} - label_train augmented shape {label_train.shape}")

train_data = generate_train_data_tensor(image_train, label_train)

print("Preprocessed and created tensor dataset")

segnet = SegNet()
print("Created segnet model")

optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=0.01)
segnet.compile(optimizer=optimizer,
               loss="categorical_crossentropy", metrics=["accuracy"])

print("Starting training")
history = segnet.fit(train_data, epochs=3)
print("Training completed")

print(history)


plt_history(history)

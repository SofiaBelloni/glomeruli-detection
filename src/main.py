import numpy as np
from sklearn.model_selection import train_test_split
# The path can also be read from a config file, etc.
import os
if hasattr(os, 'add_dll_directory'):
    # Python >= 3.8 on Windows
    OPENSLIDE_PATH = r'C:\Users\sofia\openslide-win64-20230414\openslide-win64-20230414\bin'
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide

path_to_dataset = "../dataset/slides/dataset.npy"
path_to_labels = "../dataset/annotations/labels.npy"

dataset = np.load(path_to_dataset)
labels = np.load(path_to_labels)

image_train, image_test, label_train, label_test = train_test_split(dataset, labels, test_size=0.25, random_state=42)
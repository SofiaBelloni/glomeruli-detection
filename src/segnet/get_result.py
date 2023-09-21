import tensorflow as tf
import numpy as np
import random

model_path='../../saved_model/segnet_V1.2__balanced_dataset_'
image_validation_path = '../../dataset512x512/image_validation_balanced.npy'
image_test_path = '../../dataset512x512/image_test_balanced.npy'

path_to_save_dataset = '../../dataset512x512/image_test_step_2.npy'
path_to_save_labels = '../../dataset512x512/label_test_step_2.npy'

model = tf.keras.models.load_model(model_path)

data_test = np.load(image_test_path) /255
data_validation = np.load(image_validation_path) /255

print(f"Dataset test shape {data_test.shape}")
print(f"Dataset validation shape {data_validation.shape}")

ds = np.append(data_test, data_validation, axis = 0)

print(f"Dataset shape {ds.shape}")

res_lab = model.predict(ds)

print(f"Label shape {res_lab.shape}")

np.save(path_to_save_dataset, ds)
np.save(path_to_save_labels, res_lab)
import numpy as np
from extrapolate_glomeruli import glomeruli_crop, get_glomeruli, get_glomeruli_labels, get_no_glomeruli, get_no_glomeruli_labels
import tensorflow as tf
from sklearn.decomposition import PCA
from clustering import run_clustering
from data_prepr_aug import data_aug_no_impl
import cv2

# legion
path_to_dataset = "../../dataset512x512/dataset.npy"
path_to_labels = "../../dataset512x512/labels.npy"
# local
#path_to_dataset = "dataset/dataset512x512/RECHERCHE-015.npy"
#path_to_labels = "dataset/dataset512x512/RECHERCHE-015_label.npy"
dataset = np.load(path_to_dataset)
labels = np.load(path_to_labels)
print("Dataset shape:", dataset.shape)
print("Labels shape:", labels.shape)

glomeruli = get_glomeruli(dataset, labels)
glomeruli_labels = get_glomeruli_labels(dataset, labels)

n_glomeruli = get_no_glomeruli(dataset, labels)
n_glomeruli_labels = get_no_glomeruli_labels(dataset, labels)
print("Glomeruli shape:", glomeruli.shape)

resized_glomeruli = np.array([cv2.resize(glomerulo, dsize=(200, 200), interpolation=cv2.INTER_CUBIC) for glomerulo in glomeruli])
resized_glomeruli_labels = np.array([cv2.resize(glomerulo, dsize=(200, 200), interpolation=cv2.INTER_CUBIC) for glomerulo in glomeruli_labels])

resized_n_glomeruli = np.array([cv2.resize(glomerulo, dsize=(200, 200), interpolation=cv2.INTER_CUBIC) for glomerulo in n_glomeruli])
resized_n_glomeruli_labels = np.array([cv2.resize(glomerulo, dsize=(200, 200), interpolation=cv2.INTER_CUBIC) for glomerulo in n_glomeruli_labels])


train_glomeruli, _ = data_aug_no_impl(resized_glomeruli_labels[0].shape, resized_glomeruli, resized_glomeruli_labels)
train_n_glomeruli, _ = data_aug_no_impl(resized_n_glomeruli_labels[0].shape, resized_n_glomeruli, resized_n_glomeruli_labels)

data_train = np.append(train_glomeruli,train_n_glomeruli, axis= 0)

train_glomeruli_labels = np.ones(train_glomeruli.shape[0])
train_n_glomeruli_labels = np.zeros(train_n_glomeruli.shape[0])

label_train = np.append(train_glomeruli_labels,train_n_glomeruli_labels, axis= 0)


def process_data(image, label):
    return tf.cast(image, tf.float32)/255, label

train_data = tf.data.Dataset.from_tensor_slices((data_train, label_train))

train_data = train_data.map(
        process_data, num_parallel_calls=tf.data.AUTOTUNE)

train_data = train_data.shuffle(buffer_size=1024).batch(128)

#glomeruli_data, glomeruli_labels, ori, ori_l = glomeruli_crop(
#   glomeruli, glomeruli_labels)
#glomeruli_data = glomeruli_data.astype('float32') / 255
#
#np.save("../../dataset512x512/glomeruli_data_no_dataaug.npy", glomeruli_data)
#np.save("../../dataset512x512/glomeruli_labels_no_dataaug.npy", glomeruli_labels)

glomeruli_data=np.load("../../dataset512x512/glomeruli_data_no_dataaug.npy")
glomeruli_labels=np.load("../../dataset512x512/glomeruli_labels_no_dataaug.npy")

#print("Glomeruli found shape:", glomeruli_data.shape)

#vgg19 = tf.keras.applications.vgg19.VGG19(
#    include_top=False,
#    weights='imagenet',
#            input_shape=(200, 200, 3),
#    pooling='max',
#    classifier_activation='relu'
#)

resnet=tf.keras.applications.resnet50.ResNet50(
    include_top=False,
    weights='imagenet',
    input_tensor=None,
    input_shape=(200,200,3),
    pooling='max',
)

resnet.fit(train_data, epochs=50)
#
res = resnet.predict(glomeruli_data)
np.save("../../dataset512x512/glomeruli_features_res_finetuning.npy", res)
glomeruli_features = tf.keras.layers.GlobalAveragePooling2D()(res)

np.save("../../dataset512x512/glomeruli_features_resnet_finetuning.npy", glomeruli_features)
print("Glomeruli features shape:", glomeruli_features.shape)


glomeruli_features=np.load("../../dataset512x512/glomeruli_features_resnet_tuned.npy")

run_clustering(glomeruli_data, glomeruli_features, "resnet50_tuned", "resnet50_tuned")

glomeruli_features_pca = PCA(n_components=100).fit_transform(glomeruli_features)

run_clustering(glomeruli_data, glomeruli_features_pca, "resnet50_pca_tuned", "resnet50")
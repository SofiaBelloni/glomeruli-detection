import numpy as np
from data_prepr_aug import generate_train_data
from extrapolate_glomeruli import glomeruli_crop, get_glomeruli, get_glomeruli_labels
import tensorflow as tf
from sklearn.decomposition import PCA
from clustering import run_clustering
import cv2

# Settings
# Choose between "vgg19", "resnet50" and "autoencoder"
feature_extractor = "autoencoder"
load_saved_data = False  # Whether to load saved data or not
pca_components = 100  # Number of PCA components
finetune = True
batch_size = 8
shuffle_buffer_size = 256
save_data = False
# legion
path_to_dataset = "../../dataset512x512/dataset.npy"
path_to_labels = "../../dataset512x512/labels.npy"
# local
# path_to_dataset = "dataset/dataset512x512/RECHERCHE-015.npy"
# path_to_labels = "dataset/dataset512x512/RECHERCHE-015_label.npy"
# colab
# path_to_dataset = "/content/drive/MyDrive/MLA/dataset/RECHERCHE-015.npy"
# path_to_labels = "/content/drive/MyDrive/MLA/dataset/RECHERCHE-015_label.npy"
# legion
path_to_saved_model = "../../saved_models/autoencoderv6_dense.h5"
# colab
# path_to_saved_model = "/content/drive/MyDrive/MLA/saved_models/autoencoderv6_dense.h5"
path_to_saved_data = "../../dataset512x512/glomeruli_data_no_dataaug.npy"
path_to_saved_labels = "../../dataset512x512/glomeruli_labels_no_dataaug.npy"


def get_vgg19():
    return tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet', input_shape=(200, 200, 3), pooling=None, classifier_activation="softmax")


def get_resnet():
    return tf.keras.applications.resnet50.ResNet50(include_top=False, weights='imagenet', input_shape=(200, 200, 3), pooling=None, classifier_activation="softmax")


def get_autoencoder():
    return tf.keras.models.load_model(path_to_saved_model)

def load_data():
    if load_saved_data:
        return np.load(path_to_saved_data), np.load(path_to_saved_labels)
    else:
        dataset = np.load(path_to_dataset)
        labels = np.load(path_to_labels)

        glomeruli = get_glomeruli(dataset, labels)
        glomeruli_labels = get_glomeruli_labels(dataset, labels)

        glomeruli_data, glomeruli_labels, _, _ = glomeruli_crop(
            glomeruli, glomeruli_labels)
        glomeruli_data = glomeruli_data.astype('float32') / 255
        if save_data:
            np.save(path_to_saved_data, glomeruli_data)
            np.save(path_to_saved_labels, glomeruli_labels)

        print(glomeruli_data.shape)

        return glomeruli_data, glomeruli_labels


def fine_tuning(model):
    dataset = np.load(path_to_dataset)
    labels = np.load(path_to_labels)

    train_data, label_data = generate_train_data(
        dataset, labels, batch_size, shuffle_buffer_size)

    inputs = tf.keras.Input(shape=(200, 200, 3))
    x = model(inputs)
    x = tf. keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(4096, activation="relu")(x)
    x = tf.keras.layers.Dense(4096, activation="relu")(x)
    outputs = tf.keras.layers.Dense(2, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)

    # First, we only train the top layers (which were randomly initialized)
    for i, layer in enumerate(model.layers):
        if i > 10:
            layer.trainable = True
        else:
            layer.trainable = False

    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    print("Fine-tuning model")
    # finetune the model
    model.fit(train_data, label_data, epochs=10, batch_size=batch_size)

    layer_to_remove = model.layers[-4].name
    # Create a new model with the outputs of the original model
    feature_extractor = tf.keras.models.Model(
        inputs=model.input, outputs=model.get_layer(layer_to_remove).output)
    return model, feature_extractor


def main():
    # Load data
    glomeruli_data, _ = load_data()
    # Choose feature extractor
    if feature_extractor == "vgg19":
        model = get_vgg19()
        if finetune:
            _, model = fine_tuning(model)
        glomeruli_features = model.predict(glomeruli_data)
    elif feature_extractor == "resnet50":
        model = get_resnet()
        if finetune:
            _, model = fine_tuning(model)
        glomeruli_features = model.predict(glomeruli_data)
    elif feature_extractor == "autoencoder":
        model = get_autoencoder()
        encoder = model.layers[1]
        glomeruli_features = encoder.predict(glomeruli_data)
    # Extract features
    if save_data:
        np.save("../../dataset512x512/glomeruli_features_{}.npy".format(feature_extractor),
                glomeruli_features)
    # Run clustering
    run_clustering(glomeruli_data, glomeruli_features,
                   feature_extractor, feature_extractor)
    # Run PCA
    glomeruli_features_pca = PCA(
        n_components=pca_components).fit_transform(glomeruli_features)
    # Run clustering with PCA
    run_clustering(glomeruli_data, glomeruli_features_pca,
                   "{}_pca".format(feature_extractor), feature_extractor)


if __name__ == "__main__":
    main()

import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA
from clustering import run_clustering

feature_extractor = "autoencoder"
base_url = "autoencoder"
pca_components = 100
path_to_saved_model = "../../saved_model/autoencoder_v2.2_dense_512.h5"
path_to_dataset = "../../dataset512x512/image_test_step_2d_cropped_darked.npy"
path_to_original_dataset = "../../dataset512x512/image_test_step_2d_cropped.npy"
#path_to_saved_model = "saved_models/autoencoder_v2.2_dense_512.h5"
#path_to_dataset = "dataset512x512/image_test_step_2d_cropped_darked.npy"
#path_to_original_dataset = "dataset512x512/image_test_step_2d_cropped.npy"
save_data = False


def get_autoencoder():
    model = tf.keras.models.load_model(path_to_saved_model)
    encoder = model.layers[1]
    return encoder


def load_data():
    ds = np.load(path_to_dataset)
    ds_o = np.load(path_to_original_dataset)
    return ds, ds_o


def main():
    dataset, dataset_original = load_data()
    if dataset.max() > 1:
        dataset = dataset.astype('float32') / 255

    model = get_autoencoder()
    features = model.predict(dataset)
    # Extract features
    if save_data:
        np.save("../../dataset512x512/glomeruli_features_{}.npy".format(base_url),
                features)
    print(features.shape)
    print(features.max())
    print(features.min())
    # Run clustering
    run_clustering(dataset_original, features,
                   feature_extractor, base_url)
    # # Run PCA
    features_pca = PCA(
        n_components=pca_components).fit_transform(features)
    # # Run clustering with PCA
    run_clustering(dataset_original, features_pca,
                   "{}_pca".format(feature_extractor), base_url)


main()

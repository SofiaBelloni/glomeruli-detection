import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA
from clustering import run_clustering

feature_extractor = "vgg19"
base_url = "vgg19_tuned"
pca_components = 100
batch_size = 8
save_data = False
#path_to_saved_vgg19 = "../../saved_model/vgg19_feature_extractor_.h5"
#path_to_dataset = "../../dataset512x512/image_test_step_2d_cropped_darked.npy"
#path_to_original_dataset = "../../dataset512x512/image_test_step_2d_cropped.npy"
path_to_saved_vgg19 = "saved_models/vgg19_feature_extractor_.h5"
path_to_dataset = "dataset512x512/image_test_step_2d_cropped_darked.npy"
path_to_original_dataset = "dataset512x512/image_test_step_2d_cropped.npy"


def get_vgg19():
    return tf.keras.models.load_model(path_to_saved_vgg19)


def load_data():
    ds = np.load(path_to_dataset)
    ds_o = np.load(path_to_original_dataset)
    return ds, ds_o


def main():
    dataset, dataset_original = load_data()
    print(dataset.shape)
    model = get_vgg19()
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
    # Run PCA
    features_pca = PCA(
        n_components=pca_components).fit_transform(features)
    # Run clustering with PCA
    run_clustering(dataset_original, features_pca,
                   "{}_pca".format(feature_extractor), base_url)


main()

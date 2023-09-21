import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA
from clustering import run_clustering

feature_extractor = "resnet50"
base_url = "resnet50_tuned"
pca_components = 100
batch_size = 8
save_data = False
path_to_saved_resnet50 = "../../saved_model/resnet50_feature_extractor_.h5"
path_to_dataset = "../../dataset512x512/image_test_step_2d_cropped_darked.npy"
path_to_original_dataset = "../../dataset512x512/image_test_step_2d_cropped.npy"
# path_to_saved_model = "saved_models/resnet50_feature_extractor_.h5"
# path_to_dataset = "dataset512x512/image_test_step_2d_cropped_darked.npy"
# path_to_original_dataset = "dataset512x512/image_test_step_2d_cropped.npy"


def get_resnet50():
    return tf.keras.models.load_model(path_to_saved_resnet50)


def load_data():
    ds = np.load(path_to_dataset)
    ds_o = np.load(path_to_original_dataset)
    return ds, ds_o


def main():
    dataset, dataset_original = load_data()
    print(dataset.shape)
    model = get_resnet50()
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

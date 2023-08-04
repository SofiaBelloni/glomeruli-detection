from isomap import define_best_params_isomap, best_isomap_kmeans
from tsne import define_best_params_tsne, best_tsne_kmeans
from kmeans import define_best_params_kmeans
from sklearn.cluster import SpectralClustering
from spectral_clustering import define_best_params_spectral
from sklearn.cluster import KMeans
from plot import plot_scatter, plot_results, plot_images
from joblib import dump, load
import json

# b_usr="../../cluster_result/"
b_usr = "cluster_result/"
values = ["silhouette"]
# values=["sse", "silhouette"]


def run_clustering(dataset, features, title, base_url):

    # KMeans
    try:
        best_params_k = define_best_params_kmeans(features)
        for score_type in values:
            best_params_kmeans = best_params_k[score_type]
            kmeans = KMeans(n_clusters=best_params_kmeans)
            kmeans_labels = kmeans.fit_predict(features)

            # Update file names with score type
            plot_results(None, kmeans, best_params_kmeans, "kmeans", features,
                         f"{title}_kmeans_{score_type}", f"{b_usr}{base_url}_kmeans/")
            plot_images(dataset, kmeans_labels,
                        f"{title}_kmeans_{score_type}", f"{b_usr}{base_url}_kmeans/")
            save_model(kmeans, f"{title}_kmeans_{score_type}",
                       f"{b_usr}{base_url}_kmeans/")
            print(f"Best params kmeans: {best_params_kmeans}")
    except Exception as e:
        print(f"Something went wrong in KMeans: {e}")

    # ISOMAP
    try:
        best_params_i = define_best_params_isomap(features)
        for score_type in values:
            best_params_isomap = best_params_i[score_type]
            isomap, isomap_kmeans = best_isomap_kmeans(best_params_isomap)
            isomap_projection = isomap.fit_transform(features)
            isomap_labels = isomap_kmeans.fit_predict(isomap_projection)
    
            # Update file names with score type
            plot_results(isomap, isomap_kmeans, best_params_isomap, "isomap", features,
                         f"{title}_isomap_{score_type}", f"{b_usr}{base_url}_isomap/")
            plot_images(dataset, isomap_labels,
                        f"{title}_isomap_{score_type}", f"{b_usr}{base_url}_isomap/")
            save_model(isomap, f"{title}_isomap_{score_type}",
                       f"{b_usr}{base_url}_isomap/")
            save_model(
                isomap_kmeans, f"{title}_isomap_kmeans_{score_type}", f"{b_usr}{base_url}_isomap/")
            save_best_params(
                best_params_isomap, f"{title}_isomap_{score_type}", f"{b_usr}{base_url}_isomap/")
    except Exception as e:
        print(f"Something went wrong in Isomap-KMeans: {e}")
    

    # t-SNE
    try:
        best_params_t = define_best_params_tsne(features)
        for score_type in values:
            best_params_tsne = best_params_t[score_type]
            tsne, tsne_kmeans = best_tsne_kmeans(best_params_tsne)
            tsne_projection = tsne.fit_transform(features)
            tsne_labels = tsne_kmeans.fit_predict(tsne_projection)

            # Update file names with score type
            plot_results(tsne, tsne_kmeans, best_params_tsne, "tsne", features,
                         f"{title}_tsne_{score_type}", f"{b_usr}{base_url}_tsne/")
            plot_images(dataset, tsne_labels,
                        f"{title}_tsne_{score_type}", f"{b_usr}{base_url}_tsne/")
            save_model(tsne, f"{title}_tsne_{score_type}",
                       f"{b_usr}{base_url}_tsne/")
            save_model(
                tsne_kmeans, f"{title}_tsne_kmeans_{score_type}", f"{b_usr}{base_url}_tsne/")
            save_best_params(
                best_params_tsne, f"{title}_tsne_{score_type}", f"{b_usr}{base_url}_tsne/")
    except Exception as e:
        print(f"Something went wrong in TSNE-KMeans: {e}")

        # SpectralClustering
    try:
        for score_type in ["silhouette"]:
            best_params_spectral = define_best_params_spectral(features)[
                score_type]
            spectral = SpectralClustering(**best_params_spectral)
            spectral_labels = spectral.fit_predict(features)

            # Update file names with score type
            plot_results(None, spectral, best_params_spectral, "spectral", features,
                         f"{title}_spectral_{score_type}", f"{b_usr}{base_url}_spectral/")
            plot_images(dataset, spectral_labels,
                        f"{title}_spectral_{score_type}", f"{b_usr}{base_url}_spectral/")
            save_model(spectral, f"{title}_spectral_{score_type}",
                       f"{b_usr}{base_url}_spectral/")
            save_best_params(
                best_params_spectral, f"{title}_spectral_{score_type}", f"{b_usr}{base_url}_spectral/")
    except Exception as e:
        print(f"Something went wrong in Spectral Clustering: {e}")


def save_model(model, name, url):
    dump(model, f'{url}{name}.joblib')


def save_best_params(best_params, name, url):
    with open(f'{url}best_params_{name}.json', 'w') as f:
        json.dump(best_params, f)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn.manifold import Isomap, TSNE
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.cross_decomposition import PLSRegression
from sklearn.neighbors import NeighborhoodComponentsAnalysis
import random


def plot_scatter(x, y, labels, title):
    cmap = plt.cm.get_cmap('tab20', len(set(labels)))
    plt.scatter(x, y, c=labels, cmap=cmap)
    plt.title(title)
    plt.show()


def plot_results(transformer, kmeans, best_params, transformer_type, dataset, title, url):
    labels = kmeans.labels_
    cmap = plt.cm.get_cmap('viridis', len(np.unique(labels)))

    if transformer_type == 'kmeans' or transformer_type == 'spectral' :
        # LDA
        lda = LinearDiscriminantAnalysis(n_components=2)
        dataset_reduced = lda.fit_transform(dataset, labels)

        plt.figure()
        plt.scatter(dataset_reduced[:, 0],
                    dataset_reduced[:, 1], c=labels, cmap=cmap)
        plt.savefig(f"{url}{title}_{transformer_type}_lda.png")
        plt.clf()

        # PLSRegression
        pls = PLSRegression(n_components=2)
        dataset_reduced, _ = pls.fit_transform(dataset, labels)

        plt.figure()
        plt.scatter(dataset_reduced[:, 0],
                    dataset_reduced[:, 1], c=labels, cmap=cmap)
        plt.savefig(f"{url}{title}_{transformer_type}_pls.png")
        plt.clf()

        # NCA
        nca = NeighborhoodComponentsAnalysis(n_components=2)
        dataset_reduced = nca.fit_transform(dataset, labels)

        plt.figure()
        plt.scatter(dataset_reduced[:, 0],
                    dataset_reduced[:, 1], c=labels, cmap=cmap)
        plt.savefig(f"{url}{title}_{transformer_type}_nca.png")
        plt.clf()
    else:
        if transformer_type == 'isomap':
            n_components = best_params['isomap__n_components']
        elif transformer_type == 'tsne':
            n_components = best_params['n_components']

        if n_components <= 3:
            projection = transformer.fit_transform(dataset)
            if n_components == 2:
                plt.figure()
                plt.scatter(projection[:, 0],
                            projection[:, 1], c=labels, cmap=cmap)
                plt.savefig(f"{url}{title}_{transformer_type}_2D.png")
                plt.clf()
            else:  # n_components == 3
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(projection[:, 0], projection[:, 1],
                           projection[:, 2], c=labels, cmap=cmap)
                plt.savefig(f"{url}{title}_{transformer_type}_3D.png")
                plt.clf()
        else:
            if transformer_type == 'isomap':
                best_params['isomap__n_components'] = 2
                transformer.set_params(n_components=2)
            else:  # TSNE
                best_params['n_components'] = 2
                transformer.set_params(n_components=2)
            projection = transformer.fit_transform(dataset)
            plt.figure()
            plt.scatter(projection[:, 0],
                        projection[:, 1], c=labels, cmap=cmap)
            plt.savefig(
                f"{url}{title}_{transformer_type}_2D_n-comp-more-then-2.png")
            plt.clf()


def plot_images(dataset, labels, title, url):
    unique_labels = np.unique(labels)
    num_labels = len(unique_labels)

    fig, axes = plt.subplots(10, num_labels, figsize=(
        num_labels * 3, 30))  # Si adatta alla quantità di label

    for i, label in enumerate(unique_labels):
        # Trova le immagini con la label corrispondente
        images = dataset[labels == label]
        # Seleziona 10 immagini casuali per questa label
        images_to_plot = random.sample(list(images), min(10, len(images)))

        for j, img in enumerate(images_to_plot):
            # Usa imshow per visualizzare l'immagine
            axes[j, i].imshow(img, cmap='gray')
            axes[j, i].axis('off')

        # Imposta il titolo della colonna
        axes[0, i].set_title(f'Label {label}')

    # Imposta il titolo del plot
    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Per dare spazio al titolo
    plt.savefig(f"{url}{title}_images.png")

from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score
from sklearn.model_selection import ParameterGrid

parameters = {
    'n_clusters': [5, 6, 7, 10],  # Il numero di cluster da formare
    'eigen_solver': ['arpack', 'lobpcg', None],  # L'eigen_solver da utilizzare
    # Numero di volte che l'algoritmo k-means sarà eseguito con diverse inizializzazioni del centroide
    'n_init': [10, 20],
    # Coefficiente del kernel per il calcolo dell'affinità
    'gamma': [1.0, 0.1, 0.01],
    # Il metodo di calcolo dell'affinità
    'affinity': ['nearest_neighbors', 'rbf']
}


def define_best_params_spectral(dataset):
    results = []
    for params in ParameterGrid(parameters):
        model = SpectralClustering(
            **params, assign_labels="discretize", random_state=0)
        labels = model.fit_predict(dataset)

        silhouette_avg = silhouette_score(dataset, labels)

        results.append({
            'params': params,
            'silhouette_score': silhouette_avg
        })
    best_params_silhouette = max(
        results, key=lambda x: x['silhouette_score'])['params']

    return {"silhouette": best_params_silhouette}

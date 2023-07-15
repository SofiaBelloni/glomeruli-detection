from sklearn.manifold import Isomap
from sklearn.cluster import KMeans
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.metrics import silhouette_score

param_grid = {
    'isomap__n_components': [2, 3, 5, 10, 30],
    #'isomap__n_components': [2, 3, 5],
    'isomap__n_neighbors': [5, 10, 20, 50, 100],
    #'isomap__n_neighbors': [5, 10, 20, 50],
    'kmeans__n_clusters': [5, 7, 10]
}
pipe = Pipeline(steps=[
    ('isomap', Isomap()),
    ('kmeans', KMeans())
])


def define_best_params_isomap(dataset):
    results = []

    for params in ParameterGrid(param_grid):
        pipe.set_params(**params)
        pipe.fit(dataset)
        
        sse = pipe['kmeans'].inertia_
        silhouette_avg = silhouette_score(dataset, pipe['kmeans'].labels_)

        results.append({
            'params': params,
            'sse': sse,
            'silhouette_score': silhouette_avg
        })
    best_params_sse = min(results, key=lambda x: x['sse'])['params']
    best_params_silhouette = max(results, key=lambda x: x['silhouette_score'])['params']
    return {"sse": best_params_sse, "silhouette": best_params_silhouette}


def best_isomap_kmeans(best_params):
    isomap_params = {k.replace('isomap__', ''): v for k, v in best_params.items() if k.startswith('isomap__')}
    isomap=Isomap(**isomap_params)
    
    kmeans_params = {k.replace('kmeans__', ''): v for k, v in best_params.items() if k.startswith('kmeans__')}
    kmeans=KMeans(**kmeans_params)
    return isomap, kmeans

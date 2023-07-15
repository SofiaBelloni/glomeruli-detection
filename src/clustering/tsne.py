from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import silhouette_score

tsne_params = {
    'n_components': [2, 3],
    'perplexity': [30, 50, 80, 100],
    #'perplexity': [30, 50],
    'early_exaggeration': [12, 24, 36],
    #'learning_rate': [10, 50, 100, 200, 500],
    'learning_rate': [100, 200],
    #'n_iter': [1000, 2000, 3000, 5000]
    'n_iter': [1000]
}

kmeans_params = {
    'n_clusters': [5, 7, 10],
}


def define_best_params_tsne(dataset):
    tsne_results = []

    for tsne_param in ParameterGrid(tsne_params):
        tsne = TSNE(**tsne_param)
        X_transformed = tsne.fit_transform(dataset)
        for kmeans_param in ParameterGrid(kmeans_params):
            kmeans = KMeans(**kmeans_param)
            kmeans.fit(X_transformed)

            sse = kmeans.inertia_
            silhouette_avg = silhouette_score(X_transformed, kmeans.labels_)
            
            tsne_results.append({
                'params': {**tsne_param, **kmeans_param},
                'sse': sse,
                'silhouette_score': silhouette_avg
            })
    best_params_sse = min(tsne_results, key=lambda x: x['sse'])['params']
    best_params_silhouette = max(tsne_results, key=lambda x: x['silhouette_score'])['params']
    return {"sse": best_params_sse, "silhouette": best_params_silhouette}

def best_tsne_kmeans(best_params):
    tsne_param_names = ['n_components', 'perplexity', 'early_exaggeration', 'learning_rate', 'n_iter']
    tsne_params = {name: best_params[name] for name in tsne_param_names}
    tsne = TSNE(**tsne_params)
    
    kmeans_param_names = ['n_clusters']
    kmeans_params = {name: best_params[name] for name in kmeans_param_names}
    kmeans=KMeans(**kmeans_params)
    return tsne, kmeans

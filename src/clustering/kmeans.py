from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

parameters = {'n_clusters': [5, 6, 7, 10]}


def define_best_params_kmeans(dataset):
    results = []

    for param in parameters['n_clusters']:
        kmeans = KMeans(n_clusters=param)
        kmeans.fit(dataset)

        sse = kmeans.inertia_
        silhouette_avg = silhouette_score(dataset, kmeans.labels_)
        calinski_harabasz = calinski_harabasz_score(dataset, kmeans.labels_)
        davies_bouldin = davies_bouldin_score(dataset, kmeans.labels_)

        results.append({
            'params': param,
            'sse': sse,
            'silhouette_score': silhouette_avg,
            'calinski_harabasz_score': calinski_harabasz,
            'davies_bouldin_score': davies_bouldin
        })

    best_params_sse = min(results, key=lambda x: x['sse'])['params']
    best_params_silhouette = max(
        results, key=lambda x: x['silhouette_score'])['params']
    best_params_calinski_harabasz = max(
        results, key=lambda x: x['calinski_harabasz_score'])['params']
    best_params_davies_bouldin = min(
        results, key=lambda x: x['davies_bouldin_score'])['params']

    return {"sse": best_params_sse, "silhouette": best_params_silhouette, "calinski_harabasz": best_params_calinski_harabasz, "davies_bouldin": best_params_davies_bouldin}

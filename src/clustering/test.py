import numpy as np
from sklearn.cluster import DBSCAN, SpectralClustering
from sklearn.manifold import Isomap, TSNE, MDS, SpectralEmbedding
from plot import plot_scatter
#from umap import UMAP
from sklearn.cluster import AgglomerativeClustering

glomeruli_features = np.load("dataset/glomeruli_features_res.npy")

#isomap = Isomap(n_neighbors=100, n_components=2)
#data = isomap.fit_transform(glomeruli_features)
#tsne = TSNE(early_exaggeration=12, learning_rate=200, n_components=2, n_iter=1000, perplexity=30)
#data = tsne.fit_transform(glomeruli_features)
#mds = MDS(n_components=2, normalized_stress='auto')
#data = mds.fit_transform(glomeruli_features)
spectral = SpectralEmbedding(n_components=2, n_neighbors=30, eigen_solver='arpack', affinity='nearest_neighbors', n_jobs=1)
data = spectral.fit_transform(glomeruli_features)
#umap = UMAP(n_neighbors=15, min_dist=0.1, n_components=2)
#data = umap.fit_transform(glomeruli_features)


#agglomerative = AgglomerativeClustering(n_clusters=6, affinity='euclidean', linkage='ward')
#x=agglomerative.fit_predict(data)
#dbscan = DBSCAN(eps=0.5, min_samples=5)
#x = dbscan.fit_predict(data)
spectral = SpectralClustering(n_clusters=3, assign_labels="discretize", random_state=0, affinity="rbf", gamma=0.1)
x = spectral.fit_predict(data)

plot_scatter(data[:, 0], data[:, 1], x, " ")
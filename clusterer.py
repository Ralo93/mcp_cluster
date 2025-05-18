from umap import UMAP
from hdbscan import HDBSCAN


class Clusterer:
    def __init__(self, umap_dim=3, min_cluster_size=10):
        self.umap = UMAP(n_components=umap_dim, min_dist=0.0, metric='cosine', random_state=42)
        # Add prediction_data=True to generate prediction data for transform
        self.hdbscan = HDBSCAN(
            min_cluster_size=min_cluster_size, 
            metric='euclidean', 
            cluster_selection_method='eom',
            prediction_data=True  # Add this parameter
        )
        self.reduced = None

    def reduce(self, embeddings):
        self.reduced = self.umap.fit_transform(embeddings)
        return self.reduced

    def cluster(self, reduced=None):
        if reduced is None:
            if self.reduced is None:
                raise ValueError("No reduced embeddings available. Call reduce() first.")
            reduced = self.reduced
        return self.hdbscan.fit_predict(reduced)


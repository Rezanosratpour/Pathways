
import matplotlib.pyplot as plt

from joblib import Parallel, delayed
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)


def _evaluate_one_k(X, k, random_state=7, n_init=20, silhouette_sample_size=None):
    """
    Evaluate clustering metrics for one k.
    Returns a dict so it can be parallelized safely.
    """
    kmeans = KMeans(
        n_clusters=k,
        init="k-means++",
        n_init=n_init,
        random_state=random_state,
    )

    labels = kmeans.fit_predict(X)
    centroids = kmeans.cluster_centers_

    # Core metrics
    inertia = kmeans.inertia_
    calinski = calinski_harabasz_score(X, labels)
    davies = davies_bouldin_score(X, labels)

    # Silhouette can be very expensive for large X
    if silhouette_sample_size is not None and X.shape[0] > silhouette_sample_size:
        silhouette = silhouette_score(
            X,
            labels,
            sample_size=silhouette_sample_size,
            random_state=random_state,
        )
    else:
        silhouette = silhouette_score(X, labels)

    # Fast vectorized within-group distance
    # Original code used Euclidean distance, not squared distance
    assigned_centroids = centroids[labels]
    sum_within = np.linalg.norm(X - assigned_centroids, axis=1).sum()

    # Fast vectorized between-group distance
    global_centroid = X.mean(axis=0)
    cluster_sizes = np.bincount(labels, minlength=k)
    centroid_to_global = np.linalg.norm(centroids - global_centroid, axis=1)
    sum_between = np.sum(centroid_to_global * cluster_sizes)

    return {
        "k": k,
        "inertias": inertia,
        "silhouette_scores": silhouette,
        "calinski_scores": calinski,
        "davies_scores": davies,
        "SDW": sum_within,
        "SDB": sum_between,
    }


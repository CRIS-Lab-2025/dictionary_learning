from sklearn.cluster import KMeans


def k_means(X, k, distance_metric="Euclidean"):
    """
    X: the data to cluster
    k: the number of clusters
    """
    if distance_metric == "Euclidean":
        k_means = (KMeans(n_clusters=k))
        k_means.fit(X)
        return k_means
    else:
        # Cosine similarity you have to implement lmao. smh smh.
        return "Not implemented"
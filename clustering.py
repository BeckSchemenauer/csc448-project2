import numpy as np
from numba import njit, prange
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import fcluster
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

@njit
def _corr_distance(xi: np.ndarray, xj: np.ndarray) -> float:
    """
    Correlation distance = 1 - Pearson correlation.
    """
    n = xi.shape[0]

    mean_i = 0.0
    mean_j = 0.0
    for k in range(n):
        mean_i += xi[k]
        mean_j += xj[k]
    mean_i /= n
    mean_j /= n

    num = 0.0
    den_i = 0.0
    den_j = 0.0
    for k in range(n):
        ai = xi[k] - mean_i
        aj = xj[k] - mean_j
        num += ai * aj
        den_i += ai * ai
        den_j += aj * aj

    if den_i == 0.0 or den_j == 0.0:
        return 1.0

    corr = num / np.sqrt(den_i * den_j)
    return 1.0 - corr


@njit(parallel=True)
def correlation_distance_condensed(X: np.ndarray) -> np.ndarray:
    """
    Compute condensed (upper-triangle) correlation distance vector
    between rows of X.

    Parameters
    ----------
    X : ndarray of shape (n_genes, n_timepoints)

    Returns
    -------
    dists : ndarray of shape (n_genes * (n_genes - 1) // 2,)
        Condensed distance matrix compatible with scipy linkage.
    """
    n = X.shape[0]
    size = n * (n - 1) // 2
    dists = np.empty(size, dtype=np.float32)

    idx = 0
    for i in prange(n - 1):
        # Compute starting index for row i
        start = i * n - (i * (i + 1)) // 2
        for j in range(i + 1, n):
            dists[start + (j - i - 1)] = _corr_distance(X[i], X[j])

    return dists

def hierarchical_silhouette_by_k(
    dist_vec,
    k_range=range(2, 6),
    method="average"
):
    """
    Compute silhouette scores for hierarchical clustering
    for k in k_range.

    Parameters
    ----------
    dist_vec : np.ndarray
        Condensed distance vector (e.g. correlation distance).
    k_range : iterable
        Cluster counts to evaluate (default: 2..5).
    method : str
        Linkage method.

    Returns
    -------
    results : dict
        {k: {"silhouette": float, "labels": np.ndarray, "linkage": Z}}
    """
    # Linkage matrix
    Z = linkage(dist_vec, method=method)

    # Convert to full distance matrix for silhouette
    dist_mat = squareform(dist_vec)

    results = {}

    for k in k_range:
        labels = fcluster(Z, t=k, criterion="maxclust")
        score = silhouette_score(dist_mat, labels, metric="precomputed")

        results[k] = {
            "silhouette": float(score),
            "labels": labels,
            "linkage": Z
        }

    return results

def kmeans_silhouette_by_k(
    X: np.ndarray,
    k_range=range(2, 6),
    random_state: int = 42,
    n_init: int = 20,
    max_iter: int = 300
):
    """
    Run k-means for k in k_range and compute silhouette score (Euclidean)
    for each k.

    Parameters
    ----------
    X : np.ndarray
        Data matrix of shape (n_genes, n_timepoints), e.g. your z-scored log2 matrix.
    k_range : iterable
        Cluster counts to evaluate (default: 2..5).
    random_state : int
        Reproducibility seed.
    n_init : int
        Number of k-means initializations.
    max_iter : int
        Max iterations per run.

    Returns
    -------
    results : dict
        {k: {"silhouette": float, "labels": np.ndarray, "model": KMeans}}
    """
    results = {}

    for k in k_range:
        model = KMeans(
            n_clusters=k,
            init="k-means++",
            n_init=n_init,
            max_iter=max_iter,
            random_state=random_state
        )
        labels = model.fit_predict(X)
        score = silhouette_score(X, labels, metric="euclidean")

        results[k] = {
            "silhouette": float(score),
            "labels": labels,
            "model": model
        }

    return results

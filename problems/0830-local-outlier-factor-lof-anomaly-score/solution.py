import numpy as np

def local_outlier_factor(X, k):
    """
    Compute the Local Outlier Factor (LOF) score for each point in X.

    Args:
        X: array-like of shape (n, d)
        k: int, neighborhood size

    Returns:
        list of length n with the LOF score of each point
    """
    X = np.array(X)
    n, d = X.shape
    X2 = np.sum(X**2,axis=1,keepdims=True)
    dist = np.sqrt(X2 + X2.T - 2 * X @ X.T)
    N_k = np.argsort(dist, axis = 1)[:,1:k+1]
    k_dist = dist[np.arange(n),N_k[:,-1]]
    reach = np.maximum(
        k_dist[N_k],
        dist[np.arange(n)[:,None],N_k])
    lrd = 1 / reach.mean(axis=1)
    lof = np.mean(lrd[N_k] / lrd[:,None], axis =1)
    return lof
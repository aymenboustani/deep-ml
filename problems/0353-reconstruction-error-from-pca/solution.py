import numpy as np

def pca_reconstruction_error(X: np.ndarray, n_components: int) -> float:
    """
    Compute the mean squared reconstruction error from PCA.
    
    Args:
        X: Data matrix of shape (n_samples, n_features)
        n_components: Number of principal components to keep
        
    Returns:
        The mean squared reconstruction error (float)
    """
    mu = X.mean(axis = 0, keepdims = True)
    Xc = X - mu
    n = X.shape[0]
    C = Xc.T @ Xc / n
    val, vec = np.linalg.eigh(C)
    N = vec.shape[0]
    idx = np.argsort(val)[::-1][:n_components]
    comp = vec[:,idx]
    proj = Xc @ comp
    reconstr = proj @ comp.T + mu
    return np.mean((X - reconstr) ** 2,axis=(0,1))

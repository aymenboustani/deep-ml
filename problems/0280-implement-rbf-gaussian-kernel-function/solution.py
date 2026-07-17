import numpy as np

def rbf_kernel(X1: np.ndarray, X2: np.ndarray, gamma: float) -> np.ndarray:
    """
    Compute the RBF (Gaussian) kernel matrix between X1 and X2.
    
    Args:
        X1: First set of samples with shape (n1, d)
        X2: Second set of samples with shape (n2, d)
        gamma: Kernel coefficient (controls kernel width)
    
    Returns:
        Kernel matrix of shape (n1, n2)
    """
    # Your code here
    return np.array([[np.exp(- gamma * np.linalg.norm(x - y) ** 2) for y in X2] for x in X1])
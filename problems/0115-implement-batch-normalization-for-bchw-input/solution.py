import numpy as np

def batch_normalization(
    X: np.ndarray,
    gamma: np.ndarray,
    beta: np.ndarray,
    running_mean: np.ndarray = None,
    running_var: np.ndarray = None,
    momentum: float = 0.1,
    epsilon: float = 1e-5,
    training: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform Batch Normalization on BCHW input.
    
    Args:
        X: Input array of shape (B, C, H, W)
        gamma: Scale parameter of shape (1, C, 1, 1)
        beta: Shift parameter of shape (1, C, 1, 1)
        running_mean: Running mean for inference, shape (1, C, 1, 1)
        running_var: Running variance for inference, shape (1, C, 1, 1)
        momentum: Momentum for updating running statistics (following PyTorch convention)
        epsilon: Small constant for numerical stability
        training: If True, use batch statistics; if False, use running statistics
    
    Returns:
        Tuple of (normalized_output, updated_running_mean, updated_running_var)
    """
    if training:
        mu = np.mean(X, axis = (0,2,3), keepdims = True)
        var = np.var(X, axis = (0,2,3), keepdims = True)
        if running_mean is not None and running_var is not None:
            running_mean = (1 - momentum) * running_mean + momentum * mu
            running_var = (1 - momentum) * running_var + momentum * var
        else:
            running_mean = mu
            running_var = var
        return gamma * (X - mu) / np.sqrt(var) + beta, running_mean, running_var
    else:
        return gamma * (X - running_mean) / np.sqrt(running_var) + beta,  running_mean, running_var
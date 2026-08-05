import numpy as np

def xgboost_objective(gradients: np.ndarray, hessians: np.ndarray,
                      left_indices: np.ndarray, right_indices: np.ndarray,
                      lambda_reg: float = 1.0, gamma: float = 0.0) -> dict:
    """
    Calculate XGBoost objective function components for a potential split.
    
    Args:
        gradients: First-order gradients for each sample
        hessians: Second-order hessians for each sample
        left_indices: Indices of samples going to left child
        right_indices: Indices of samples going to right child
        lambda_reg: L2 regularization parameter
        gamma: Tree complexity penalty
        
    Returns:
        Dictionary with 'left_weight', 'right_weight', and 'gain'
    """
    left_grad, right_grad = gradients[left_indices].sum(), gradients[right_indices].sum()
    left_hes, right_hes = hessians[left_indices].sum(), hessians[right_indices].sum()
    wl = - left_grad / (left_hes + lambda_reg)
    wr = - right_grad / (right_hes + lambda_reg)
    gain = -0.5 * (left_grad * wl + right_grad * wr +(left_grad + right_grad) ** 2 / (left_hes + right_hes + lambda_reg)) - gamma
    return {
        'left_weight': wl.round(4),
        'right_weight': wr.round(4),
        'gain':  gain.round(4)                              
    }
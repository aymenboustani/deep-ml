import numpy as np

def compute_norm(arr: np.ndarray, norm_type: str) -> float:
    """
    Compute the specified norm of the input array.
    
    Args:
        arr: Input numpy array (1D or 2D)
        norm_type: Type of norm ('l1', 'l2', or 'frobenius')
    
    Returns:
        The computed norm as a float
    """
    # Your code here
    order = None
    if norm_type == 'l1':
        order = 1
    elif norm_type == "l2":
        order = 2
    else:
        order = "fro"
    return np.linalg.norm(arr,ord = order)
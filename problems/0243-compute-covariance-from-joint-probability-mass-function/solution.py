import numpy as np

def covariance_from_joint_pmf(x_values: list, y_values: list, joint_pmf: np.ndarray) -> float:
    """
    Compute the covariance of X and Y from their joint PMF.
    
    Args:
        x_values: List of possible values for X
        y_values: List of possible values for Y
        joint_pmf: 2D numpy array where joint_pmf[i][j] = P(X=x_values[i], Y=y_values[j])
    
    Returns:
        Covariance of X and Y as a float
    """
    # Your code here
    x, y = np.array(x_values,dtype=np.float64), np.array(y_values,dtype=np.float64)
    Px = np.sum(joint_pmf,axis=1)
    Py = np.sum(joint_pmf,axis=0)
    mu_x = np.sum(x * Px)
    mu_y = np.sum(y * Py)
    xy = np.outer(x,y)
    exy = np.sum(xy * joint_pmf, axis =(0,1))
    return exy - mu_x * mu_y
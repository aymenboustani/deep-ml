import numpy as np

def exponential_distribution(x: list, lam: float) -> dict:
    """
    Compute exponential distribution properties.
    
    Args:
        x: Points at which to evaluate PDF and CDF
        lam: Rate parameter (lambda) of the distribution
        
    Returns:
        Dictionary with 'pdf', 'cdf', 'mean', and 'variance' keys
    """
    # Your code here
    if lam < 0 :
        return {
        'pdf' : None,
        'cdf' : None,
        'mean': None,
        'variance': None
    }
    x = np.array(x).astype(np.float64)
    x[x<0.0] = np.inf
    pdf = lam * np.exp(-lam * x)
    cdf = 1 - np.exp(-lam * x)
    cdf[cdf==1] = 0
    mean = round(1 / lam, 4)
    variance = round(1 / lam ** 2, 4)
    return {
        'pdf' : pdf.round(4).tolist(),
        'cdf' : cdf.round(4).tolist(),
        'mean': mean,
        'variance': variance
    }
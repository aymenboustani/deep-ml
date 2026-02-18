import numpy as np

def descriptive_statistics(data: list | np.ndarray) -> dict:
    """
    Calculate various descriptive statistics metrics for a given dataset.
    
    Args:
        data: List or numpy array of numerical values
    
    Returns:
        Dictionary containing mean, median, mode, variance, standard deviation,
        percentiles (25th, 50th, 75th), and interquartile range (IQR)
    """
    # Your code here
    unique, counts = np.unique(data, return_counts = True)
    max_id = np.argmax(counts)
    mode = unique[max_id]
    return {
        'mean': np.mean(data),
        'median' : np.median(data),
        'mode' : mode,
        'variance' : np.var(data),
        'standard_deviation' : np.var(data) ** 0.5,
        '25th_percentile' : np.quantile(data, 0.25),
        '50th_percentile' : np.quantile(data, 0.5),
        '75th_percentile' : np.quantile(data, 0.75),
        'interquartile_range' : np.quantile(data, 0.75) - np.quantile(data, 0.25)
    }
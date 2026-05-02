import numpy as np

def compute_cross_entropy_loss(predicted_probs: np.ndarray, true_labels: np.ndarray, epsilon = 1e-15) -> float:
    # Your code here
    N = len(true_labels)
    return - np.sum(np.log(predicted_probs + epsilon) * true_labels) / N
import numpy as np

def noisy_topk_gating(
    X: np.ndarray,
    W_g: np.ndarray,
    W_noise: np.ndarray,
    N: np.ndarray,
    k: int
) -> np.ndarray:
    """
    Args:
        X: Input data, shape (batch_size, features)
        W_g: Gating weight matrix, shape (features, num_experts)
        W_noise: Noise weight matrix, shape (features, num_experts)
        N: Noise samples, shape (batch_size, num_experts)
        k: Number of experts to keep per example
    Returns:
        Gating probabilities, shape (batch_size, num_experts)
    """
    def softplus(x):
        return np.log(1 + np.exp(x))

    def softmax(x):
        return np.exp(x) / np.exp(x).sum()

    base = X @ W_g
    noise = X @ W_noise
    h = base + np.multiply(N, softplus(noise))
    masked = np.where(h >= np.partition(h, -k, axis=1)[:, -k][:, None], h, -np.inf)
    return softmax(masked)

    
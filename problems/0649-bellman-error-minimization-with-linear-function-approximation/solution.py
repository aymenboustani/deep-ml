import numpy as np

def bellman_error_minimization(
    features: np.ndarray,
    transition_probs: np.ndarray,
    rewards: np.ndarray,
    gamma: float,
    state_distribution: np.ndarray,
    learning_rate: float,
    n_iterations: int,
    initial_weights: np.ndarray = None
) -> tuple:
    """
    Minimize the Mean Squared Bellman Error using gradient descent
    with linear function approximation.
    
    Args:
        features: (n_states, n_features) feature matrix
        transition_probs: (n_states, n_states) transition matrix under the policy
        rewards: (n_states,) expected rewards under the policy
        gamma: discount factor
        state_distribution: (n_states,) weighting distribution over states
        learning_rate: gradient descent step size
        n_iterations: number of gradient descent updates
        initial_weights: (n_features,) initial weight vector, defaults to zeros
    
    Returns:
        Tuple of (weights, msbe, values)
    """
    S, d = features.shape
    w = initial_weights.copy() if initial_weights is not None else np.zeros(d)
    D = np.diag(state_distribution)
    A = features - gamma * transition_probs @ features
    msbe = 0
    for _ in range(n_iterations):
        delta = A @ w - rewards
        msbe = delta.T @ D @ delta
        grad = 2 * A.T @ D @ delta
        w = w - learning_rate * grad
    delta = A @ w - rewards
    msbe = delta.T @ D @ delta
    return w.round(4).tolist(), msbe, (features @ w).round(4).tolist()
import numpy as np

def epsilon_greedy(Q, epsilon=0.1):
    """
    Selects an action using epsilon-greedy policy.
    Q: np.ndarray of shape (n,) -- estimated action values
    epsilon: float in [0, 1]
    Returns: int, selected action index
    """
    # Your code here
    p = np.random.random_sample()
    if p <= epsilon:
        return np.random.choice(range(len(Q)), size=1)
    else:
        return Q.argmax()
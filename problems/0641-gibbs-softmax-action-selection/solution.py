import numpy as np

def gibbs_softmax_action_selection(q_values: list, temperature: float, seed: int) -> tuple:
    """
    Perform Gibbs softmax (Boltzmann) action selection.

    Args:
        q_values: list of floats, estimated action values
        temperature: float, temperature parameter (tau > 0)
        seed: int, random seed for reproducibility

    Returns:
        tuple: (probabilities as list of floats, selected action as int)
    """
    np.random.seed(seed)
    N = len(q_values)
    q_values = np.array(q_values)
    scores = np.exp(q_values / temperature)
    p = scores / scores.sum()
    idx = np.random.choice(list(range(N)), p=p)
    return p.tolist(), idx

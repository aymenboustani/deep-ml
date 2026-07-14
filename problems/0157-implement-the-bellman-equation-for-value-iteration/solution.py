import numpy as np

def bellman_update(V, transitions, gamma):
    """
    Perform one step of value iteration using the Bellman equation.
    Args:
      V: np.ndarray, state values, shape (n_states,)
      transitions: list of dicts. transitions[s][a] is a list of (prob, next_state, reward, done)
      gamma: float, discount factor
    Returns:
      np.ndarray, updated state values
    """
    # TODO: Implement Bellman update
    V_new = np.zeros_like(V)
    for i in range(len(V_new)):
      t = transitions[i]
      assert isinstance(V, np.ndarray), "V is not a numpy array"
      paths = {a : sum(t[a][j][0] * (t[a][j][2] + gamma * V[t[a][j][1]]) for j in range(len(t[a]))) for a in t.keys() }
      V_new[i] = max(paths.values())
    return V_new
import numpy as np
def simulate_markov_chain(transition_matrix, initial_state, num_steps):
    # Your code here
    states = np.array([i for i in range(transition_matrix.shape[0])])
    path = [initial_state]
    for _ in range(num_steps):
        last = path[-1]
        new = np.random.choice(states, p = transition_matrix[last,:])
        path.append(new)
    return path
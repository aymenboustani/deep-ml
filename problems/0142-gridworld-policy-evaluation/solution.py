def gridworld_policy_evaluation(policy: dict, gamma: float, threshold: float) -> list[list[float]]:
    """
    Evaluate state-value function for a policy on a 5x5 gridworld.
    
    Args:
        policy: dict mapping (row, col) to action probability dicts
        gamma: discount factor
        threshold: convergence threshold
    Returns:
        5x5 list of floats
    """
    # Your code here
    rewards = {(i,j) : -1 for i in range(5) for j in range(5)}
    corners = [(0,0),(0,4),(4,0),(4,4)]
    for corner in corners:
        rewards[corner] = 0
    V = {(i,j) : 0 for i in range(5) for j in range(5)}
    def P(s_, s, a):
        x, y = s

        if a == "up":
            next_state = (x, min(y + 1, 4))
        elif a == "down":
            next_state = (x, max(y - 1, 0))
        elif a == "right":
            next_state = (min(x + 1, 4), y)
        elif a == "left":
            next_state = (max(x - 1, 0), y)

        return int(s_ == next_state)
    diff = threshold + 1
    while diff >= threshold:
        V_ = V.copy()
        for i in range(5):
            for j in range(5):
                c = (i,j)
                if c not in corners:
                    V_[c] = sum(
                        policy[c][a] * sum(
                            P(s_,c,a) * (rewards[c] + gamma * V[s_])
                            for s_ in V.keys()
                        )
                        for a in policy[c].keys()
                    )
        diffs = {k : abs(V_[k] - V[k]) for k in V.keys()}
        diff = max(diffs.values())
        V = V_
    final = {i : {j : V[(i,j)] for j in range(5)} for i in range(5)}
    return final
def sample_average_action_values(k: int, actions: list, rewards: list) -> tuple:
    """
    Estimate action values using sample averaging.
    
    Args:
        k: Number of possible actions (labeled 0 to k-1)
        actions: List of actions taken at each time step
        rewards: List of rewards received at each time step
        
    Returns:
        Tuple of (Q, N) where Q is estimated values and N is selection counts
    """
    act = list(range(k))
    timesteps = len(actions)
    N = {a: actions.count(a) for a in act}
    Q = {a : round(sum(r for r, action in zip(rewards,actions) if action == a) / N[a],4) if N[a] != 0 else 0.0 for a in act}
    return list(Q.values()), list(N.values())
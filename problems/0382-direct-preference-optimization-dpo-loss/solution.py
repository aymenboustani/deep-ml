import numpy as np

def dpo_loss(log_probs_chosen_policy: list, log_probs_rejected_policy: list,
            log_probs_chosen_ref: list, log_probs_rejected_ref: list,
            beta: float) -> dict:
    """
    Compute the Direct Preference Optimization (DPO) loss.
    
    Args:
        log_probs_chosen_policy: Log-probs of chosen responses under policy
        log_probs_rejected_policy: Log-probs of rejected responses under policy
        log_probs_chosen_ref: Log-probs of chosen responses under reference model
        log_probs_rejected_ref: Log-probs of rejected responses under reference model
        beta: Temperature parameter for KL constraint strength
    
    Returns:
        Dictionary with 'loss', 'chosen_rewards', and 'rejected_rewards'
    """
    # Your code here
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    log_probs_chosen_policy = np.array(log_probs_chosen_policy)
    log_probs_rejected_policy = np.array(log_probs_rejected_policy)
    log_probs_chosen_ref = np.array(log_probs_chosen_ref)
    log_probs_rejected_ref = np.array(log_probs_rejected_ref)
    loss = -np.log(sigmoid(beta * (log_probs_chosen_policy - log_probs_chosen_ref - log_probs_rejected_policy + log_probs_rejected_ref))).mean().round(4)
    chosen = beta * (log_probs_chosen_policy - log_probs_chosen_ref)
    rejected = beta * (log_probs_rejected_policy - log_probs_rejected_ref)
    return {
        'loss': loss,
        'chosen_rewards': chosen.round(4).tolist(),
        'rejected_rewards': rejected.round(4).tolist()
    }
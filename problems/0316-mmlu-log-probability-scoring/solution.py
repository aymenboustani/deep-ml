import numpy as np

def mmlu_log_prob_score(log_probs: list, correct_answers: list) -> dict:
    """
    Compute MMLU-style log-probability scoring metrics.
    
    Args:
        log_probs: List of lists, where each inner list contains 
                   log-probabilities for each answer choice
        correct_answers: List of correct answer indices (0-indexed)
    
    Returns:
        Dictionary with 'accuracy', 'predictions', and 'avg_correct_prob'
    """
    probs, targets = np.exp(log_probs) /  np.exp(log_probs).sum(axis=1,keepdims=True), np.array(correct_answers)
    pred = probs.argmax(axis=1)
    accuracy = np.mean(pred == targets)
    avg = np.round(probs[np.arange(probs.shape[0]),targets].mean(),4)
    return {
        "accuracy" : accuracy,
        "predictions" : pred.tolist(),
        "avg_correct_prob": avg
    }
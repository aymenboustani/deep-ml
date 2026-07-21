import numpy as np

def qk_norm_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, temperature: float = 1.0) -> tuple:
    """
    Apply QK-Norm attention: L2-normalize queries and keys before computing
    scaled dot-product attention.
    
    Args:
        Q: Query matrix, shape (seq_len_q, d_k)
        K: Key matrix, shape (seq_len_k, d_k)
        V: Value matrix, shape (seq_len_k, d_v)
        temperature: Temperature scaling parameter (default: 1.0)
    
    Returns:
        Tuple of (attention_output, attention_weights)
    """
    Q = Q / ((Q**2).sum(axis=1,keepdims=True) ** 0.5 + 1e-8)
    K = K / ((K**2).sum(axis=1,keepdims=True) ** 0.5 + 1e-8)
    scores = Q @ K.T  / temperature
    scores = scores - np.max(scores,axis=1,keepdims=True)
    def softmax(x):
        return np.exp(x) / np.exp(x).sum(axis=1,keepdims=True)
    attention = softmax(scores)
    output = attention @ V
    return output, attention
import numpy as np

def multiquery_attention(X: np.ndarray, W_queries: list, W_key: np.ndarray, W_value: np.ndarray, W_out: np.ndarray) -> np.ndarray:
    """
    Compute Multi-Query Attention.
    
    Args:
        X: Input array of shape (seq_len, d_model)
        W_queries: List of query weight matrices, each (d_model, d_k), one per head
        W_key: Shared key weight matrix of shape (d_model, d_k)
        W_value: Shared value weight matrix of shape (d_model, d_v)
        W_out: Output projection matrix of shape (num_heads * d_v, d_model)
    
    Returns:
        Output array of shape (seq_len, d_model), rounded to 4 decimal places
    """
    def softmax(x):
        x = x - np.max(x,axis=-1,keepdims=True)
        return np.exp(x) / np.exp(x).sum(axis=-1,keepdims=True)
    W_q = np.array(W_queries)
    seq_len = X.shape[0]
    dv = W_value.shape[1]
    num_heads, d_model, d_k = W_q.shape
    Q = X @ W_q
    K = X @ W_key
    V = X @ W_value
    attn = softmax(Q @ K.T / np.sqrt(d_k)) @ V
    attn = attn.transpose(1,0,2)
    attn = attn.reshape(seq_len,num_heads * dv)
    return (attn @ W_out).round(4)


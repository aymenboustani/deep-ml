import numpy as np

def transformer_encoder_layer(X: np.ndarray, weights: dict, num_heads: int, eps: float = 1e-5) -> np.ndarray:
    """
    Forward pass of a single Transformer Encoder Layer.

    Args:
        X: Input tensor of shape (batch_size, seq_len, d_model)
        weights: Dictionary containing all weight matrices and normalization parameters
        num_heads: Number of attention heads
        eps: Epsilon for layer normalization

    Returns:
        Output tensor of shape (batch_size, seq_len, d_model)
    """
    # Your code here
    batch_size, seq_len, d_model = X.shape
    dk = d_model // num_heads
    W_q = weights['W_q']
    W_k = weights['W_k']
    W_v = weights['W_v']
    W_o = weights['W_o']
    W1 = weights['W1']
    W2 = weights['W2']
    b1 = weights['b1']
    b2 = weights['b2']
    gamma1 = weights['gamma1']
    gamma2 = weights['gamma2']
    beta1 = weights['beta1']
    beta2 = weights['beta2']

    def softmax(x):
        return np.exp(x) / np.exp(x).sum(axis=-1,keepdims=True)

    def relu(x):
        return np.maximum(0,x)

    Q, K, V = X @ W_q, X @ W_k, X @ W_v
    Q = Q.reshape(batch_size,seq_len,num_heads,dk).transpose(0,2,1,3)
    K = K.reshape(batch_size,seq_len,num_heads,dk).transpose(0,2,1,3)
    V = V.reshape(batch_size,seq_len,num_heads,dk).transpose(0,2,1,3)
    QK = Q @ K.transpose(0, 1, 3, 2) / dk ** 0.5
    QK = QK - QK.max(axis=-1,keepdims=True)
    Y = softmax(QK) @ V
    Y = Y.transpose(0,2,1,3).reshape(batch_size,seq_len,d_model)
    O = Y @ W_o
    O = X + O
    O = gamma1 * (O - O.mean(axis=-1,keepdims=True)) / (O.var(axis=-1,keepdims=True) + eps) ** 0.5 + beta1
    h1 = relu(O @ W1 + b1)
    h2 = h1 @ W2 + b2
    h2 = h2 + O
    return gamma2 * (h2 - h2.mean(axis=-1,keepdims=True)) / (h2.var(axis=-1,keepdims=True) + eps) ** 0.5 + beta2

    

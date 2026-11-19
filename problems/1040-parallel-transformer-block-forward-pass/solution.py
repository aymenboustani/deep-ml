import numpy as np

def parallel_transformer_block(X, gamma, beta, W_qkv, W_o, W1, W2, eps=1e-5):
    """
    Parallel transformer block: y = x + Attn(LN(x)) + FFN(LN(x)).
    Single-head self-attention, ReLU FFN, no biases.
    Returns a numpy array of shape (seq_len, d_model).
    """

    def softmax(x):
        return np.exp(x) / np.exp(x).sum(axis=1,keepdims=True)
    X, W_qkv, W_o, W1, W2 = map(np.asarray, ( X, W_qkv, W_o, W1, W2))
    seq_len, d = X.shape
    mu = X.mean(axis=1,keepdims=True)
    var = X.var(axis=1,keepdims=True)
    x_hat  = gamma * (X - mu) / np.sqrt(var + eps) + beta
    QKV = x_hat @ W_qkv
    Q, K, V = np.split(QKV,3, axis =1)
    attn = softmax(Q @ K.T / np.sqrt(d)) @ V @ W_o
    ffn = np.maximum(0,x_hat @ W1) @ W2
    return X + attn + ffn

import numpy as np

def moe(x: np.ndarray, We: np.ndarray, Wg: np.ndarray, n_experts: int, top_k: int) -> np.ndarray:
    """
    Args:
        x: Input tensor of shape (n_batch, l_seq, d_model)
        We: Expert weights of shape (n_experts, d_model, d_model)
        Wg: Gating weights of shape (d_model, n_experts)
        n_experts: Number of experts
        top_k: Number of experts to route each token to
    Returns:
        Output tensor of shape (n_batch, l_seq, d_model)
    """
    g = x @ Wg # (n_batch,l_seq,n_experts)
    scores = g - np.max(g, axis = 2, keepdims= True)
    alpha = np.exp(scores) / np.exp(scores).sum(axis=2,keepdims=True)
    top_k_idx = np.argpartition(alpha, -top_k, axis = 2)[:,:,-top_k:]
    mask = np.zeros_like(alpha)
    np.put_along_axis(mask,top_k_idx,1,axis=2)
    alpha = mask * alpha # (n_batch,l_seq,n_experts)
    alpha = alpha / alpha.sum(axis =2, keepdims =True) # (n_batch,l_seq,n_experts)
    O = np.einsum('bld,edh->bleh', x, We) #(n_batch, l_seq, n_experts,d_model)
    y = O * alpha[...,np.newaxis] #(n_batch, l_seq, experts,d_model)
    y = np.sum(y,axis=2,keepdims=True)
    return y[:,:,0,:]
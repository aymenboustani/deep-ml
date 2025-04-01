import numpy as np

def softmax(x):
	xmax = x.max(axis = - 1, keepdims = True)
	return np.exp(x - xmax) / np.exp(x - xmax).sum(axis = - 1, keepdims = True)

def compute_qkv(X, W_q, W_k, W_v):
	return X @ W_q, X @ W_k, X @ W_v

def self_attention(Q, K, V):
	return softmax(Q @ K.T / np.sqrt(Q.shape[1])) @ V

def multi_head_attention(Q, K, V, n_heads):
	Qs = np.array_split(Q, n_heads, axis = 1)
	Ks = np.array_split(K, n_heads, axis = 1)
	Vs = np.array_split(V, n_heads, axis = 1)
	return np.concatenate([self_attention(q, k, v) for q, k, v in zip(Qs,Ks,Vs)], axis = 1)
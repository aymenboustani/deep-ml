import numpy as np

def compute_qkv(X, W_q, W_k, W_v):
	Q, K, V = np.dot(X, W_q), np.dot(X, W_k), np.dot(X, W_v)
	return Q, K, V

def self_attention(Q, K, V):
	dk = K.shape[1]
	attention = Q @ K.T / np.sqrt(dk)
	attention = np.exp(attention) / np.sum(np.exp(attention), axis = 1, keepdims = True)
	attention_output = attention @ V
	
	return attention_output

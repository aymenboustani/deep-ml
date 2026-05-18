import numpy as np

def scaled_dot_product_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, mask: np.ndarray = None) -> tuple:
	"""
	Compute Scaled Dot-Product Attention.
	
	Args:
		Q: Query matrix of shape (seq_len_q, d_k)
		K: Key matrix of shape (seq_len_k, d_k)
		V: Value matrix of shape (seq_len_k, d_v)
		mask: Optional binary mask of shape (seq_len_q, seq_len_k)
	
	Returns:
		Tuple of (output, attention_weights)
	"""

	def softmax(x):
		return np.exp(x) / np.exp(x).sum(axis =1, keepdims =True)
	# Your code here
	d_k = Q.shape[1]
	attention_weights = Q @ K.T / d_k ** 0.5

	attention_weights = attention_weights - np.max(attention_weights, axis =1, keepdims=True)
	if mask is not None: attention_weights[mask == 0] = float('-inf')
	attention_weights = softmax(attention_weights)
	return attention_weights @ V, attention_weights

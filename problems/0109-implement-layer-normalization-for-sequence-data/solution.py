import numpy as np

def layer_normalization(X: np.ndarray, gamma: np.ndarray, beta: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
	"""
	Perform Layer Normalization.
	"""
	# Your code here
	B, seq_len, d_model = X.shape
	X = X.reshape(-1, d_model)
	def norm(x):
		x = (x - x.mean()) / np.sqrt(x.var() + epsilon)
		return gamma * x + beta
	X = np.apply_along_axis(norm, axis = 1, arr = X)
	return X.reshape(B, seq_len, d_model)
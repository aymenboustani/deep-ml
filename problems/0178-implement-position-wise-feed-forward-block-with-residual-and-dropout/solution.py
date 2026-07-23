import numpy as np

def ffn(x: list[float], W1: list[list[float]], b1: list[float], W2: list[list[float]], b2: list[float], dropout_p: float=0.1, seed: int=42) -> list[float]:
	"""
	Implement a position-wise feed-forward block with residual and dropout.

	Args:
		x: input vector
		W1, b1: first linear layer parameters
		W2, b2: second linear layer parameters
		dropout_p: dropout probability
		seed: random seed for reproducibility

	Returns:
		Output vector after FFN block (rounded to 4 decimals)
	"""
	# Your code here
	np.random.seed(seed)
	np.random.RandomState(seed)
	relu = lambda t : np.maximum(0,t)
	x, W1, b1, W2, b2 = np.array(x), np.array(W1), np.array(b1), np.array(W2), np.array(b2)
	h1 = relu(W1 @ x + b1)
	h2 = relu(W2 @ h1 + b2)
	mask = np.random.rand(*h2.shape)
	mask = (mask >= (dropout_p)).astype(float)
	h2 = h2 * mask / (1-dropout_p)
	return np.round(h2+x,4)
	
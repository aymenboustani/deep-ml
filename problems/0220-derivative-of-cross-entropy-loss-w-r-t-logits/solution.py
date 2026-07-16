def cross_entropy_derivative(logits: list[float], target: int) -> list[float]:
	"""
	Compute the derivative of cross-entropy loss with respect to logits.
	
	Args:
		logits: Raw model outputs (before softmax)
		target: Index of the true class (0-indexed)
		
	Returns:
		Gradient vector where gradient[i] = dL/d(logits[i])
	"""
	# Your code here
	import numpy as np
	def softmax(x):
		return np.exp(x) / np.exp(x).sum(keepdims=True)
	scores = softmax(logits)
	gradient = np.array([scores[j] for j in range(len(scores))])
	gradient[target] -= 1
	return gradient
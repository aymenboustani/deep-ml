def softmax_derivative(x: list[float]) -> list[list[float]]:
	"""
	Compute the Jacobian matrix of the softmax function.
	
	Args:
		x: Input vector of real numbers
		
	Returns:
		Jacobian matrix J where J[i][j] = d(softmax_i)/d(x_j)
	"""
	import numpy as np
	def softmax(y):
		return np.exp(y) / np.exp(y).sum(keepdims=True)
	scores = softmax(x)
	n = scores.shape[0]
	J = np.eye(n)
	for i in range(n):
		for j in range(n):
			if i == j:
				J[i,j] = scores[i] * (1 - scores[i])
			else:
				J[i,j] = -scores[i] * scores[j]
	return J.tolist()

	
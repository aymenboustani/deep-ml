import numpy as np

def train_softmaxreg(X: np.ndarray, y: np.ndarray, learning_rate: float, iterations: int) -> tuple[list[float], ...]:
	"""
	Gradient-descent training algorithm for Softmax regression, optimizing parameters with Cross Entropy loss.
	"""
	# Your code here
	def softmax(scores):
		scores = scores - np.max(scores, axis =1, keepdims = True)
		scores = np.exp(scores)
		return scores / scores.sum(axis = 1, keepdims=True)

	N, D = X.shape
	C = np.max(y) + 1
	bias = np.ones(N)
	X = np.c_[bias, X]
	beta = np.zeros((D + 1, C))
	one_hot = np.eye(C)[y]
	eta = learning_rate
	losses = []
	for _ in range(iterations):
		pred = X @ beta
		loss = -np.sum(one_hot * np.log(softmax(pred)))
		losses.append(loss)
		beta = beta - eta * (X.T @ (softmax(pred) - one_hot))
	return beta.T, losses

import numpy as np

def train_logreg(X: np.ndarray, y: np.ndarray, learning_rate: float, iterations: int) -> tuple[list[float], ...]:
	"""
	Gradient-descent training algorithm for logistic regression, optimizing parameters with Binary Cross Entropy loss.
	"""
	# Your code here
	def sigmoid(x):
		return 1 / (1 + np.exp(-x))

	N, D = X.shape
	bias = np.ones(N)
	X = np.c_[bias, X]
	w = np.zeros((D + 1, 1))
	losses = []
	y = y.reshape(-1, 1)

	for _ in range(iterations):
		preds = sigmoid(X @ w)
		error = preds - y
		loss = - np.mean(y.T @ np.log(preds) + (1-y).T @ np.log(1-preds))
		dL = X.T @ error
		losses.append(loss.round(4).item())
		w = w - learning_rate * dL
	return w.round(4).flatten().tolist(), losses
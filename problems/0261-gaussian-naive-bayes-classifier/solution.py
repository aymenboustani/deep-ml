import numpy as np

def gaussian_naive_bayes(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
	"""
	Implements Gaussian Naive Bayes classifier.
	
	Args:
		X_train: Training features (shape: N_train x D)
		y_train: Training labels (shape: N_train)
		X_test: Test features (shape: N_test x D)
	
	Returns:
		Predicted class labels for X_test (shape: N_test)
	"""
	# Your code here
	X = X_train
	labels = np.unique(y_train)
	C = labels.shape[0]
	N, D = X.shape
	Py = np.bincount(y_train) / N
	mean, var = np.zeros((C,D)), np.zeros((C,D))
	for l in labels:
		for j in range(D):
			mean[l,j] = np.mean(X[y_train == l][:,j])
			var[l,j] = np.var(X[y_train == l][:,j])
	pred = []
	for x in X_test:
		cands = np.array([])
		for l in labels:
			cand = np.log(Py[l]) - np.sum(0.5 * np.log(2 * np.pi * var[l]) + (x - mean[l]) ** 2 / 2 / var[l])
			cands = np.append(cands,cand)
		pred.append(cands.argmax().item())
	return pred
	

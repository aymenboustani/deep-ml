def calculate_covariance_matrix(vectors: list[list[float]]) -> list[list[float]]:
	# Your code here
	import numpy as np
	matrix = np.array(vectors)
	n, m = matrix.shape
	mean = np.mean(matrix, axis = 1)[:, np.newaxis]
	centered = matrix - mean
	cov = np.zeros((n,n))
	for i in range(n):
		for j in range(n):
			cov[i,j] = np.dot(centered[i,:], centered[j,:]) / (m-1)
	return cov.tolist()
	
import numpy as np

def grassmann_distance(U_A: list[list[float]], U_B: list[list[float]]) -> float:
	"""
	Compute the projection-based Grassmann distance between two subspaces
	represented by column-orthonormal matrices U_A and U_B.
	"""
	U_A, U_B = np.array(U_A), np.array(U_B)
	i, j = U_A.shape[1], U_B.shape[1]
	p = min(i,j)
	M = U_A.T @ U_B
	S = np.linalg.svd(M)[1]
	return np.sqrt(p - np.sum(S**2))
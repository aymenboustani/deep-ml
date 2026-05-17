import numpy as np
def solve_jacobi(A: np.ndarray, b: np.ndarray, n: int) -> list:
	D = A.shape[1]
	x = np.zeros((D,1))
	x_old = np.ones_like(x)
	for _ in range(n):
		x_old = x.copy()
		for i in range(len(x)):
			x[i] = 1 / A[i,i] * (b[i] - sum(A[i, j] * x_old[j] for j in range(len(x)) if j != i))
		if np.linalg.norm(x - x_old) < 1e-3:
			break
	return x.flatten()
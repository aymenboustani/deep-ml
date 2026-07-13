import numpy as np

def jacobian_matrix(f, x: list[float], h: float = 1e-5) -> list[list[float]]:
	"""
	Compute the Jacobian matrix using numerical differentiation.
	
	Args:
		f: Function that takes a list and returns a list
		x: Point at which to evaluate the Jacobian
		h: Step size for finite differences
	
	Returns:
		Jacobian matrix as list of lists
	"""
	# Your code here
	x = np.array(x).astype(float)
	F = lambda u : np.array(f(u))
	output = F(x)
	der = lambda v, y : (F(v) - F(y)) / h
	n , d = output.shape[0], x.shape[0]
	J = np.zeros((n,d))
	for i in range(n):
		for j in range(d):
			x_temp = x.copy()
			x_temp[j] += h
			J[i,j] = der(x_temp, x)[i]
	return J.tolist()
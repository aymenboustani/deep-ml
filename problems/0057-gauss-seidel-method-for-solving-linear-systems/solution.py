import numpy as np

def gauss_seidel(A, b, n, x_ini=None):
	x_ini = x_ini if x_ini is not None else np.zeros((A.shape[0], 1))
	x = x_ini
	for _ in range(n):
		for i in range(len(x)):
			a = 1 / A[i,i]
			sum1 = sum(A[i,j] * x[j] for j in range(len(x)) if j < i)
			sum2 = sum(A[i,j] * x_ini[j] for j in range(len(x)) if j > i)
			x[i] = a * (b[i] - sum1 - sum2)
		x_ini = x
	return x.flatten()

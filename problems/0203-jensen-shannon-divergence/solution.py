import numpy as np

def jensen_shannon_divergence(P: list[float], Q: list[float]) -> float:
	"""
	Compute the Jensen-Shannon Divergence between two probability distributions.
	
	Args:
		P: First probability distribution
		Q: Second probability distribution
	
	Returns:
		Jensen-Shannon Divergence value
	"""
	# Your code here
	P, Q = np.array(P), np.array(Q)
	M = 0.5 * (P + Q)
	def DKL(x,y):
		p = x * np.log(x/y)
		return np.sum(p)
	P = P +1e-10
	Q = Q +1e-10
	return 0.5 * (DKL(P,M) + DKL(Q, M))
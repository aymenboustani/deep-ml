
def orthogonal_projection(v, L):
	"""
	Compute the orthogonal projection of vector v onto line L.

	:param v: The vector to be projected
	:param L: The line vector defining the direction of projection
	:return: List representing the projection of v onto L
	"""
	def dot(u, v):
		return sum(u_ * v_ for u_, v_ in zip(u, v))
	
	num = dot(v, L)
	denum = dot(L, L)
	return [num / denum * elem for elem in L]

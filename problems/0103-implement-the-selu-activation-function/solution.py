def selu(x: float) -> float:
	"""
	Implements the SELU (Scaled Exponential Linear Unit) activation function.

	Args:
		x: Input value

	Returns:
		SELU activation value
	"""
	alpha = 1.6732632423543772
	scale = 1.0507009873554804
	# Your code here
    from math import exp
	return scale * x if x > 0 else scale * alpha * (exp(x) - 1)
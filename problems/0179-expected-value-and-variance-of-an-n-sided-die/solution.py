def dice_statistics(n: int) -> tuple[float, float]:
	"""
	Compute the expected value and variance of a fair n-sided die roll.

	Args:
		n (int): Number of sides of the die

	Returns:
		tuple: (expected_value, variance)
	"""
	# Your code here
	mu = sum(i for i in range(1, n + 1)) / n
	n2  = sum(i ** 2 for i in range(1, n + 1)) / n
	var = n2 - mu ** 2
	return mu, var
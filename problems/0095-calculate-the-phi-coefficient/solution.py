def phi_corr(x: list[int], y: list[int]) -> float:
	"""
	Calculate the Phi coefficient between two binary variables.

	Args:
	x (list[int]): A list of binary values (0 or 1).
	y (list[int]): A list of binary values (0 or 1).

	Returns:
	float: The Phi coefficient rounded to 4 decimal places.
	"""
	# Your code here
	x00 = sum(a == b == 0 for a, b in zip(x, y))
	x01 = sum(a == 0 and b == 1 for a, b in zip(x, y))
	x10 = sum(a == 1 and b == 0 for a, b in zip(x, y))
	x11 = sum(a == b == 1 for a, b in zip(x, y))
	num = x00 * x11 - x01 * x10
	denum = ((x00 + x01) * (x10 + x11) * (x00 + x10) * (x01 + x11)) ** 0.5
	phi = num / denum
	return round(phi,4)
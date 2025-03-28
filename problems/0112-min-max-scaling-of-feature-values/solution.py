def min_max(x: list[int]) -> list[float]:
	# Your code here
    a, b = min(x), max(x)
    return [(X - a) / (b - a) for X in x] if b != a else [0] * len(x)
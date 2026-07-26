def binary_cross_entropy(y_true: list[float], y_pred: list[float], epsilon: float = 1e-15) -> float:
	"""
	Compute binary cross-entropy loss.
	
	Args:
		y_true: True binary labels (0 or 1)
		y_pred: Predicted probabilities (between 0 and 1)
		epsilon: Small value for numerical stability
	
	Returns:
		Mean binary cross-entropy loss
	"""
	# Your code here
	import math
	return - sum(y * math.log(pred) + (1 - y) * math.log(1 - pred) for y, pred in zip(y_true,y_pred)) / len(y_true)
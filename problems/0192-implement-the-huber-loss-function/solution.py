def huber_loss(y_true, y_pred, delta=1.0):
	"""
	Compute the Huber Loss between true and predicted values.

	Args:
		y_true (float | list[float]): Ground truth values
		y_pred (float | list[float]): Predicted values
		delta (float): Transition threshold between MSE and MAE behavior

	Returns:
		float: Average Huber loss
	"""
	# Your code here
	L = [0.5 * (y - y_) ** 2 if abs(y - y_) <= delta else (abs(y - y_) - 0.5 * delta) * delta for y, y_ in zip(y_true, y_pred)] if isinstance(y_true, list) and isinstance(y_pred, list) else [0.5 * (y_true - y_pred) ** 2 if abs(y_true - y_pred) <= delta else (abs(y_true - y_pred) - 0.5 * delta) * delta]
    return sum(L) / len(L)
def calculate_f1_score(y_true, y_pred):
	"""
	Calculate the F1 score based on true and predicted labels.

	Args:
		y_true (list): True labels (ground truth).
		y_pred (list): Predicted labels.

	Returns:
		float: The F1 score rounded to three decimal places.
	"""
	# Your code here
	TP = sum(y == y_ == 1 for y, y_ in zip(y_true, y_pred))
    FP = sum(y == 0 and y_ == 1 for y, y_ in zip(y_true, y_pred))
    FN = sum(y == 1 and y_ == 0 for y, y_ in zip(y_true, y_pred))
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0.0
	return round(f1,3)
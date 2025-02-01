import numpy as np
def recall(y_true, y_pred):
	TP = sum(t == p and t == 1 for t, p in zip(y_true, y_pred))
	FN = sum(t != p and t == 1 for t, p in zip(y_true, y_pred))
	return round(TP / (TP + FN), 3)

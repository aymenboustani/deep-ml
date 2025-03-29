import numpy as np

def f_score(y_true, y_pred, beta):
	"""
	Calculate F-Score for a binary classification task.

	:param y_true: Numpy array of true labels
	:param y_pred: Numpy array of predicted labels
	:param beta: The weight of precision in the harmonic mean
	:return: F-Score rounded to three decimal places
	"""
	TP = sum(y_ == y == 1 for y_, y in zip(y_pred, y_true))
	FN = sum(y_ != y and y_ == 0 for y_, y in zip(y_pred,y_true))
	FP = sum(y_ != y and y_ == 1 for y_, y in zip(y_pred,y_true))
	recall = TP / (TP + FN)
	precision = TP /(TP + FP)
	f_score = (1 + beta ** 2) * precision * recall / (recall + precision * beta ** 2)
    return round(f_score, 3)
 


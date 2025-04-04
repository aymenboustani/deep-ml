
import numpy as np

def dice_score(y_true, y_pred):
	# Write your code here
    TP = sum(y_ == y == 1 for y_, y in zip(y_pred, y_true))
	FN = sum(y_ != y and y_ == 0 for y_, y in zip(y_pred,y_true))
	FP = sum(y_ != y and y_ == 1 for y_, y in zip(y_pred,y_true))
    res = 2 * TP / (2 * TP + FP + FN) if (2 * TP + FP + FN) != 0 else 0.0
    return round(res, 3)

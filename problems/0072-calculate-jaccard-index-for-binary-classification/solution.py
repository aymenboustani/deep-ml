
import numpy as np

def jaccard_index(y_true, y_pred):
	# Write your code here
    intersect = sum(y_ == y == 1 for y_, y in zip(y_true,y_pred))
    result = intersect / (sum(y_true == 1) + sum(y_pred == 1) - intersect)
	return round(result, 3)

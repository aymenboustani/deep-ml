
import numpy as np

def gini_impurity(y):
	"""
	Calculate Gini Impurity for a list of class labels.

	:param y: List of class labels
	:return: Gini Impurity rounded to three decimal places
	"""
	unique = set(y)
    val = 1 - sum([(sum(y_ == u for y_ in y) / len(y)) ** 2 for u in unique])
	return round(val,3)
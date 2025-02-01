
import numpy as np

def r_squared(y_true, y_pred):
	# Write your code here
	SSR = np.sum((y_true - y_pred) ** 2)
	mu = np.mean(y_true)
	SST = np.sum((y_true - mu) ** 2)
	return 1 - SSR / SST

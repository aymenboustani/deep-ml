import numpy as np

def rmse(y_true, y_pred):
	rmse_res = np.sqrt(1 / y_true.size * np.sum((y_true - y_pred) ** 2))
	return round(rmse_res,3)

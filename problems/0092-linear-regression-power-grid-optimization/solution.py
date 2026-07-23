import math
import numpy as np

PI = 3.14159

def power_grid_forecast(consumption_data):
	# 1) Subtract the daily fluctuation (10 * sin(2Ï * i / 10)) from each data point.
	# 2) Perform linear regression on the detrended data.
	# 3) Predict day 15's base consumption.
	# 4) Add the day 15 fluctuation back.
	# 5) Round, then add a 5% safety margin (rounded up).
	# 6) Return the final integer.
	f = 10 * np.sin(2 * np.pi* np.arange(1, len(consumption_data) +1) / 10)
	data = np.array(consumption_data, dtype = np.float64)
	data -= f
	x = np.arange(1, len(consumption_data) +1)
	x = np.c_[np.ones(len(data)),x]
	w = np.linalg.inv(x.T @ x) @ x.T @ data
	pred = np.array([1,15]) @ w + 10 * np.sin(2 * np.pi * 15 / 10)
	return np.ceil(1.05 * pred)
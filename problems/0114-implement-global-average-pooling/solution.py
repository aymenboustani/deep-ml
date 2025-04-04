import numpy as np

def global_avg_pool(x: np.ndarray) -> np.ndarray:
	# Your code here
	h, w, c = x.shape
    x = x.reshape(h * w, c)
    return [x[:,i].sum() / h / w for i in range(x.shape[1])] 
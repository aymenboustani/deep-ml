import numpy as np

def residual_block(x: np.ndarray, w1: np.ndarray, w2: np.ndarray) -> np.ndarray:
	# Your code here
	x = np.maximum(0, w1 @ x)
    return w2 @ x + x
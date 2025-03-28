
import numpy as np

def cosine_similarity(v1, v2):
	# Implement your code here
	return np.round(np.dot(v1,v2) / np.linalg.norm(v1) / np.linalg.norm(v2),3)

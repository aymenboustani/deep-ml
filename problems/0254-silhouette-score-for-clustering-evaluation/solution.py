import numpy as np

def silhouette_score(X: np.ndarray, labels: np.ndarray) -> float:
	"""
	Calculate the Silhouette Score for clustering evaluation.
	
	X: shape (n_samples, n_features) - data points
	labels: shape (n_samples,) - cluster label for each point
	
	Returns: silhouette score (float between -1 and 1)
	"""
	s = []
	unique = np.unique(labels)
	if unique.shape[0] == 1 or unique.shape[0] == X.shape[0]: return 0.0
	masks = {label: labels == label for label in unique}
	for idx, x in enumerate(X):
		label = labels[idx]
		masks[label][idx] = False
		distances = {l : np.linalg.norm(X[mask] - x,axis=1,ord=2).mean() for l, mask in masks.items()}
		a = distances[label]
		b = np.min([distances[l] for l in unique if l != label])
		s.append((b-a)/max(a,b))
		masks[label][idx] = True
	return sum(s) / len(s)



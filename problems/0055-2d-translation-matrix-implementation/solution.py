import numpy as np
def translate_object(points, tx, ty):
    T = [[1, 0, tx],
         [0, 1, ty],
         [0, 0, 1]]
    T = np.array(T)
    points = np.array(points)
    points = np.hstack([points, np.ones((points.shape[0], 1))])
    points = (T @ points.T).T[:, :2]
	return points

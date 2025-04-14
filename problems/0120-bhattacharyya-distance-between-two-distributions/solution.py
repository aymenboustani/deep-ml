import numpy as np

def bhattacharyya_distance(p: list[float], q: list[float]) -> float:
    # Your code here
    p, q = np.array(p), np.array(q)
    if len(p) != len(q) :
        return 0.0
    else:
        BC = np.sqrt(p * q).sum()
        return - np.log(BC)
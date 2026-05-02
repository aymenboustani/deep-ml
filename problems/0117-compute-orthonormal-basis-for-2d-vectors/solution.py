import numpy as np

def orthonormal_basis(vectors: list[list[float]], tol: float = 1e-10) -> list[np.ndarray]:
    # Your code here
    def proj(x,y):
        return np.dot(x,y) * y
    
    vectors = np.array(vectors).astype(float)
    v = np.copy(vectors)
    norm0 = np.linalg.norm(v[0])
    if norm0 == 0:
        return []
    else:
        v[0] = v[0] / norm0
    basis = [v[0]]
    for i in range(1, len(vectors)):
        sum_ = np.sum(np.array([proj(v[i], v[j]) for j in range(i)]), axis = 0)
        w = v[i] - sum_
        norm = np.linalg.norm(w)
        if norm > tol:
            basis.append(w / norm)
    return basis
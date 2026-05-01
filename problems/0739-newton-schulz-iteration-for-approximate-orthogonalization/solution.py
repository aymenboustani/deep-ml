import numpy as np

def newton_schulz(M, num_iters: int, a: float, b: float, c: float):
    """
    Apply Newton-Schulz iterations to approximately orthogonalize M.
    Returns the resulting matrix as a nested list of floats.
    """
    # Your code here
    norm = np.linalg.norm(M, ord = 'fro')
    if norm == 0:
        return M
    M = M / norm
    for _ in range(num_iters):
        MMT = M @ M.T
        M = a * M + b * MMT @ M + c * MMT ** 2 @ M
    return M.tolist()

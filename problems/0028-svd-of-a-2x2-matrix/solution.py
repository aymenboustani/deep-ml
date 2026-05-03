import numpy as np

def svd_2x2(A: np.ndarray) -> tuple:
    """
    Compute SVD of a 2x2 matrix.
    
    Args:
        A: 2x2 numpy array
    
    Returns:
        U: 2x2 orthogonal matrix (left singular vectors)
        s: 1D array of singular values
        V: 2x2 matrix (right singular vectors)
    """
    # Your code here
    a11, a12, a21, a22 = A[0,0], A[0,1], A[1,0], A[1,1]
    y1, x1, y2, x2 = a21 + a12, a11 - a22, a21 - a12, a11 + a22
    h1, h2 = np.sqrt(y1 ** 2 + x1 ** 2), np.sqrt(y2 ** 2 + x2 ** 2)
    sigma1, sigma2 = (h1 + h2) / 2, np.abs(h1 - h2) / 2
    S = np.diag([sigma1, sigma2])
    t1, t2 = x1 / h1, x2 / h2
    cc = np.sqrt((1+t1) * (1+t2))
    ss = np.sqrt((1-t1) * (1-t2))
    sc = np.sqrt((1-t1) * (1+t2))
    cs = np.sqrt((1+t1) * (1-t2))
    c1 = (cc - ss) / 2
    s1 = (sc + cs) / 2
    U = np.array(
        [
            [c1, -s1],
            [s1, c1]
        ]
    )
    V = np.linalg.inv(S) @ U.T @ A
    return U, np.diag(S), V
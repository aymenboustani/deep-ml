import numpy as np

def elastic_net_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    alpha1: float = 0.1,
    alpha2: float = 0.1,
    learning_rate: float = 0.01,
    max_iter: int = 1000,
    tol: float = 1e-4,
) -> tuple:
    # Implement Elastic Net regression here
    w = np.zeros(X.shape[1])
    n = X.shape[0]
    b = np.zeros(1)
    eta = learning_rate
    dJ = np.ones_like(w)
    for _ in range(max_iter):
        if np.linalg.norm(dJ, ord = 1) < tol:
            break
        preds = X @ w + b
        error = preds - y
        dJ = np.dot(error.T, X)/ n + alpha1 * np.sign(w) + 2 * alpha2 * w
        b = b - eta * error.sum() / n
        w = w - eta * dJ
    return w, b.item()

        

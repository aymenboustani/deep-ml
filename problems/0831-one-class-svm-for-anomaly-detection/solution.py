import numpy as np

def one_class_svm(X_train, X_test, nu: float = 0.1, learning_rate: float = 0.001, max_iter: int = 2000) -> list:
    """
    Train a One-Class SVM (primal form) using subgradient descent and predict labels for X_test.

    Args:
        X_train: array-like of shape (n, d), training data assumed to be 'normal'.
        X_test:  array-like of shape (m, d), points to score.
        nu: upper bound on fraction of training outliers, in (0, 1].
        learning_rate: step size for subgradient descent.
        max_iter: number of iterations.

    Returns:
        A list of length m with values in {1, -1}.
    """
    X_train, X_test = map(np.asarray, (X_train,X_test))
    n, d = X_train.shape
    w = np.zeros((d,))
    rho = 0
    preds = []
    for _ in range(max_iter):
        diff = rho - X_train @ w
        A = diff > 0
        gw = w - X_train[A].sum(axis=0) / nu / n
        gp = A.sum() / nu / n - 1
        w -= learning_rate * gw
        rho -= learning_rate * gp
    preds = (X_test @ w - rho >= 0).astype(float)
    preds[preds == 0] = -1
    return preds.tolist()


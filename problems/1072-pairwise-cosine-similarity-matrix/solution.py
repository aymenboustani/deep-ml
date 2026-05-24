import numpy as np

def pairwise_cosine_similarity(X):
    # Your code here
    X = np.array(X)
    n = X.shape[0]
    S = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            norm1, norm2 = np.linalg.norm(X[i]), np.linalg.norm(X[j])
            S[i,j] = np.dot(X[i], X[j])
            if norm1 != 0: S[i,j] = S[i,j] / norm1
            if norm2 != 0: S[i,j] = S[i,j] / norm2
    return S
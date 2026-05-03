import numpy as np

def cramers_rule(A, b):
    # Your code here
    A, b = np.array(A), np.array(b)
    detA = np.linalg.det(A)
    D = A.shape[1]
    sol = []
    if detA == 0: return -1
    for j in range(D):
        new_A = A.copy()
        new_A[:,j] = b
        det = np.linalg.det(new_A)
        sol.append(det / detA)
    return np.array(sol)
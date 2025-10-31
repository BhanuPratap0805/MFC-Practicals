import numpy as np

def gram_schmidt(V):
    # V is a numpy array with column vectors as basis
    (n, m) = V.shape
    Q = np.zeros((n, m))
    for i in range(m):
        q = V[:, i]
        for j in range(i):
            q = q - np.dot(Q[:, j], V[:, i]) * Q[:, j]
        norm = np.linalg.norm(q)
        if norm < 1e-10:
            raise ValueError("Vectors are linearly dependent or zero vector found.")
        Q[:, i] = q / norm
    return Q

# Example usage:
# Input matrix with basis vectors as columns
V = np.array([[1.0, 1.0, 0.0],
              [1.0, 0.0, 1.0],
              [0.0, 1.0, 1.0]])

Q = gram_schmidt(V)
print("Orthonormal basis:")
print(Q)

import numpy as np

# Check linear dependence by comparing rank with number of vectors
def check_linear_dependence(vectors):
    M = np.column_stack(vectors)
    return np.linalg.matrix_rank(M) < M.shape[1]

# Generate linear combination: sum c_i * v_i
def linear_combination(vectors, coeffs):
    return sum(c * v for c, v in zip(coeffs, vectors))

# Find transition matrix P so that new_basis = old_basis @ P
def transition_matrix(old_basis, new_basis):
    A = np.column_stack(old_basis)
    B = np.column_stack(new_basis)
    return np.linalg.lstsq(A, B, rcond=None)[0]

# Example vectors in R^3
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])
v3 = np.array([7, 8, 9])

vectors = [v1, v2, v3]

print("Linearly dependent?" , check_linear_dependence(vectors))

coeffs = [1, -2, 1]
print("Linear combination:", linear_combination(vectors, coeffs))

old_basis = [v1, v2]
new_basis = [v1 + v2, v2]

P = transition_matrix(old_basis, new_basis)
print("Transition matrix:\n", P)

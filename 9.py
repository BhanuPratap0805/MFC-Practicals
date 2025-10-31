import numpy as np

A = np.array([[4, 1],
              [2, 3]], dtype=float)

# Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)

print("Eigenvalues:")
print(eigenvalues)

print("\nEigenvectors (columns):")
print(eigenvectors)

# Check if eigenvectors form a basis (matrix is diagonalizable if eigenvectors matrix is invertible)
if np.linalg.matrix_rank(eigenvectors) == A.shape[0]:
    print("\nMatrix is diagonalizable.")
else:
    print("\nMatrix is NOT diagonalizable.")

# Cayley-Hamilton theorem verification
# Get characteristic polynomial coefficients (highest to constant)
coeffs = np.poly(A)  # e.g. [1, -trace, det]
n = A.shape[0]

# Evaluate characteristic polynomial at A: p(A) = sum coeffs[i] * A^(n-i)
pA = np.zeros_like(A)
for i, c in enumerate(coeffs):
    pA += c * np.linalg.matrix_power(A, n - i)

print("\nCayley-Hamilton evaluation p(A) (should be close to zero matrix):")
print(pA)
print("Is zero matrix approximation:", np.allclose(pA, np.zeros_like(A)))

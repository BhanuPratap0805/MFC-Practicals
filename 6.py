import numpy as np
from scipy.linalg import null_space

def basis_column_space(A):
    # Column space basis: pivot columns from the RREF or use np.linalg.qr for orthonormal basis
    # Using QR decomposition returns orthonormal basis for column space
    Q, R = np.linalg.qr(A)
    rank = np.linalg.matrix_rank(A)
    return Q[:, :rank]

def basis_null_space(A):
    # Null space basis using scipy.linalg.null_space
    return null_space(A)

def basis_row_space(A):
    # Row space basis: row space of A is column space of A.T
    return basis_column_space(A.T)

def basis_left_null_space(A):
    # Left null space basis is null space of A.T
    return basis_null_space(A.T)

def main():
    rows = int(input("Enter the number of rows of the matrix: "))
    cols = int(input("Enter the number of columns of the matrix: "))

    print(f"Enter the elements of the matrix row-wise (space separated values for each row):")
    matrix = []
    for i in range(rows):
        row = list(map(float, input(f"Row {i+1}: ").strip().split()))
        while len(row) != cols:
            print(f"Please enter exactly {cols} elements.")
            row = list(map(float, input(f"Row {i+1}: ").strip().split()))
        matrix.append(row)
    A = np.array(matrix)

    print("\nMatrix A:")
    print(A)

    print("\nBasis for the Column Space:")
    col_basis = basis_column_space(A)
    print(col_basis)

    print("\nBasis for the Null Space:")
    null_basis = basis_null_space(A)
    print(null_basis)

    print("\nBasis for the Row Space:")
    row_basis = basis_row_space(A)
    print(row_basis)

    print("\nBasis for the Left Null Space:")
    left_null_basis = basis_left_null_space(A)
    print(left_null_basis)

if __name__ == "__main__":
    main()

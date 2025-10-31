import numpy as np

def get_matrix_input():
    n = int(input("Enter the size of the square matrix (n x n): "))
    print(f"Enter the elements of the {n} x {n} matrix row-wise (space separated):")
    matrix = []
    for i in range(n):
        row = list(map(float, input(f"Row {i+1}: ").strip().split()))
        while len(row) != n:
            print(f"Error: Please enter exactly {n} elements.")
            row = list(map(float, input(f"Row {i+1}: ").strip().split()))
        matrix.append(row)
    return np.array(matrix)

def cofactor_matrix(matrix):
    n = matrix.shape[0]
    cofactors = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            minor = np.delete(np.delete(matrix, i, axis=0), j, axis=1)
            sign = (-1) ** (i + j)
            cofactors[i, j] = sign * np.linalg.det(minor)
    return cofactors

def adjoint_matrix(matrix):
    return cofactor_matrix(matrix).T

def main():
    matrix = get_matrix_input()
    print("\nMatrix:")
    print(matrix)

    det = np.linalg.det(matrix)
    print(f"\nDeterminant: {det}")

    cofactors = cofactor_matrix(matrix)
    print("\nCofactor matrix:")
    print(cofactors)

    adjoint = adjoint_matrix(matrix)
    print("\nAdjoint matrix:")
    print(adjoint)

    if np.isclose(det, 0):
        print("\nMatrix is singular, inverse does not exist.")
    else:
        inverse = adjoint / det
        print("\nInverse matrix:")
        print(inverse)

if __name__ == "__main__":
    main()

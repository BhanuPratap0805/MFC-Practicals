import numpy as np

def gauss_jordan(matrix):
    mat = matrix.astype(float)
    rows, cols = mat.shape

    for i in range(rows):
        # Make the diagonal contain all ones
        pivot = mat[i, i]
        if pivot == 0:
            # Search for a non-zero pivot and swap rows if found
            for r in range(i+1, rows):
                if mat[r, i] != 0:
                    mat[[i, r]] = mat[[r, i]]
                    pivot = mat[i, i]
                    break
            else:
                continue  # Skip if pivot can't be found
        mat[i] = mat[i] / pivot

        # Make all other entries in this column zero
        for r in range(rows):
            if r != i:
                factor = mat[r, i]
                mat[r] = mat[r] - factor * mat[i]

    return mat

def main():
    n = int(input("Enter the number of variables: "))
    print(f"Enter the {n}x{n} coefficient matrix row-wise (space-separated):")
    matrix = []
    for i in range(n):
        row = list(map(float, input(f"Row {i+1}: ").split()))
        while len(row) != n:
            print(f"Please enter exactly {n} elements.")
            row = list(map(float, input(f"Row {i+1}: ").split()))
        matrix.append(row)
    matrix = np.array(matrix)

    # Augment with zero column vector for homogeneous system
    augmented = np.hstack((matrix, np.zeros((n,1))))

    rref = gauss_jordan(augmented)

    print("\nReduced Row Echelon Form of the augmented matrix:")
    print(rref)

    # Analyze solution: Check rank and free variables
    rank = np.sum(np.any(np.abs(rref[:, :-1]) > 1e-12, axis=1))
    if rank < n:
        print("\nThe system has infinitely many solutions with free variables.")
    else:
        print("\nThe system has only the trivial solution (all variables = 0).")

if __name__ == "__main__":
    main()

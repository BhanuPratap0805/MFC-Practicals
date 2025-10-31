import numpy as np

def gauss_elimination(A, b):
    A = A.astype(float)
    b = b.astype(float)
    n = len(b)

    # Forward elimination to convert A to upper triangular
    for k in range(n-1):
        for i in range(k+1, n):
            if A[k, k] == 0:
                raise ZeroDivisionError("Zero pivot encountered, cannot proceed.")
            factor = A[i, k] / A[k, k]
            A[i, k:n] = A[i, k:n] - factor * A[k, k:n]
            b[i] = b[i] - factor * b[k]

    # Back substitution
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        if A[i, i] == 0:
            if abs(b[i]) > 1e-12:  # No solution
                raise ValueError("No solution exists")
            else:
                x[i] = 0  # Free variable, choose zero
                continue
        x[i] = (b[i] - np.dot(A[i, i+1:], x[i+1:])) / A[i, i]
    return x

def main():
    n = int(input("Enter number of variables: "))
    print("Enter the coefficients row-wise:")
    A = []
    for i in range(n):
        row = list(map(float, input(f"Row {i+1}: ").strip().split()))
        while len(row) != n:
            print(f"Please enter exactly {n} values.")
            row = list(map(float, input(f"Row {i+1}: ").strip().split()))
        A.append(row)
    A = np.array(A)

    choice = input("Is the system homogeneous? (y/n): ").strip().lower()
    if choice == 'y':
        b = np.zeros(n)
    else:
        print("Enter the constants vector:")
        b = list(map(float, input().strip().split()))
        while len(b) != n:
            print(f"Please enter exactly {n} values.")
            b = list(map(float, input().strip().split()))
        b = np.array(b)

    try:
        solution = gauss_elimination(A, b)
        print("\nSolution vector:")
        print(solution)
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    main()

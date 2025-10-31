import numpy as np

def find_nonzero_row(matrix, pivot_row, col):
    nrows = matrix.shape[0]
    for row in range(pivot_row, nrows):
        if matrix[row, col] != 0:
            return row
    return None

def swap_rows(matrix, row1, row2):
    matrix[[row1, row2]] = matrix[[row2, row1]]

def make_pivot_one(matrix, pivot_row, col):
    pivot_element = matrix[pivot_row, col]
    if pivot_element != 0:
        matrix[pivot_row] = matrix[pivot_row] / pivot_element

def eliminate_below(matrix, pivot_row, col):
    nrows = matrix.shape[0]
    for row in range(pivot_row + 1, nrows):
        factor = matrix[row, col]
        matrix[row] = matrix[row] - factor * matrix[pivot_row]

def row_echelon_form(matrix):
    matrix = matrix.copy().astype(float)
    nrows, ncols = matrix.shape
    pivot_row = 0
    for col in range(ncols):
        nonzero_row = find_nonzero_row(matrix, pivot_row, col)
        if nonzero_row is not None:
            swap_rows(matrix, pivot_row, nonzero_row)
            make_pivot_one(matrix, pivot_row, col)
            eliminate_below(matrix, pivot_row, col)
            pivot_row += 1
    return matrix

def matrix_rank(matrix):
    ref = row_echelon_form(matrix)
    rank = 0
    for row in ref:
        if not np.allclose(row, 0):
            rank += 1
    return rank, ref

# Taking matrix input from user
rows = int(input("Enter the number of rows of the matrix: "))
cols = int(input("Enter the number of columns of the matrix: "))

print(f"Enter the elements of the matrix row-wise (space separated values for each row):")
user_matrix = []
for i in range(rows):
    row_elements = list(map(float, input(f"Row {i+1}: ").strip().split()))
    while len(row_elements) != cols:
        print(f"Error: Please enter exactly {cols} elements.")
        row_elements = list(map(float, input(f"Row {i+1}: ").strip().split()))
    user_matrix.append(row_elements)

user_matrix = np.array(user_matrix)

rank, echelon = matrix_rank(user_matrix)
print("\nRow Echelon Form:")
print(echelon)
print("\nRank of the matrix:", rank)

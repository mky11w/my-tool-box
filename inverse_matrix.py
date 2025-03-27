import numpy as np

# Example matrix (square and invertible)
A = np.array([[4, -2],
              [-2, 2]])

# Check if the matrix is invertible
if np.linalg.det(A) == 0:
    print("Matrix is not invertible.")
else:
    A_inv = np.linalg.inv(A)
    print("Inverse of the matrix:")
    print(A_inv)

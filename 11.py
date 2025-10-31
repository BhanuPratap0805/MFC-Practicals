import numpy as np

# Example: Define scalar field f on a 2D grid (e.g., f = x^2 + y^2)
x = np.linspace(-2, 2, 5)
y = np.linspace(-2, 2, 5)
X, Y = np.meshgrid(x, y)
f = X**2 + Y**2  # scalar field

# Compute the gradient of f
grad_f = np.gradient(f, x, y)  # grad_f is a list of arrays [df/dx, df/dy]

print("df/dx:")
print(grad_f[0])

print("\ndf/dy:")
print(grad_f[1])

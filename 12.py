import numpy as np

# Define 2D vector field components on grid
x = np.linspace(-1, 1, 5)
y = np.linspace(-1, 1, 5)
X, Y = np.meshgrid(x, y)

# Example vector field F = (F_x, F_y)
F_x = 2 * Y**2 + X - 4  # component Fx(x,y)
F_y = np.cos(X)          # component Fy(x,y)

# Compute partial derivatives
dFxdx = np.gradient(F_x, x, axis=1)  # ∂Fx/∂x
dFydy = np.gradient(F_y, y, axis=0)  # ∂Fy/∂y

# Divergence = ∂Fx/∂x + ∂Fy/∂y
divergence = dFxdx + dFydy

print("Divergence of the vector field:")
print(divergence)

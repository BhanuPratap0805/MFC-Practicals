import numpy as np

# Define coordinate arrays
x = np.linspace(-1, 1, 5)
y = np.linspace(-1, 1, 5)
z = np.linspace(-1, 1, 5)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# Example vector field: F = (F_x, F_y, F_z)
F_x = Y  # e.g. Fy component
F_y = -X
F_z = Z**2

# Compute partial derivatives
dFz_dy = np.gradient(F_z, y, axis=1)
dFy_dz = np.gradient(F_y, z, axis=2)
dFx_dz = np.gradient(F_x, z, axis=2)
dFz_dx = np.gradient(F_z, x, axis=0)
dFy_dx = np.gradient(F_y, x, axis=0)
dFx_dy = np.gradient(F_x, y, axis=1)

# Curl components
curl_x = dFz_dy - dFy_dz
curl_y = dFx_dz - dFz_dx
curl_z = dFy_dx - dFx_dy

print("Curl vector field components:")
print("Curl_x:\n", curl_x)
print("Curl_y:\n", curl_y)
print("Curl_z:\n", curl_z)

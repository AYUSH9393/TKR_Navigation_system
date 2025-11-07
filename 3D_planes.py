import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the plane equations
axial_plane = [-8.02312186e-01, 5.96514579e-01, -2.15757537e-02, 6.88014991e+01]
sagittal_plane = [0.00000000e+00, -3.61460642e-02, -9.99346518e-01, -6.05275737e+01]
coronal_plane = [-5.96904646e-01,-8.01787889e-01, 2.90004278e-02, -1.49520646e+02]

# Define a function to plot a plane
def plot_plane(ax, plane, point_on_plane, size=50, color='r', alpha=0.5):
    """
    Plot a plane given its coefficients (A, B, C, D) and a point it passes through.
    """
    A, B, C, D = plane
    x = np.linspace(point_on_plane[0] - size, point_on_plane[0] + size, 10)
    y = np.linspace(point_on_plane[1] - size, point_on_plane[1] + size, 10)
    X, Y = np.meshgrid(x, y)
    Z = (-A * X - B * Y - D) / C  # Solve for Z in the plane equation Ax + By + Cz + D = 0
    
    ax.plot_surface(X, Y, Z, color=color, alpha=alpha)

# Define a point on the plane (you can adjust this based on your data)
point_on_plane = [-34.04928727, -163.11277414, -54.66741306]  # Example point (replace with your actual point) -34.04928727 -163.11277414  -54.66741306

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the planes
plot_plane(ax, axial_plane, point_on_plane, color='r', alpha=0.5)  # Axial plane (red)
plot_plane(ax, sagittal_plane, point_on_plane, color='g', alpha=0.5)  # Sagittal plane (green)
plot_plane(ax, coronal_plane, point_on_plane, color='b', alpha=0.5)  # Coronal plane (blue)

# Plot the point where the planes intersect
ax.scatter(point_on_plane[0], point_on_plane[1], point_on_plane[2], color='k', s=100, label='Intersection Point')

# Set labels and legend
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()

# Show the plot
plt.show()
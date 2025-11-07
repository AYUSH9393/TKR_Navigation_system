import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sympy as sp
from sympy.geometry import Plane, Point, Point3D,Line3D
import math

def distance_point_plane(point, plane):
    """
    Calculate the distance from a 3D point to a plane.
    
    Parameters:
    point (tuple/list): The 3D point (x, y, z).
    plane (tuple/list): The plane coefficients (A, B, C, D) from the equation Ax + By + Cz + D = 0.
    
    Returns:
    float: The shortest distance from the point to the plane.
    """
    x, y, z = point
    A, B, C, D = plane
    numerator = abs(A * x + B * y + C * z + D)
    denominator = (A**2 + B**2 + C**2) ** 0.5
    return numerator / denominator

def project_point_to_plane_from_equation(point, plane_coefficients):
    """
    Project a 3D point onto a plane defined by the equation ax + by + cz + d = 0,
    and return the projected point and the signed distance.

    Parameters:
    - point: The fixed 3D reference point (numpy array of shape (3,)).
    - plane_coefficients: The coefficients of the plane equation [a, b, c, d].

    Returns:
    - projected_point: The projected point on the plane (numpy array of shape (3,)).
    - signed_distance: The signed distance between the original point and the plane.
                       Negative if the projected point is in the direction of the original point,
                       positive otherwise.
    """
    # Extract the plane coefficients
    a, b, c, d = plane_coefficients
    
    # Normal vector of the plane
    plane_normal = np.array([a, b, c])
    
    #plane_normal = np.cross(c-a,d-b)

    # Ensure the normal vector is normalized
    plane_normal = plane_normal / np.linalg.norm(plane_normal)
    
    # Calculate the signed distance from the point to the plane
    signed_distance = (a * point[0] + b * point[1] + c * point[2] + d) / np.linalg.norm([a, b, c])
    
    # Calculate the projected point
    projected_point = point - signed_distance * plane_normal
    
    # Determine the direction of the projected point relative to the original point
    direction_vector = projected_point - point
    direction_dot_product = np.dot(direction_vector, plane_normal)
    
    # Adjust the sign of the distance based on the direction
    if direction_dot_product > 0:
        signed_distance = abs(signed_distance)  # Negative if in the direction of the original point
    else:
        signed_distance = -abs(signed_distance)  # Positive otherwise
    
    return projected_point, signed_distance

# def project_point_to_plane_sympy(point, plane_coefficients):
#     a  = Plane(Point3D(*point), normal_vector=plane_coefficients)
#     b = Point3D()
#     a.distance(b)
#     c = Line3D(Point3D(*point), Point3D(*point))
#     a.distance(c)
#     return a.distance(b), a.distance(c)

def distance_point_to_plane(x1, y1, z1, A, B, C, D):
    """
    Calculate the distance from a point (x1, y1, z1) to a plane Ax + By + Cz + D = 0.

    Parameters:
    x1, y1, z1 (float): Coordinates of the point.
    A, B, C, D (float): Coefficients of the plane equation.

    Returns:
    float: The distance from the point to the plane.
    """
    numerator = abs(A * x1 + B * y1 + C * z1 + D)
    denominator = math.sqrt(A**2 + B**2 + C**2)
    
    if denominator == 0:
        raise ValueError("The coefficients A, B, and C cannot all be zero.")
    
    distance = numerator / denominator
    return distance

# def point_plane_distance(point, plane_normal, plane_point):
#     """
#     Calculates the distance between a point and a plane. 
    
#     Args:
#         point: A numpy array representing the point coordinates.
#         plane_normal: A numpy array representing the plane normal vector.
#         plane_point: A numpy array representing a point on the plane.
    
#     Returns:
#         The distance between the point and the plane.
#     """
    
#     vector_to_point = point - plane_point
#     distance = np.dot(vector_to_point, plane_normal) / np.linalg.norm(plane_normal)
#     return distance

def point_plane_distance(point, plane_normal, plane_point):
    """
    Optimized version of your current function with better numerical stability.
    Returns SIGNED distance (matches your current behavior before applying -int()).
    
    Args:
        point: numpy array [x, y, z] - coordinates of the point
        plane_normal: numpy array [a, b, c] - normal vector of the plane
        plane_point: numpy array [x0, y0, z0] - any point on the plane
    
    Returns:
        float: signed distance (positive/negative indicates which side of plane)
    """
    # Vector from plane point to the given point
    vector_to_point = point - plane_point
    
    # Calculate dot product manually for better performance
    dot_product = (vector_to_point[0] * plane_normal[0] + 
                  vector_to_point[1] * plane_normal[1] + 
                  vector_to_point[2] * plane_normal[2])
    
    # Calculate normal magnitude manually with numerical stability
    normal_magnitude = math.sqrt(plane_normal[0]**2 + 
                                plane_normal[1]**2 + 
                                plane_normal[2]**2)
    
    # Add small epsilon to prevent division by zero
    if normal_magnitude < 1e-12:
        return 0.0
    
    # Return signed distance (same as your current function)
    return dot_product / normal_magnitude

def calculate_plane_normal_main(corners):
    """
    Calculate the normal vector of a plane given the four corners of an ArUco marker.

    Args:
        corners: A 4x3 numpy array where each row represents the 3D coordinates of a corner.

    Returns:
        A normalized normal vector of the plane.
    """
    if corners.shape != (4, 3):
        raise ValueError("Input corners must be a 4x3 numpy array.")
    
    # Define two vectors lying on the plane using three corner points
    vector1 = corners[1] - corners[0]
    vector2 = corners[2] - corners[0]
    
    # Compute the cross product of the two vectors to get the normal vector
    plane_normal = np.cross(vector1, vector2)
    
    # Normalize the normal vector
    plane_normal = plane_normal / np.linalg.norm(plane_normal)
    
    return plane_normal
    
def calculate_plane_normal(corners):  
    """  
    Calculate the normal vector of a plane given the four corners of an ArUco marker.  
    Uses all 4 corners to be more robust to noise.
  
    Args:  
        corners: A 4x3 numpy array where each row represents the 3D coordinates of a corner.  
  
    Returns:  
        A normalized normal vector of the plane.  
    """  
    if corners.shape != (4, 3):  
        raise ValueError("Input corners must be a 4x3 numpy array.")  
      
    # Method 1: Average multiple cross products
    normals = []
    for i in range(4):
        v1 = corners[(i+1)%4] - corners[i]
        v2 = corners[(i+2)%4] - corners[i]
        normal = np.cross(v1, v2)
        if np.linalg.norm(normal) > 1e-6:  # Avoid zero-length normals
            normals.append(normal / np.linalg.norm(normal))
    
    # Average all normals
    plane_normal = np.mean(normals, axis=0)
    plane_normal = plane_normal / np.linalg.norm(plane_normal)
      
    return plane_normal

def project_point_to_plane_sympy(point, plane_coefficients):
    """
    Project a 3D point onto a plane defined by the equation ax + by + cz + d = 0,
    and return the projected point and the signed distance using SymPy.

    Parameters:
    - point: The fixed 3D reference point (list or tuple of 3 elements).
    - plane_coefficients: The coefficients of the plane equation [a, b, c, d].

    Returns:
    - projected_point: The projected point on the plane (SymPy Point3D).
    - signed_distance: The signed distance between the original point and the plane.
    """
    # Extract the plane coefficients
    a, b, c, d = plane_coefficients
    
    # Define the symbols
    x, y, z = sp.symbols('x y z')
    
    # Define the plane equation
    plane_eq = a*x + b*y + c*z + d
    
    # Define the point as a SymPy Point3D
    point_sympy = sp.Point3D(point[0], point[1], point[2])
    
    # Define the plane as a SymPy Plane
    plane_sympy = sp.Plane(sp.Point3D(0, 0, -d/c), normal_vector=(a, b, c)) if c != 0 else \
                  sp.Plane(sp.Point3D(0, -d/b, 0), normal_vector=(a, b, c)) if b != 0 else \
                  sp.Plane(sp.Point3D(-d/a, 0, 0), normal_vector=(a, b, c))
    
    # Project the point onto the plane
    projected_point = plane_sympy.projection(point_sympy)
    
    # Calculate the signed distance
    signed_distance = plane_sympy.distance(point_sympy)
    
    # Determine the sign of the distance based on the position of the point relative to the plane
    # The distance is positive if the point is on the side of the plane in the direction of the normal vector
    # and negative otherwise.
    # We can use the plane equation to determine the sign
    value_at_point = plane_eq.subs({x: point[0], y: point[1], z: point[2]})
    if value_at_point > 0:
        signed_distance = -signed_distance
    
    return projected_point, signed_distance

def project_on_plane(vector, plane_normal):
    """
    Projects a vector onto a plane defined by its normal.

    Parameters:
    - vector: The vector to project onto the plane (numpy array of shape (3,)).
    - plane_normal: The normal vector defining the plane (numpy array of shape (3,)).

    Returns:
    - The orthogonal projection of the vector onto the plane.
    """
    # Calculate the dot product of the plane normal with itself
    num = np.dot(plane_normal, plane_normal)
    
    # If the plane normal is too small (close to zero), return the original vector
    if num < np.finfo(float).eps:
        return vector
    
    # Calculate the dot product of the vector and the plane normal
    num2 = np.dot(vector, plane_normal)
    
    # Compute the projection of the vector onto the plane
    projection = vector - (num2 / num) * plane_normal
    
    return projection

# def signed_angle(from_vector, to_vector, axis):
#     """
#     Calculates the signed angle between two vectors around a specified axis.

#     Parameters:
#     - from_vector: The starting vector (numpy array of shape (3,)).
#     - to_vector: The target vector (numpy array of shape (3,)).
#     - axis: The axis around which the angle is measured (numpy array of shape (3,)).

#     Returns:
#     - The signed angle in degrees.
#     """
#     # Calculate the unsigned angle between the two vectors
#     angle = np.degrees(np.arccos(np.clip(np.dot(from_vector, to_vector) / (np.linalg.norm(from_vector) * np.linalg.norm(to_vector)), -1.0, 1.0)))
    
#     # Calculate the cross product components
#     cross_y = from_vector[1] * to_vector[2] - from_vector[2] * to_vector[1]
#     cross_z = from_vector[2] * to_vector[0] - from_vector[0] * to_vector[2]
#     cross_x = from_vector[0] * to_vector[1] - from_vector[1] * to_vector[0]
    
#     # Calculate the sign of the angle using the dot product of the axis and the cross product
#     sign = np.sign(axis[0] * cross_x + axis[1] * cross_y + axis[2] * cross_z)
    
#     # Return the signed angle
#     return angle * sign ,angle

def signed_angle(from_vector, to_vector, axis):
    """
    Calculates the signed angle between two vectors around a specified axis.
    """
    # Normalize input vectors
    from_vector = from_vector / np.linalg.norm(from_vector)
    to_vector = to_vector / np.linalg.norm(to_vector)
    axis = axis / np.linalg.norm(axis)
    
    # Calculate angle
    dot = np.dot(from_vector, to_vector)
    dot = np.clip(dot, -1.0, 1.0)
    angle = np.degrees(np.arccos(dot))
    
    # Calculate cross product for sign
    cross = np.cross(from_vector, to_vector)
    sign = np.sign(np.dot(cross, axis))
    
    # Handle parallel vectors
    if np.allclose(cross, 0):
        if dot < 0:
            return 180.0, 180.0
        return 0.0, 0.0
        
    return angle * sign, angle

def plane_from_square_corners(corners):
    """
    Calculate the plane equation (Ax + By + Cz + D = 0) from the four corner points of a square.
    
    Parameters:
    corners (list of lists/arrays): List of 4 corner points, each as a list or array [x, y, z].
    
    Returns:
    tuple: Coefficients (A, B, C, D) of the plane equation.
    """
    if len(corners) != 4:
        raise ValueError("Exactly 4 corner points are required.")
    
    # Convert corner points to numpy arrays for easier calculations
    p1, p2, p3, p4 = [np.array(point) for point in corners]
    
    # Calculate two vectors in the plane
    v1 = p2 - p1
    v2 = p3 - p1
    
    # Calculate the normal vector of the plane using the cross product
    normal = np.cross(v1, v2)
    
    # Normalize the normal vector (optional but recommended for consistency)
    normal = normal / np.linalg.norm(normal)
    
    # Calculate D using one of the points
    A, B, C = normal
    #D = -np.dot(normal, p1)
    D = -np.dot(normal, np.mean(corners, axis=0))
    
    return (A, B, C, D), normal, D



# # Example usage
# corners = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]  # A square in the xy-plane
# plane_coefficients = plane_from_square_corners(corners)
# print("Plane equation coefficients (A, B, C, D):", plane_coefficients)

def line_of_intersection(plane1, plane2):
    """
    Compute the line of intersection between two planes.
    
    Parameters:
    plane1 (tuple): Coefficients (A1, B1, C1, D1) of the first plane equation.
    plane2 (tuple): Coefficients (A2, B2, C2, D2) of the second plane equation.
    
    Returns:
    tuple: A point on the line (p0) and the direction vector (d) of the line.
    """
    A1, B1, C1, D1 = plane1
    A2, B2, C2, D2 = plane2
    
    # Normal vectors of the planes
    n1 = np.array([A1, B1, C1])
    n2 = np.array([A2, B2, C2])
    
    # Direction vector of the line (cross product of the normals)
    d = np.cross(n1, n2)
    
    # Check if planes are parallel (cross product is zero)
    if np.linalg.norm(d) < 1e-10:
        raise ValueError("The planes are parallel and do not intersect.")
    
    # Find a point on the line by solving the system of equations
    # We set z = 0 and solve for x and y (if possible)
    try:
        # Solve the system:
        # A1*x + B1*y = -D1
        # A2*x + B2*y = -D2
        A = np.array([[A1, B1], [A2, B2]])
        b = np.array([-D1, -D2])
        x, y = np.linalg.solve(A, b)
        p0 = np.array([x, y, 0])  # Point on the line with z = 0
    except np.linalg.LinAlgError:
        # If the system is singular, set y = 0 and solve for x and z
        try:
            A = np.array([[A1, C1], [A2, C2]])
            b = np.array([-D1, -D2])
            x, z = np.linalg.solve(A, b)
            p0 = np.array([x, 0, z])  # Point on the line with y = 0
        except np.linalg.LinAlgError:
            # If still singular, set x = 0 and solve for y and z
            A = np.array([[B1, C1], [B2, C2]])
            b = np.array([-D1, -D2])
            y, z = np.linalg.solve(A, b)
            p0 = np.array([0, y, z])  # Point on the line with x = 0
    
    return p0, d

def create_line(point1, point2):
    """
    Create a line from two 3D points.
    
    Parameters:
    point1 (list/np.array): First point [x1, y1, z1].
    point2 (list/np.array): Second point [x2, y2, z2].
    
    Returns:
    tuple: A point on the line (p0) and the direction vector (d).
    """
    p0 = np.array(point1)
    d = np.array(point2) - p0
    return p0, d

def plane_perpendicular_to_line(line, point_on_plane):
    """
    Generate a plane perpendicular to a line, passing through a specified point.
    
    Parameters:
    line (tuple): The line (p0, direction) returned by `create_line`.
    point_on_plane (list/np.array): A point [x, y, z] that the plane must pass through.
    
    Returns:
    tuple: Plane coefficients (A, B, C, D) for the equation Ax + By + Cz + D = 0.
    """
    p0, direction = line
    normal = direction
    
    # Normalize the normal vector (optional but recommended)
    normal = normal / np.linalg.norm(normal)
    
    # Calculate D using the point_on_plane
    A, B, C = normal
    D = -np.dot(normal, point_on_plane)
    
    return (A, B, C, D) ,normal

def plot_plane(plane_coefficients, x_range=(-10, 10), y_range=(-10, 10)):
    """
    Visualize a plane in 3D using matplotlib.
    
    Parameters:
    plane_coefficients (tuple): Coefficients (A, B, C, D) of the plane equation Ax + By + Cz + D = 0.
    x_range (tuple): Range of x values for the grid.
    y_range (tuple): Range of y values for the grid.
    """
    A, B, C, D = plane_coefficients
    
    # Create a grid of x and y values
    x = np.linspace(x_range[0], x_range[1], 100)
    y = np.linspace(y_range[0], y_range[1], 100)
    X, Y = np.meshgrid(x, y)
    
    # Solve for z using the plane equation
    Z = (-A * X - B * Y - D) / C
    
    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the plane
    ax.plot_surface(X, Y, Z, alpha=0.5, color='blue', label='Plane')
    
    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Set title
    ax.set_title(f'Plane: {A:.2f}x + {B:.2f}y + {C:.2f}z + {D:.2f} = 0')
    
    plt.show()

def plane_from_point_and_normal(p0, normal):
    """
    Generate a plane from a point and a normal vector.

    Parameters:
        p0 (numpy.ndarray): A point on the plane, shape (3,).
        normal (numpy.ndarray): The normal vector to the plane, shape (3,).

    Returns:
        numpy.ndarray: The plane coefficients [A, B, C, D] for the plane equation Ax + By + Cz + D = 0.
    """
    # Ensure inputs are numpy arrays
    p0 = np.asarray(p0)
    normal = np.asarray(normal)

    # Normalize the normal vector
    normal = normal / np.linalg.norm(normal)

    # Calculate D using the dot product
    D = -np.dot(normal, p0)

    # Return the plane coefficients [A, B, C, D]
    return np.append(normal, D)

def rotate_vector(vector, axis, angle):
    """
    Rotate a vector around a given axis by a specified angle.
    
    Parameters:
    vector (np.array): The original direction vector.
    axis (np.array): The axis to rotate around.
    angle (float): The rotation angle in radians.
    
    Returns:
    np.array: The rotated vector.
    """
    axis = axis / np.linalg.norm(axis)  # Normalize axis
    cos_theta = np.cos(angle)
    sin_theta = np.sin(angle)
    cross_prod = np.cross(axis, vector)
    dot_prod = np.dot(axis, vector)
    
    rotated_vector = (
        cos_theta * vector +
        sin_theta * cross_prod +
        (1 - cos_theta) * dot_prod * axis
    )
    return rotated_vector


def angle_between_normals(normal1, normal2):
    """
    Calculate the angle between two normal vectors in degrees.

    Parameters:
        normal1 (numpy.ndarray): The first normal vector, shape (3,).
        normal2 (numpy.ndarray): The second normal vector, shape (3,).

    Returns:
        float: The angle between the two vectors in degrees.
    """
    # Ensure the inputs are numpy arrays
    normal1 = np.asarray(normal1)
    normal2 = np.asarray(normal2)

    # Compute the dot product of the two vectors
    dot_product = np.dot(normal1, normal2)

    # Compute the magnitudes (lengths) of the vectors
    magnitude1 = np.linalg.norm(normal1)
    magnitude2 = np.linalg.norm(normal2)

    # Compute the cosine of the angle
    cos_theta = dot_product / (magnitude1 * magnitude2)

    # Handle numerical inaccuracies (ensure cos_theta is within [-1, 1])
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    # Compute the angle in radians
    theta_radians = np.arccos(cos_theta)

    # Convert the angle to degrees
    theta_degrees = np.degrees(theta_radians)

    return theta_degrees

def create_line_single_point(point, direction):
    """
    Create a parametric function representing a 3D line.

    Parameters:
    - point (tuple/list): The 3D starting point (x0, y0, z0).
    - direction (tuple/list): The direction vector (dx, dy, dz).

    Returns:
    - function: A parametric function that takes a parameter `t` and returns a point (x, y, z) on the line.
    """
    x0, y0, z0 = point
    dx, dy, dz = direction

    def line_function(t):
        """
        Parametric function for the line.

        Parameters:
        - t (float): The parameter along the line.

        Returns:
        - tuple: The (x, y, z) coordinates of the point on the line at parameter `t`.
        """
        x = x0 + dx * t
        y = y0 + dy * t
        z = z0 + dz * t
        return (x, y, z)

    return line_function

def distance_to_plane_hit(line, plane_normal, plane_d):
    """
    Calculate the distance between the starting point of the line and the point where the line intersects the plane.

    Parameters:
    - line (function): The parametric line function returned by `create_line`.
    - plane_normal (tuple/list): The normal vector (a, b, c) of the plane.
    - plane_d (float): The constant term `d` in the plane equation `ax + by + cz + d = 0`.

    Returns:
    - float: The distance between the starting point and the intersection point.
    - tuple: The intersection point (x, y, z) on the plane.
    """
    # Extract the starting point of the line (t=0)
    x0, y0, z0 = line(0)

    # Extract the direction vector of the line
    dx, dy, dz = line(1)[0] - x0, line(1)[1] - y0, line(1)[2] - z0

    # Plane equation: ax + by + cz + d = 0
    a, b, c = plane_normal

    # Substitute the parametric line equations into the plane equation and solve for t
    numerator = -(a * x0 + b * y0 + c * z0 + plane_d)
    denominator = a * dx + b * dy + c * dz

    if denominator == 0:
        raise ValueError("The line is parallel to the plane and does not intersect.")

    t = numerator / denominator

    # Find the intersection point
    intersection_point = line(t)

    # Calculate the Euclidean distance between the starting point and the intersection point
    distance = np.sqrt(
        (intersection_point[0] - x0) ** 2 +
        (intersection_point[1] - y0) ** 2 +
        (intersection_point[2] - z0) ** 2
    )
    signed_distance = distance * np.sign(denominator)

    return signed_distance, intersection_point

def get_sagittal_plane(axial_plane, point_on_plane):
    A, B, C, D = axial_plane
    new_A = C
    new_B = B
    new_C = -A
    new_D = -np.dot([new_A, new_B, new_C], point_on_plane)
    normal = np.array([new_A, new_B, new_C])
    #new_D = -D
    return (new_A, new_B, new_C, new_D),normal

def get_coronal_plane(axial_plane, point_on_plane):
    A, B, C, D = axial_plane
    new_A = A
    new_B = C
    new_C = -B
    new_D = -np.dot([new_A, new_B, new_C], point_on_plane)
    normal = np.array([new_A, new_B, new_C])
    #new_D = -D
    return (new_A, new_B, new_C, new_D),normal

# def angle_between_planes(p1, p2, degrees=True):
#     """
#     Calculate the angle between two planes given their coefficients.
    
#     Parameters:
#     plane1 (tuple): Coefficients (A, B, C, D) of the first plane (Ax + By + Cz + D = 0).
#     plane2 (tuple): Coefficients (A, B, C, D) of the second plane.
#     degrees (bool): If True, return angle in degrees; if False, return in radians.
    
#     Returns:
#     float: The angle between the planes (acute angle, 0 to 90 degrees or 0 to π/2 radians).
#     """
#     # Extract normal vectors (A, B, C) from plane coefficients
#     normal1 = np.array([p1[0], p1[1], p1[2]])
#     normal2 = np.array([p2[0], p2[1], p2[2]])
    
#     # Calculate the dot product
#     dot_product = np.dot(normal1, normal2)
    
#     # Calculate magnitudes of the normal vectors
#     mag1 = np.linalg.norm(normal1)
#     mag2 = np.linalg.norm(normal2)
    
#     # Avoid division by zero
#     if mag1 == 0 or mag2 == 0:
#         raise ValueError("One or both planes have a zero normal vector, which is invalid.")
    
#     # Calculate cosine of the angle (ensure acute angle with absolute value)
#     cos_theta = abs(dot_product) / (mag1 * mag2)
    
#     # Handle numerical precision issues (cos_theta should be between 0 and 1)
#     cos_theta = min(1.0, max(0.0, cos_theta))
    
#     # Calculate the angle in radians
#     theta = np.arccos(cos_theta)
    
#     # Convert to degrees if requested
#     if degrees:
#         theta = np.degrees(theta)
    
#     return theta

#this for point projection on plane
def angle_between_planes(p1, p2, degrees=True):
    """
    Calculate the angle between two planes.
    
    Parameters:
    p1 (sympy.Plane or tuple): The first plane (either a sympy.Plane object or coefficients [A, B, C, D]).
    p2 (sympy.Plane or tuple): The second plane (either a sympy.Plane object or coefficients [A, B, C, D]).
    degrees (bool): If True, return angle in degrees; if False, return in radians.
    
    Returns:
    float: The angle between the planes (acute angle, 0 to 90 degrees or 0 to π/2 radians).
    """
    # Extract normal vectors and coefficients for p1
    if isinstance(p1, Plane):
        normal1 = np.array(p1.normal_vector, dtype=float)
    else:
        normal1 = np.array([p1[0], p1[1], p1[2]], dtype=float)
    
    # Extract normal vectors and coefficients for p2
    if isinstance(p2, Plane):
        normal2 = np.array(p2.normal_vector, dtype=float)
    else:
        normal2 = np.array([p2[0], p2[1], p2[2]], dtype=float)
    
    # Calculate the dot product
    dot_product = np.dot(normal1, normal2)
    
    # Calculate magnitudes of the normal vectors
    mag1 = np.linalg.norm(normal1)
    mag2 = np.linalg.norm(normal2)
    
    # Avoid division by zero
    if mag1 == 0 or mag2 == 0:
        raise ValueError("One or both planes have a zero normal vector, which is invalid.")
    
    # Calculate cosine of the angle (ensure acute angle with absolute value)
    cos_theta = abs(dot_product) / (mag1 * mag2)
    
    # Handle numerical precision issues (cos_theta should be between 0 and 1)
    cos_theta = min(1.0, max(0.0, cos_theta))
    
    # Calculate the angle in radians
    theta = np.arccos(cos_theta)
    
    # Convert to degrees if requested
    if degrees:
        theta = np.degrees(theta)
    
    return theta


def generate_orthogonal_planes(line, point_on_plane):
    # Generate axial plane
    axial_plane, axial_normal = plane_perpendicular_to_line(line, point_on_plane)
    
    # Find an arbitrary vector not collinear with axial_normal
    arbitrary = np.array([1.0, 0.0, 0.0])
    if np.abs(np.dot(axial_normal, arbitrary)) > 0.99:  # Check if vectors are nearly parallel
        arbitrary = np.array([0.0, 1.0, 0.0])
    
    # Compute sagittal plane normal (perpendicular to axial_normal and arbitrary vector)
    sagittal_normal = np.cross(axial_normal, arbitrary)
    sagittal_normal /= np.linalg.norm(sagittal_normal)
    
    # Compute coronal plane normal (perpendicular to axial_normal and sagittal_normal)
    coronal_normal = np.cross(axial_normal, sagittal_normal)
    coronal_normal /= np.linalg.norm(coronal_normal)
    
    # Calculate D for sagittal and coronal planes
    A_sag, B_sag, C_sag = sagittal_normal
    D_sag = -np.dot(sagittal_normal, point_on_plane)
    sagittal_plane = (A_sag, B_sag, C_sag, D_sag)
    
    A_cor, B_cor, C_cor = coronal_normal
    D_cor = -np.dot(coronal_normal, point_on_plane)
    coronal_plane = (A_cor, B_cor, C_cor, D_cor)
    
    return {
        'axial': axial_plane,
        'sagittal': sagittal_plane,
        'coronal': coronal_plane
    }, (axial_normal, sagittal_normal, coronal_normal)

# def generate_sagittal_plane_with_projection(pivot_ref_list, axial_plane):
#     """
#     Generates the sagittal plane by projecting points onto the axial plane.

#     Parameters:
#     - pivot_ref_list: List of 3 reference points [origin, point1, point2].
#     - axial_plane: The axial plane (sympy.Plane) onto which points are projected.

#     Returns:
#     - sagittal_plane: Plane perpendicular to the line formed by projected points.
#     - Angle_axial_sig: Angle between sagittal and axial planes.
#     """
#     point0 = tuple(pivot_ref_list[0])
#     point1 = tuple(pivot_ref_list[1])
#     point2 = tuple(pivot_ref_list[2])
#     # Ensure axial_plane is a sympy.Plane object
    
#     # if not isinstance(axial_plane, Plane):
#     #     raise TypeError("axial_plane must be a sympy.Plane object.")
    
#     # Project pivot_ref_list[1] and pivot_ref_list[2] onto the axial plane
#     projected_p1 = axial_plane.projection(Point3D(*point1))
#     projected_p2 = axial_plane.projection(Point3D(*point2))

#     # Create a line from the projected points (direction vector)
#     #line_direction = projected_p1 - projected_p2
#     line_vector = projected_p1 - projected_p2
#     # Use absolute dominant axis to determine direction (e.g., medial-lateral)
#     abs_line = [abs(line_vector.x), abs(line_vector.y), abs(line_vector.z)]
#     dominant_axis = np.argmax(abs_line)
    
#     # Ensure direction is consistent (e.g., positive along dominant anatomical axis)
#     if line_vector[dominant_axis] < 0:
#         line_vector = -line_vector
#     Sig_Dir = np.array([line_vector.x, line_vector.y, line_vector.z], dtype=float)

#     # Create sagittal plane: perpendicular to the line, passing through pivot_ref_list[0]
#     sagittal_plane = Plane(Point3D(*point1), normal_vector=Sig_Dir)

#     # Extract the plane equation coefficients (A, B, C, D) for Ax + By + Cz + D = 0
#     A, B, C = sagittal_plane.normal_vector
#     D = -sagittal_plane.p1.dot(sagittal_plane.normal_vector)
#     sagittal_plane_eq = np.array([A, B, C, D], dtype=float)

#     # Calculate angle between sagittal and axial planes
#     Angle_axial_sig = float(axial_plane.angle_between(sagittal_plane).evalf())

#     return sagittal_plane_eq, Angle_axial_sig, Sig_Dir

def generate_sagittal_plane_with_projection(pivot_ref_list, axial_plane,CT_side):
    """
    Generates the sagittal plane by projecting points onto the axial plane.
    """
    #global CT_side
    # Fix: Add input validation
    if len(pivot_ref_list) != 3:
        raise ValueError("pivot_ref_list must contain exactly 3 points")

    point0 = tuple(pivot_ref_list[0])
    point1 = tuple(pivot_ref_list[1])
    point2 = tuple(pivot_ref_list[2])

    # Project points onto the axial plane
    projected_p1 = axial_plane.projection(Point3D(*point1))
    projected_p2 = axial_plane.projection(Point3D(*point2))

    # Create a line from the projected points
    line_vector = projected_p1 - projected_p2
    line_vector = np.array([line_vector.x, line_vector.y, line_vector.z], dtype=float)

    # Fix: Ensure consistent direction based on anatomical reference
    line_vector /= np.linalg.norm(line_vector)
    
    # Fix: Use anatomical reference for consistent orientation
    print("CT_side", CT_side)
    if CT_side == "R":  # Assuming CT_side is accessible
        reference_direction = np.array([1, 0, 0])  # Right side reference
    else:
        reference_direction = np.array([-1, 0, 0])  # Left side reference
        
    if np.dot(line_vector, reference_direction) < 0:
        line_vector = -line_vector

    Sig_Dir = line_vector

    # Create sagittal plane
    sagittal_plane = Plane(Point3D(*point1), normal_vector=Sig_Dir)
    
    # Extract plane coefficients
    A, B, C = sagittal_plane.normal_vector
    D = -sagittal_plane.p1.dot(sagittal_plane.normal_vector)
    sagittal_plane_eq = np.array([A, B, C, D], dtype=float)

    # Calculate angle between planes
    Angle_axial_sig = float(axial_plane.angle_between(sagittal_plane).evalf())
    
    return sagittal_plane_eq, Angle_axial_sig, Sig_Dir

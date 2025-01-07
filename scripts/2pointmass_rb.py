import sympy as sp
import numpy as np


# Define symbols
m = sp.symbols('m')  # Mass
I = sp.MatrixSymbol('I', 3, 3)  # Inertia tensor (3x3 matrix)
TCG = sp.MatrixSymbol('TCG', 3, 1)  # Center of mass position in FS (3x1 vector)
V = sp.MatrixSymbol('V', 3, 1)  # Linear velocity in FS (3x1 vector)
W = sp.MatrixSymbol('W', 3, 1)  # Angular velocity in FS (3x1 vector)
Wc = sp.MatrixSymbol('Wc', 3, 1)  # Angular velocity in CS (3x1 vector)
V_dot = sp.MatrixSymbol('V_dot', 3, 1)  # Linear acceleration in FS (3x1 vector)
W_dot = sp.MatrixSymbol('W_dot', 3, 1)  # Angular acceleration in FS (3x1 vector)
F = sp.MatrixSymbol('F', 3, 1)  # External force in FS (3x1 vector)
M = sp.MatrixSymbol('M', 3, 1)  # External moment in FS (3x1 vector)

# Identity matrix (3x3)
identity = sp.eye(3)

# Function to create skew-symmetric (cross-product) matrices
def skew(vec):
    """Returns the skew-symmetric matrix of a 3x1 vector."""
    return sp.Matrix([
        [0, -vec[2, 0], vec[1, 0]],
        [vec[2, 0], 0, -vec[0, 0]],
        [-vec[1, 0], vec[0, 0], 0]
    ])

# Convert TCG to an explicit matrix for manipulation
TCG_explicit = sp.Matrix(TCG)

# Compute the skew-symmetric matrix for TCG
TCG_skew = skew(TCG_explicit)

# Construct the combined matrix
upper_block = sp.Matrix.hstack(m * identity, -m * TCG_skew)
lower_block = sp.Matrix.hstack(m * TCG_skew, sp.Matrix(I))
m_first = sp.Matrix.vstack(upper_block, lower_block)

# Display the matrix
print("Combined Matrix:")
sp.pprint(m)

u_dot = sp.Matrix.vstack(sp.Matrix(V_dot), sp.Matrix(W_dot))

# Display u_dot
print("\nVector u_dot:")
sp.pprint(u_dot)

u = sp.Matrix.vstack(sp.Matrix(V), sp.Matrix(W))
w_skew = skew(sp.Matrix(W))
v_skew = skew(sp.Matrix(V))

u_skew_upper = sp.Matrix.hstack(w_skew, sp.zeros(3,3))
u_skew_lower = sp.Matrix.hstack(v_skew, w_skew)
u_skew = sp.Matrix.vstack(u_skew_upper, u_skew_lower)

# Display u_skew
print("\nMatrix u_skew:")
sp.pprint(u_skew)


f = sp.Matrix.vstack(sp.Matrix(F), sp.Matrix(M))

# Display f
print("\nVector f:")
sp.pprint(f)

lhs = m_first@u_dot + u_skew@m_first@u
rhs = f
# Full equation of motion
equation_of_motion = sp.Eq(lhs, rhs)

# Display the equation of motion
print("\nEquation of Motion:")  
sp.pprint(equation_of_motion)

wc_skew = skew(sp.Matrix(Wc))
u_skew_upper = sp.Matrix.hstack(wc_skew, sp.zeros(3,3))
u_skew_lower = sp.Matrix.hstack(v_skew, wc_skew)
u_skew = sp.Matrix.vstack(u_skew_upper, u_skew_lower)
u_skew_upper = sp.Matrix.hstack(wc_skew, sp.zeros(3,3))
u_skew_lower = sp.Matrix.hstack(sp.zeros(3,3), wc_skew)
u_skew1 = sp.Matrix.vstack(u_skew_upper, u_skew_lower)

# Construct the combined matrix
upper_block = sp.Matrix.hstack(m * identity, -m * TCG_skew)
lower_block = sp.Matrix.hstack(sp.zeros(3), sp.Matrix(I))
m_second = sp.Matrix.vstack(upper_block, lower_block)


sp.pprint(sp.simplify(u_skew1@m_second@u))
sp.pprint(sp.simplify(u_skew@m_second@u))


raise NotImplementedError("Please complete the code snippet")

# Substituting omega (W) and omega_dot (W_dot) with zero vectors
W_zero = sp.Matrix([[0], [0], [0]])  # Zero angular velocity
W_dot_zero = sp.Matrix([[0], [0], [0]])  # Zero angular acceleration

# Substituting into the equations of motion
equation_of_motion_subs = equation_of_motion.subs({
    W: W_zero,
    W_dot: W_dot_zero
})

# Display the simplified equation of motion
print("\nEquation of Motion with omega and omega_dot set to 0:")
sp.pprint(equation_of_motion_subs)


import numpy as np

def calculate_cg_and_inertia(masses, positions):
    """
    Calculate the center of gravity (CG) and moment of inertia tensor for a system of point masses.

    Parameters:
        masses (list): List of masses (m1, m2, ...).
        positions (list): List of positions ([x, y, z] for each mass).

    Returns:
        tuple: (center_of_gravity, inertia_tensor)
    """
    masses = np.array(masses)
    positions = np.array(positions)

    # Calculate center of gravity
    total_mass = np.sum(masses)
    center_of_gravity = np.sum(positions.T * masses, axis=1) / total_mass

    # Calculate moment of inertia tensor
    inertia_tensor = np.zeros((3, 3))
    for i, mass in enumerate(masses):
        r = positions[i] - center_of_gravity
        r_outer = np.outer(r, r)
        inertia_tensor += mass * (np.dot(r, r) * np.eye(3) - r_outer)

    return center_of_gravity, inertia_tensor

# Define point masses and their positions
masses = [0, 10,5,5,1]
positions = [[0, 0, 0], [1, 0, 12], [0, 1, 10], [0, -1, 10], [-1, 0, 12]]

# Calculate CG and inertia tensor
p_cg, inertia_tensor = calculate_cg_and_inertia(masses, positions)

print("Center of Gravity (CG):", p_cg)
print("Moment of Inertia Tensor:")
print(inertia_tensor)

# Define the specific values at p1
TCG_p1 = sp.Matrix(-p_cg)  # Center of mass at p1
# V_p1 = sp.Matrix([[0], [0], [0]])    # Linear velocity at p1
# V_dot_p1 = sp.Matrix([[0], [0], [0]])  # Linear acceleration at p1
# F_p1 = sp.Matrix([[0], [0], [0]])    # External force at p1
# M_p1 = sp.Matrix([[0], [0], [0]])    # External moment at p1
W_p1 = sp.Matrix([[0], [0], [0]])    # Angular velocity at p1
W_dot_p1 = sp.Matrix([[0], [0], [0]])  # Angular acceleration at p1

# Substituting into the equation of motion
equation_of_motion_p1 = equation_of_motion.subs({
    TCG: TCG_p1,
    W: W_p1,
    W_dot: W_dot_p1
})

# Simplify the resulting equation
equation_of_motion_p1 = sp.simplify(equation_of_motion_p1)

# Display the simplified equation of motion at p1
print("\nEquation of Motion at p1:")
sp.pprint(equation_of_motion_p1)


moment_p2 = sp.Matrix(positions[1]).cross(sp.Matrix(V_dot))


# Display the moment at p2
print("\nMoment at p2:")
sp.pprint(moment_p2)

import numpy as np

def calculate_inertia_tensor(points, masses, p1):
    """
    Calculate the 3x3 inertia tensor about a point p1.

    Parameters:
        points (np.ndarray): Array of shape (N, 3), where N is the number of mass points, and each row is (x, y, z).
        masses (np.ndarray): Array of masses corresponding to the points.
        p1 (np.ndarray): Coordinates of the reference point (x, y, z).

    Returns:
        np.ndarray: 3x3 inertia tensor about the point p1.
    """
    # Relative position vectors from p1
    relative_positions = points - p1

    # Initialize the inertia tensor
    inertia_tensor = np.zeros((3, 3))

    # Populate the inertia tensor
    for pos, mass in zip(relative_positions, masses):
        r_squared = np.dot(pos, pos)  # |r|^2
        for i in range(3):
            for j in range(3):
                if i == j:
                    inertia_tensor[i, j] += mass * (r_squared - pos[i] * pos[j])
                else:
                    inertia_tensor[i, j] -= mass * pos[i] * pos[j]

    return inertia_tensor


inertia_tensor = calculate_inertia_tensor(np.array(positions), masses, positions[0])
print("Inertia Tensor about p1:")
print(inertia_tensor)


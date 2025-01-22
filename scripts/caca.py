from casadi import *
# Define symbolic variables
m = SX.sym('m')  # Mass scalar
x_cg_c = SX.sym('x_cg_c', 3, 1)  # 3x1 vector for center of gravity
I = SX.sym('I', 3, 3)  # 3x3 matrix for inertia

# Define the skew-symmetric matrix for x_cg_c
def skew_symmetric(v):
    return vertcat(
        horzcat(0, -v[2], v[1]),
        horzcat(v[2], 0, -v[0]),
        horzcat(-v[1], v[0], 0)
    )

x_cg_c_cross = skew_symmetric(x_cg_c)

# Create the block matrix
block_matrix = vertcat(
    horzcat(m * SX.eye(3), -m * x_cg_c_cross),
    horzcat(m * x_cg_c_cross, I)
)

# Display the block matrix
print("Block Matrix:")
print(block_matrix)
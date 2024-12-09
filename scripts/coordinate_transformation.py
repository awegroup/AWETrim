import numpy as np

def transformation_matrix_w_to_azr(phi, beta):
    T_w_to_azr = np.array([
        [-np.sin(phi), np.cos(phi), 0],
        [-np.sin(beta)*np.cos(phi), -np.sin(beta)*np.sin(phi), np.cos(beta)],
        [np.cos(beta)*np.cos(phi), np.cos(beta)*np.sin(phi), np.sin(beta)]
    ])
    return T_w_to_azr

def transformation_matrix_azr_to_c(chi):
    T_azr_to_c = np.array([
        [np.sin(chi), np.cos(chi), 0],
        [-np.cos(chi), np.sin(chi), 0],
        [0, 0, 1]
    ])
    return T_azr_to_c

def transformation_matrix_w_to_c(phi, beta, chi):
    T_w_to_azr = transformation_matrix_w_to_azr(phi, beta)
    T_azr_to_c = transformation_matrix_azr_to_c(chi)
    T_w_to_c = np.dot(T_azr_to_c, T_w_to_azr)
    return T_w_to_c

def get_w_unit_vectors():
    e_x = np.array([1, 0, 0])
    e_y = np.array([0, 1, 0])
    e_z = np.array([0, 0, 1])
    return e_x, e_y, e_z

# Example angles (in radians)
phi = np.radians(0)  # Azimuth angle
beta = np.radians(0)  # Elevation angle
chi = np.radians(90)  # Course angle

# Compute the transformation matrix
T_w_to_c = transformation_matrix_w_to_c(phi, beta, chi)

# Get the W unit vectors
e_x, e_y, e_z = get_w_unit_vectors()

# Transform the W unit vectors to C unit vectors
e_tau = np.dot(T_w_to_c, e_x)
e_n = np.dot(T_w_to_c, e_y)
e_r = np.dot(T_w_to_c, e_z)

print("C unit vectors:")
print("e_tau:", e_tau)
print("e_n:", e_n)
print("e_r:", e_r)


vw = np.array([10, 0, 0])

# Example angles (in radians)
phi = np.radians(0)  # Azimuth angle
beta = np.radians(30)  # Elevation angle
chi = np.radians(90)  # Course angle

# Compute the transformation matrix
T_w_to_c = transformation_matrix_w_to_c(phi, beta, chi)

# Transform the wind velocity vector to the C frame
vc = np.dot(T_w_to_c, vw)
vc[0] += 30  # Add the kite speed in the C frame
vc[2] += 1.4
sideslip = np.arcsin(vc[1] / np.linalg.norm(vc))
print("Sideslip angle:", np.degrees(sideslip))
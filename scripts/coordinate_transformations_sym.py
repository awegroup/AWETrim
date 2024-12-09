import sympy as sp

# Define symbols for the angles
phi, beta, chi, chi_a, theta_a, chi_k, theta_k, phi_k = sp.symbols('phi beta chi chi_a theta_a chi_k theta_k phi_k')

def Rx(angle):
    return sp.Matrix([
        [1, 0, 0],
        [0, sp.cos(angle), -sp.sin(angle)],
        [0, sp.sin(angle), sp.cos(angle)]
    ])

def Ry(angle):
    return sp.Matrix([
        [sp.cos(angle), 0, sp.sin(angle)],
        [0, 1, 0],
        [-sp.sin(angle), 0, sp.cos(angle)]
    ])

def Rz(angle):
    return sp.Matrix([
        [sp.cos(angle), -sp.sin(angle), 0],
        [sp.sin(angle), sp.cos(angle), 0],
        [0, 0, 1]
    ])

# Transformation from AZR to W
transformation_AZR_W = sp.Matrix([
    [-sp.sin(phi), sp.cos(phi), 0],
    [-sp.sin(beta) * sp.cos(phi), -sp.sin(beta) * sp.sin(phi), sp.cos(beta)],
    [sp.cos(beta) * sp.cos(phi), sp.cos(beta) * sp.sin(phi), sp.sin(beta)]
])

# Transformation from C to AZR
transformation_C_AZR = sp.Matrix([
    [sp.sin(chi), sp.cos(chi), 0],
    [-sp.cos(chi), sp.sin(chi), 0],
    [0, 0, 1]
])

# Transformation from C to W
transformation_C_W = transformation_C_AZR * transformation_AZR_W



# Transformation from A to C
transformation_A_C = sp.Matrix([
    [sp.cos(theta_a) * sp.cos(chi_a), sp.cos(theta_a) * sp.sin(chi_a), -sp.sin(theta_a)],
    [-sp.sin(chi_a), sp.cos(chi_a), 0],
    [sp.sin(theta_a) * sp.cos(chi_a), sp.sin(theta_a) * sp.sin(chi_a), sp.cos(theta_a)]
])

# Transformation from K to C
transformation_K_C = sp.Matrix([
    [sp.cos(chi_k) * sp.cos(theta_k), sp.sin(chi_k) * sp.cos(theta_k), -sp.sin(theta_k)],
    [sp.cos(chi_k) * sp.sin(theta_k) * sp.sin(phi_k) - sp.sin(chi_k) * sp.cos(phi_k), sp.sin(chi_k) * sp.sin(theta_k) * sp.sin(phi_k) + sp.cos(chi_k) * sp.cos(phi_k), sp.cos(theta_k) * sp.sin(phi_k)],
    [sp.cos(chi_k) * sp.sin(theta_k) * sp.cos(phi_k) + sp.sin(chi_k) * sp.sin(phi_k), sp.sin(chi_k) * sp.sin(theta_k) * sp.cos(phi_k) - sp.cos(chi_k) * sp.sin(phi_k), sp.cos(theta_k) * sp.cos(phi_k)]
])

vw = transformation_C_W * sp.Matrix([1, 0, 0])
print(vw)


transformation_C_A = transformation_A_C.T

print(transformation_C_A)

weight = transformation_C_W * sp.Matrix([0, 0, -1])

print(weight)
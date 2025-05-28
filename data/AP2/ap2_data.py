import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Given stability derivatives
stab_derivs = {
    'CX': {
        '0': [-0.0293],
        'alpha': [0.4784, 2.5549],
        'q': [-0.6029, 4.4124],
        'deltae': [-0.0106, 0.1115]
    },
    'CY': {
        'beta': [-0.1855, -0.0299, 0.0936],
        'p': [-0.1022, -0.0140, 0.0496],
        'r': [0.1694, 0.1368],
        'deltaa': [-0.0514, -0.0024, 0.0579],
        'deltar': [0.10325, 0.0268, -0.1036]
    },
    'CZ': {
        '0': [-0.5526],
        'alpha': [-5.0676, 5.7736],
        'q': [-7.5560, 0.1251, 6.1486],
        'deltae': [-0.315, -0.0013, 0.2923]
    }
}

def compute_CX_CY_CZ(alpha, beta):
    """Compute CX, CY, CZ as functions of alpha and beta."""
    CX = (
        stab_derivs['CX']['0'][0] +
        stab_derivs['CX']['alpha'][0] * alpha +
        stab_derivs['CX']['alpha'][1] * alpha**2
    )

    CY = (
        stab_derivs['CY']['beta'][0] * beta +
        stab_derivs['CY']['beta'][1] * alpha * beta +
        stab_derivs['CY']['beta'][2] * alpha**2 * beta
    )

    CZ = (
        stab_derivs['CZ']['0'][0] +
        stab_derivs['CZ']['alpha'][0] * alpha +
        stab_derivs['CZ']['alpha'][1] * alpha**2
    )

    return CX, CY, CZ

def rotation_matrix(angle, axis):
     
    if axis == 'x':
        return np.array([[1, 0, 0],
                         [0, np.cos(angle), -np.sin(angle)],
                         [0, np.sin(angle), np.cos(angle)]])
    elif axis == "y":
         return np.array([[np.cos(angle), 0, np.sin(angle)],
                            [0, 1, 0],
                            [-np.sin(angle), 0, np.cos(angle)]])
    elif axis == "z":
        return np.array([[np.cos(angle), -np.sin(angle), 0],
                         [np.sin(angle), np.cos(angle), 0],
                         [0, 0, 1]])
         

def compute_CL_CD(alpha, beta):
    """Convert from body-frame (CX, CY, CZ) to wind-frame (CL, CD)."""
    CX, CY, CZ = compute_CX_CY_CZ(alpha, beta)

    Ry = rotation_matrix(alpha, "y")
    Rz = rotation_matrix(-beta, "z")
    Rpi = rotation_matrix(np.pi, "y")

    R = Rpi @ Ry @ Rz
    aero_coeffs = R @ np.array([CX, CY, CZ])

    return aero_coeffs[2], aero_coeffs[0]


# Generate values
alpha_vals = np.linspace(-10, 20, 50) * np.pi / 180  # Convert to radians
beta = 0
CL_vals = np.zeros((len(alpha_vals)))
CD_vals = np.zeros((len(alpha_vals)))
CX_vals = np.zeros((len(alpha_vals)))
CY_vals = np.zeros((len(alpha_vals)))
CZ_vals = np.zeros((len(alpha_vals)))

for i, alpha in enumerate(alpha_vals):
        CL_vals[i], CD_vals[i] = compute_CL_CD(alpha, beta)
        CX_vals[i], CY_vals[i], CZ_vals[i] = compute_CX_CY_CZ(alpha, beta)


# Define a second-degree polynomial fit function
def quadratic_poly(x, a, b, c):
    return a * x**2 + b * x + c

# Fit quadratic polynomials to CL and CD
popt_CL, _ = curve_fit(quadratic_poly, alpha_vals, CL_vals)
popt_CD, _ = curve_fit(quadratic_poly, alpha_vals, CD_vals)

# Generate fitted values
CL_fit = quadratic_poly(alpha_vals, *popt_CL)
CD_fit = quadratic_poly(alpha_vals, *popt_CD)

# Plot results
plt.figure(figsize=(8, 6))
plt.plot(alpha_vals * 180 / np.pi, CL_vals, 'bo', label="CL (original)")
plt.plot(alpha_vals * 180 / np.pi, CL_fit, 'b-', label=f"CL fit: {popt_CL[0]:.4f}α² + {popt_CL[1]:.4f}α + {popt_CL[2]:.4f}")

plt.plot(alpha_vals * 180 / np.pi, CD_vals, 'ro', label="CD (original)")
plt.plot(alpha_vals * 180 / np.pi, CD_fit, 'r-', label=f"CD fit: {popt_CD[0]:.4f}α² + {popt_CD[1]:.4f}α + {popt_CD[2]:.4f}")

plt.xlabel("Angle of Attack α (degrees)")
plt.ylabel("Coefficient Value")
plt.legend()
plt.title("Quadratic Fit for CL and CD")
plt.show()

# # Print the coefficients
# print("CL fit coefficients: ", popt_CL)
# print("CD fit coefficients: ", popt_CD)

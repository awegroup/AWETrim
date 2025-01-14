import numpy as np
from helper_functions import aero_coeffs_elliptical
import matplotlib.pyplot as plt

def calculate_tether_angle(d, c, b, va, alpha0_deg, alpha_deg, x_aero=0.25):
    """
    Calculate the angle of the tether force.

    Parameters:
        d (float): Distance between tether attachment point and kite.
        c (float): Chord length.
        b (float): Wingspan.
        va (float): Apparent wind speed.
        alpha0_deg (float): Angle of attack 0 in degrees.
        alpha_deg (float): Angle of attack in degrees.

    Returns:
        float: Tether force angle in degrees.
    """
    # Convert angles to radians
    alpha0 = np.radians(alpha0_deg)
    alpha = np.radians(alpha_deg)
    
    alpha = alpha0 + alpha

    # x_aero = 0.25  + 0.9* alpha

    # Calculate aspect ratio and geometric angle bridles
    A = c * b  # Wing area
    AR = b**2 / A  # Aspect ratio
    theta0 = np.arctan(c / d)
    

    # Calculate force distribution
    CL, CD, CM_ac = aero_coeffs_elliptical(alpha, AR)

    lift = 0.5 * CL * va**2 * A * 1.225  # Lift force
    drag = 0.5 * CD * va**2 * A * 1.225  # Drag force

    lift_vec = np.array([lift * np.sin(alpha), lift * np.cos(alpha), 0])
    drag_vec = np.array([-drag * np.cos(alpha), drag * np.sin(alpha), 0])

    F_aero = lift_vec + drag_vec
    r_LE = np.array([x_aero * c * np.cos(alpha0), x_aero * c * np.sin(alpha0), 0])
    r_TE = np.array([-(1-x_aero) * c * np.cos(alpha0), -(1-x_aero) * c * np.sin(alpha0), 0])
    r_LE_TE = np.array([-c * np.cos(alpha0), -c * np.sin(alpha0), 0])

    # # Aerodynamic moment at the aerodynamic center
    M_ac = CM_ac * 0.5 * 1.225 * va**2 * A * c

    # Define moment as a vector (assuming pitching moment about z-axis)
    M_ac_vec = np.array([0, 0, M_ac])

    F_TE = np.cross(r_LE_TE, M_ac_vec) / np.linalg.norm(r_LE_TE)**2
    F_LE = -F_TE

    M = np.cross(r_LE, F_aero)

    P1 = np.cross(r_LE_TE, M) / np.linalg.norm(r_LE_TE)**2
    P2 = -P1

    F_LE += F_aero + P2
    F_TE += P1

    br_LE = np.array([np.sin(theta0), np.cos(theta0), 0])
    br_TE = np.array([-np.sin(theta0), np.cos(theta0), 0])

    F_LE_br = np.dot(F_LE, br_LE) * br_LE
    F_TE_br = np.dot(F_TE, br_TE) * br_TE

    theta = np.arctan((F_TE_br[0] + F_LE_br[0]) / (F_TE_br[1] + F_LE_br[1])) * 180 / np.pi
    ratio = np.linalg.norm(F_LE_br) / (np.linalg.norm(F_TE_br)+np.linalg.norm(F_LE_br))
    tan_F = lift_vec[0] + drag_vec[0]

    return theta, ratio, tan_F

# Parameters
d = 10.0  # Distance between tether attachment point and kite
c = 2.0  # Chord length
b = 10.0  # Wingspan
va = 30.0  # Apparent wind speed
alpha_values = np.linspace(-20, 50, 100)  # Range of alpha values

# Alpha0 values to test
alpha0_values = [0]

# Plotting results for each alpha0
plt.figure(figsize=(10, 6))
for alpha0_deg in alpha0_values:
    theta_values = [calculate_tether_angle(d, c, b, va, alpha0_deg, alpha)[0] for alpha in alpha_values]
    plt.plot(alpha_values+alpha0_deg, theta_values, label=f'α₀ = {alpha0_deg}°')

# Plot customization
plt.xlabel('Angle of Attack (α) [degrees]')
plt.ylabel('Tether Force Angle (θ) [degrees]')
plt.title('Tether Force Angle vs. Angle of Attack for Different α₀')
plt.legend(title="Initial Angle of Attack (α₀)")
plt.xlim(0,20)
plt.grid(True)
plt.show()

# Plot ratio
import numpy as np
import matplotlib.pyplot as plt

def aero_coeffs_elliptical(alpha, AR):
    e = 0.7  # Oswald efficiency
    CD0 = 0.1  # Profile drag

    a0 = 4  # Thin airfoil theory
    a = a0 / (1 + a0 / (np.pi * AR))
    CL = 0.2 + a * alpha
    CD = CD0 + 0.1 * CL**2
    return CL, CD

def compute_positions(chord, ld, angle_deg, x_cp):
    angle_rad = np.radians(angle_deg)
    LE = np.array([0, ld, 0])
    TE = np.array([chord * np.cos(angle_rad), chord * np.sin(angle_rad) + ld, 0])
    B = np.array([0, 0, 0])
    CP = LE + (TE - LE) * x_cp
    return LE, TE, B, CP

def calculate_lambda_0(ld,x_cp, c, alpha_d):

    a = np.sqrt(ld**2 + (c*x_cp)**2 - 2*ld*c*x_cp*np.cos(np.pi/2+alpha_d))
    angle_rad= np.arcsin(c*x_cp*np.sin(np.pi/2+alpha_d)/a)
    # v1 = LE - B
    # v2 = CP - B
    # dot = np.dot(v1, v2)
    # norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
    # cos_theta = np.clip(dot / norm_product, -1.0, 1.0)
    # angle_rad = np.arccos(cos_theta)
    return np.degrees(angle_rad)

def compute_aero_forces(alpha_wind_rad, LE, TE, AR):
    # Airfoil orientation
    d_foil = TE[:2] - LE[:2]
    theta_foil = np.arctan2(d_foil[1], d_foil[0])

    # Angle of attack
    alpha =  alpha_wind_rad - theta_foil


    # print(f"alpha_wind_rad: {alpha_wind_rad}, theta_foil: {theta_foil}, alpha: {alpha}")
    # Aero coefficients
    CL, CD = aero_coeffs_elliptical(alpha, AR)

    # Airflow direction vector (unit)
    airflow_dir = np.array([np.cos(alpha_wind_rad), np.sin(alpha_wind_rad)])
    drag_dir = airflow_dir
    lift_dir = np.array([-drag_dir[1], drag_dir[0]])  # Perpendicular

    # Resultant force
    F = CL * lift_dir + CD * drag_dir
    return F, CL, CD, np.degrees(alpha)


def find_alpha_for_cp_alignment(LE, TE, B, CP, AR):
    cp_vec = CP[:2] - B[:2]
    cp_unit = cp_vec / np.linalg.norm(cp_vec)

    best_alpha_wind = None
    min_angle = np.inf
    best_force = None
    best_alpha_attack = None

    for alpha_wind_deg in np.linspace(-10, 30, 720):  # 0.5° steps
        alpha_wind_rad = np.radians(alpha_wind_deg)
        F, CL, CD, alpha_attack_deg = compute_aero_forces(alpha_wind_rad, LE, TE, AR)
        F_unit = F / np.linalg.norm(F)
        angle = np.degrees(np.arccos(np.clip(np.dot(F_unit, cp_unit), -1.0, 1.0)))

        if angle < min_angle:
            min_angle = angle
            best_alpha_wind = alpha_wind_deg
            best_force = F
            best_alpha_attack = alpha_attack_deg

    return best_alpha_wind, best_alpha_attack, best_force, min_angle

# --- Main setup ---
chord = 2.6
ld = 11.5
AR = 4
setups = [
    {'angle': 0, 'label': 'No depower', 'color': 'blue', 'x_cp': 0.32},
    {'angle': 0, 'label': '20° depower', 'color': 'red', 'x_cp': 0.4},
]

plt.figure(figsize=(4, 6))

for setup in setups:
    LE, TE, B, CP = compute_positions(chord, ld, setup['angle'], setup['x_cp'])
    airflow_dir = TE[:2] - LE[:2]

    # Plot kite
    plt.plot([LE[0], TE[0]], [LE[1], TE[1]], linestyle='--', color=setup['color'], label=f'{setup["label"]} chord')
    plt.plot([LE[0], B[0]], [LE[1], B[1]], linestyle='-', color=setup['color'])
    plt.plot([TE[0], B[0]], [TE[1], B[1]], linestyle='-', color=setup['color'])
    plt.plot(CP[0], CP[1], 'o', color=setup['color'], label=f'{setup["label"]} CP')
    plt.plot([B[0], CP[0]], [B[1], CP[1]], linestyle='--', color=setup['color'])

    alpha_wind, alpha_attack, F_align, min_angle = find_alpha_for_cp_alignment(LE, TE, B, CP, AR)
    print(f"{setup['label']} - alpha_wind: {alpha_wind:.1f}°, alpha_attack: {alpha_attack:.2f}°, alignment error: {min_angle:.2f}°")

    # Plot aerodynamic force vector
    F_scaled = F_align * 1.5  # Scaling for visualization
    CP_xy = CP[:2]
    plt.arrow(CP_xy[0], CP_xy[1], F_scaled[0], F_scaled[1],
              head_width=0.3, head_length=0.5, fc=setup['color'], ec=setup['color'])
    
    lambda_0 = calculate_lambda_0(ld, setup['x_cp'], chord, np.radians(setup['angle']))


    # print(f"l0: {B*180/np.pi:.2f}, a: {a:.2f}, B: {B:.2f}")

    print(f"{setup['label']} - lambda_0: {lambda_0:.2f}°")

plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.title('Kite Geometry and Aerodynamic Force Alignment')
plt.legend()
# plt.axis('equal')
plt.grid(True)
plt.show()

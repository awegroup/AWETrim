
import numpy as np
from picawe import State
import pandas as pd
import matplotlib.pyplot as plt
import time
import casadi as ca


aero_dict = {'oswald_efficiency': 0.9, 'aspect_ratio': 10, 'steering_coefficient': 0.2, 'CD0': 0.05, 'theta_t_0': np.radians(0), 'delta_theta_up': np.radians(-18.0)}
aero_input = ["inviscid", aero_dict]

csv_file = './processed_data/VSM_results_alpha_sweep.csv'
v3_polar_data = pd.read_csv(csv_file)


aero_input = {
    "model": "polars",
    "params": {
        "CD0": 0.075,
        'CL': v3_polar_data['CL'].values, 
        'CD': v3_polar_data['CD'].values, 
        'alpha': np.radians(v3_polar_data['aoa'].values), 
        'angle_pitch_depower_0': np.radians(-10),
        'delta_pitch_depower': np.radians(-15.0),
        "Cn_base": 0.05,
        # Add other aerodynamic parameters
    },
    "dependencies": {
        "alpha": {},
        "u_s": {"k_cl": 0, "k_cd": 0.15, "k_cs": 0.23, "k_cm": 0.005},
        "yaw_rate": {"k_cl": 0, "k_cd": 0, "k_cs": -0.01, "k_cm": -0.02},
        "sideslip": {"k_cl": 0, "k_cd": 0, "k_cs": 0.01, "k_cm": -0.05},
        "u_p": {"k_cl": 0, "k_cd": 0, "k_cs": 0, "k_cn": -0.05},
        # Add other dependencies as needed
    },
}


# Example Usage
kite = State(mass_wing=15, area_wing=20, aero_input=aero_input, mass_kcu = 25)


# Define the known state
known_state = {
    'speed_wind': 10,
    'distance_radial': 100.0,
    'angle_course': np.radians(90),
    'timeder_speed_tangential': 0.0,
    'speed_radial': 0.0,
    'timeder_speed_radial': 0.0,
    'input_depower': 0.0,
    'timeder_angle_course': 0.0,
}

unknown_vars = ['length_tether', 'input_steering', 'speed_tangential', 'angle_roll', 'angle_pitch', 'angle_yaw']
alpha_func = kite.extract_parameter_function('angle_of_attack')
T_func = kite.extract_parameter_function('tension_tether')

# Define the range of phi and beta
phi_values = np.radians(np.linspace(-90, 90, 50))  # Range for phi in radians
beta_values = np.radians(np.linspace(0, 90, 50))  # Range for beta in radians

# Generate combinations of phi and beta using meshgrid
phi_grid, beta_grid = np.meshgrid(phi_values, beta_values)

# Flatten the grids to create pairwise combinations
phi_combinations = phi_grid.flatten()
beta_combinations = beta_grid.flatten()

# Compute the absolute distance of phi from 0
phi_distances = np.abs(phi_combinations)

# Sort the indices based on the distance from 0
sorted_indices = np.argsort(phi_distances)

# Reorder the combinations
phi_combinations = phi_combinations[sorted_indices]
beta_combinations = beta_combinations[sorted_indices]


# Prepare to store solutions
solutions = []
# Initial guess
T_guess = 10000
u_s_guess = 0.1
v_tau_guess = 30
theta_guess = np.radians(1e-3)
phi_guess = np.radians(1e-3)
psi_guess = np.radians(1e-3)

start = time.time()
force_residual = kite.force_residual
residual = kite.rb_residual
for name, value in known_state.items():
    variable = getattr(kite, name)  # Dynamically retrieve variable from kite
    force_residual = ca.substitute(force_residual, variable, value)
    residual = ca.substitute(residual, variable, value)
# Prepare to store solutions
solutions = []
solver_options = {
    'ipopt': {
        'print_level': 0,  # Suppresses IPOPT output
        'max_iter': 200,  # Maximum number of iterations
        'sb': 'yes'        # Suppresses more detailed solver information
    },
    'print_time': False    # Disables CasADi's internal timing output
}
# Loop over combinations of phi and beta
for phi, beta in zip(phi_combinations, beta_combinations):
    
    # Substitute phi and beta into the residual equations
    subs_residual = ca.substitute(residual, kite.angle_azimuth, phi)
    subs_residual = ca.substitute(subs_residual, kite.angle_elevation, beta)
    known_state['angle_azimuth'] = phi
    known_state['angle_elevation'] = beta

    # Define variables to solve for
    sym_list = [getattr(kite, name) for name in unknown_vars]
    
    # Combine residual equations into a single vector
    g = subs_residual  # This is the vector of equations to solve
    
    # Bounds for the variables
    lbx = [95, -1, 0, -np.pi / 4, -np.pi / 4, -np.pi / 4]  # Lower bounds for T, u_s, v_tau, phi_k, theta_k
    ubx = [105, 1, 500, np.pi / 4, np.pi / 4, np.pi / 4]  # Upper bounds for T, u_s, v_tau, phi_k, theta_k
    
    # NLP problem definition
    nlp = {'x': ca.vertcat(*sym_list), 'f': 0, 'g': g}  # 'f' is set to 0 for root-finding

    # Define the NLP solver
    solver = ca.nlpsol('solver', 'ipopt', nlp, solver_options)

    # Bounds for the constraints
    # Bounds for the constraints
    lbg = [0] * g.size1()  # Lower bounds (0 for residuals)
    ubg = [0] * g.size1()  # Upper bounds (0 for residuals)

    # try:
        # Solve the system
    sol = solver(
        x0=[100, u_s_guess, v_tau_guess, phi_guess, theta_guess, psi_guess],  # Initial guess
        lbg=lbg,
        ubg=ubg,
        lbx=lbx,
        ubx=ubx
    )
    
    solution = sol['x']
    if ca.norm_1(sol['g']) < 1:
        state_combined = {**known_state}
        state_combined['length_tether'] = float(solution[0])
        state_combined['input_steering'] = float(solution[1])
        state_combined['speed_tangential'] = float(solution[2])
        state_combined['angle_roll'] = float(solution[3])
        state_combined['angle_pitch'] = float(solution[4])
        state_combined['angle_yaw'] = float(solution[5])

        # Calculate alpha value
        alpha_value = alpha_func(*[state_combined[name] for name in alpha_func.name_in()])
        T = T_func(*[state_combined[name] for name in T_func.name_in()])
        state_combined['alpha'] = float(alpha_value)
        state_combined['T'] = float(T)
        solutions.append(state_combined)

        # if beta < np.radians(50):
        # # Update guesses for better convergence
        #     u_s_guess = float(solution[1])
        #     v_tau_guess = float(solution[2])
        #     phi_guess = float(solution[3])
        #     theta_guess = float(solution[4])
        #     psi_guess = float(solution[5])


    # except RuntimeError as e:
    #     # Handle solver failure
    #     solutions.append({
    #         'phi': np.degrees(phi),
    #         'beta': np.degrees(beta),
    #         'T': None,
    #         'u_s': None,
    #         'v_tau': None
    #     })

end = time.time()
print(f"Time taken: {end - start} seconds for {len(phi_values) * len(beta_values)} iterations")

# At dt of 0.1, the time taken is:
time_per_iteration = (end - start) / (len(phi_values) * len(beta_values))
print(f"Time per iteration: {time_per_iteration} seconds")


# Display the solutions
solutions_df = pd.DataFrame(solutions)
# Filter out rows where 'T' is None
solutions_df = solutions_df[solutions_df['T'].notna()]
# solutions_df = solutions_df[(np.degrees(solutions_df['alpha']) < 20)&(np.degrees(solutions_df['alpha']) > -5)]
# solutions_df = solutions_df[solutions_df['T']>kite.m_wing*9.81]
solutions_df.reset_index(drop=True, inplace=True)


# Extract data for plotting
phi_values = solutions_df['angle_azimuth'].values
beta_values = solutions_df['angle_elevation'].values
alpha_values = np.degrees(solutions_df['alpha'].values)
tether_tensions = solutions_df['T'].values
theta_k_values = np.degrees(solutions_df['angle_pitch'].values)
phi_k_values = np.degrees(solutions_df['angle_roll'].values)
psi_k_values = np.degrees(solutions_df['angle_yaw'].values)
input_steering = solutions_df['input_steering'].values

# Convert spherical to Cartesian for 3D plotting
x = np.cos(beta_values) * np.sin(phi_values)* 1
y = np.cos(beta_values) * np.cos(phi_values) * 1
z = np.sin(beta_values) * 1

# Create a 3D plot for tether tension and angle of attack
fig = plt.figure(figsize=(16, 8))

# 3D Plot for Tether Tension
ax1 = fig.add_subplot(121, projection='3d')
sc1 = ax1.scatter(x, y, z, c=tether_tensions, cmap='viridis', marker='o')
fig.colorbar(sc1, ax=ax1, label="Tether Tension (N)")
ax1.set_title("Tether Tension (3D)")
ax1.set_xlabel("X Coordinate")
ax1.set_ylabel("Y Coordinate")
ax1.set_zlabel("Z Coordinate")

# 3D Plot for Angle of Attack (alpha)
ax2 = fig.add_subplot(122, projection='3d')
sc2 = ax2.scatter(x, y, z, c=alpha_values, cmap='plasma', marker='o')
fig.colorbar(sc2, ax=ax2, label="Angle of Attack (degrees)")
ax2.set_title("Angle of Attack (3D)")
ax2.set_xlabel("X Coordinate")
ax2.set_ylabel("Y Coordinate")
ax2.set_zlabel("Z Coordinate")

plt.tight_layout()


# Create a 3D plot for tether tension and angle of attack
fig = plt.figure(figsize=(18, 6))

# 3D Plot for Tether Tension
ax1 = fig.add_subplot(131, projection='3d')
mean = np.mean(theta_k_values)
sc1 = ax1.scatter(x, y, z, c=theta_k_values, cmap='viridis', marker='o',vmin=mean-5,vmax=mean+5)
fig.colorbar(sc1, ax=ax1, label="Pitch Angle (degrees)")
ax1.set_title("Pitch Angle (3D)")
ax1.set_xlabel("X Coordinate")
ax1.set_ylabel("Y Coordinate")
ax1.set_zlabel("Z Coordinate")

# 3D Plot for Angle of Attack (alpha)
ax2 = fig.add_subplot(132, projection='3d')
mean = np.mean(phi_k_values)
sc2 = ax2.scatter(x, y, z, c=phi_k_values, cmap='plasma', marker='o',vmin=mean-5,vmax=mean+5)
fig.colorbar(sc2, ax=ax2, label="Roll Angle (degrees)")
ax2.set_title("Roll Angle (3D)")
ax2.set_xlabel("X Coordinate")
ax2.set_ylabel("Y Coordinate")
ax2.set_zlabel("Z Coordinate")

ax3 = fig.add_subplot(133, projection='3d')
mean = np.mean(psi_k_values)
sc3 = ax3.scatter(x, y, z, c=psi_k_values, cmap='plasma', marker='o')
fig.colorbar(sc3, ax=ax3, label="Yaw Angle (degrees)")
ax3.set_title("Yaw Angle (3D)")
ax3.set_xlabel("X Coordinate")
ax3.set_ylabel("Y Coordinate")
ax3.set_zlabel("Z Coordinate")


plt.tight_layout()
plt.show()


# Create a 3D plot for tether tension and angle of attack
fig = plt.figure(figsize=(16, 8))

# 3D Plot for Tether Tension
ax1 = fig.add_subplot(121, projection='3d')
sc1 = ax1.scatter(x, y, z, c=tether_tensions, cmap='viridis', marker='o')
fig.colorbar(sc1, ax=ax1, label="Tether Tension (N)")
ax1.set_title("Tether Tension (3D)")
ax1.set_xlabel("X Coordinate")
ax1.set_ylabel("Y Coordinate")
ax1.set_zlabel("Z Coordinate")

# 3D Plot for Angle of Attack (alpha)
ax2 = fig.add_subplot(122, projection='3d')
sc2 = ax2.scatter(x, y, z, c=input_steering, cmap='plasma', marker='o')
fig.colorbar(sc2, ax=ax2, label="Steering Input")
ax2.set_title("Angle of Attack (3D)")
ax2.set_xlabel("X Coordinate")
ax2.set_ylabel("Y Coordinate")
ax2.set_zlabel("Z Coordinate")

plt.tight_layout()
plt.show()

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
state = State(mass_wing=15, area_wing=20, aero_input=aero_input, mass_kcu = 25)


# Define the known state
known_state = {
    'speed_wind': 10,
    'distance_radial': 100.0,
    'angle_course': np.radians(0),
    'timeder_speed_tangential': 0.0,
    'speed_radial': 0.0,
    'timeder_speed_radial': 0.0,
    'input_depower': 0.0,
    'input_steering': 0.0,
}

unknown_vars = ['length_tether', 'timeder_angle_course', 'speed_tangential', 'angle_roll', 'angle_pitch', 'angle_yaw']
alpha_func = state.extract_parameter_function('angle_of_attack')
T_func = state.extract_parameter_function('tension_tether')

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

start = time.time()

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
qs_guess = [known_state["distance_radial"], 0, 30, 1e-3, 1e-3, 1e-3]
# Loop over combinations of phi and beta
for phi, beta in zip(phi_combinations, beta_combinations):

    # Substitute phi and beta into the residual equations

    current_state = {
        'angle_azimuth': phi,
        'angle_elevation': beta
    }

    current_state, converged = state.solve_quasi_steady_state({**known_state,**current_state}, unknown_vars, qs_guess, solver_options= solver_options, dof = 6, return_not_converged=False)
    if converged:
        alpha_value = alpha_func(*[current_state[name] for name in alpha_func.name_in()])
        T = T_func(*[current_state[name] for name in T_func.name_in()])
        current_state['alpha'] = float(alpha_value)
        current_state['T'] = float(T)
        # qs_guess = [current_state[name] for name in unknown_vars]
        solutions.append(current_state)



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
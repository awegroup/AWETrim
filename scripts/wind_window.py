
import numpy as np
from picawe import KiteSystem, Environment, Control
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
        'theta_t_0': np.radians(-10),
        'delta_theta_up': np.radians(-15.0),
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
kite = KiteSystem(m_wing=15, A=20, aero_input=aero_input, m_kcu = 0)

# Define the known state
known_state = {
    'v_w': 10,
    'r': 100.0,
    'chi': np.radians(180),
    'dot_v_tau': 0.0,
    'v_r': -1.0,
    'dot_v_r': 0.0,
    'u_p': 0.0,
    'dot_chi': 0.0,
}

unknown_vars = ['T', 'u_s', 'v_tau']
unknown_vars_rb = ['T', 'u_s', 'v_tau', 'phi_k', 'theta_k', 'psi_k']
alpha_func = kite.extract_parameter_function('angle_of_attack')

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
        # 'max_iter': 1000,  # Maximum number of iterations
        'sb': 'yes'        # Suppresses more detailed solver information
    },
    'print_time': False    # Disables CasADi's internal timing output
}
# Loop over combinations of phi and beta
for phi, beta in zip(phi_combinations, beta_combinations):
    
    # Substitute phi and beta into the residual equations
    subs_force_residual = ca.substitute(force_residual, kite.phi, phi)
    subs_force_residual = ca.substitute(subs_force_residual, kite.beta, beta)
    subs_residual = ca.substitute(residual, kite.phi, phi)
    subs_residual = ca.substitute(subs_residual, kite.beta, beta)
    known_state['phi'] = phi
    known_state['beta'] = beta

    # Define variables to solve for
    sym_list = [getattr(kite, name) for name in unknown_vars_rb]
    
    # Combine residual equations into a single vector
    g = subs_residual  # This is the vector of equations to solve
    
    # Bounds for the variables
    lbx = [0, -1, 0, -np.pi / 4, -np.pi / 4, -np.pi / 4]  # Lower bounds for T, u_s, v_tau, phi_k, theta_k
    ubx = [1e6, 1, 500, np.pi / 4, np.pi / 4, np.pi / 4]  # Upper bounds for T, u_s, v_tau, phi_k, theta_k
    
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
        x0=[T_guess, u_s_guess, v_tau_guess, phi_guess, theta_guess, psi_guess],  # Initial guess
        lbg=lbg,
        ubg=ubg,
        lbx=lbx,
        ubx=ubx
    )
    
    solution = sol['x']
    if ca.norm_1(sol['g']) < 1:
        state_combined = {**known_state}
        state_combined['T'] = float(solution[0])
        state_combined['u_s'] = float(solution[1])
        state_combined['v_tau'] = float(solution[2])
        state_combined['phi_k'] = float(solution[3])
        state_combined['theta_k'] = float(solution[4])
        state_combined['psi_k'] = float(solution[5])

        # Calculate alpha value
        alpha_value = alpha_func(*[state_combined[name] for name in alpha_func.name_in()])
        state_combined['alpha'] = float(alpha_value)
        solutions.append(state_combined)


        # Update guesses for better convergence

        # T_guess = float(solution[0])
        # u_s_guess = float(solution[1])
        # v_tau_guess = float(solution[2])
        # phi_guess = float(solution[3])
        # theta_guess = float(solution[4])
        # psi_guess = float(solution[5])


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
phi_values = solutions_df['phi'].values
beta_values = solutions_df['beta'].values
alpha_values = np.degrees(solutions_df['alpha'].values)
tether_tensions = solutions_df['T'].values
theta_k_values = np.degrees(solutions_df['theta_k'].values)
phi_k_values = np.degrees(solutions_df['phi_k'].values)

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
fig = plt.figure(figsize=(16, 8))

# 3D Plot for Tether Tension
ax1 = fig.add_subplot(121, projection='3d')
sc1 = ax1.scatter(x, y, z, c=theta_k_values, cmap='viridis', marker='o')
fig.colorbar(sc1, ax=ax1, label="Pitch Angle (degrees)")
ax1.set_title("Pitch Angle (3D)")
ax1.set_xlabel("X Coordinate")
ax1.set_ylabel("Y Coordinate")
ax1.set_zlabel("Z Coordinate")

# 3D Plot for Angle of Attack (alpha)
ax2 = fig.add_subplot(122, projection='3d')
sc2 = ax2.scatter(x, y, z, c=phi_k_values, cmap='plasma', marker='o',vmin=-10,vmax=10)
fig.colorbar(sc2, ax=ax2, label="Roll Angle (degrees)")
ax2.set_title("Roll Angle (3D)")
ax2.set_xlabel("X Coordinate")
ax2.set_ylabel("Y Coordinate")
ax2.set_zlabel("Z Coordinate")

plt.tight_layout()
plt.show()
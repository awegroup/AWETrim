
import numpy as np
from picawe import KiteSystem, Environment, Control
import pandas as pd
import matplotlib.pyplot as plt
import time
import casadi as ca


aero_dict = {'oswald_efficiency': 0.9, 'aspect_ratio': 10, 'steering_coefficient': 0.2, 'CD0': 0.05}
aero_input = ["inviscid", aero_dict]

csv_file = './processed_data/VSM_results_alpha_sweep.csv'
v3_polar_data = pd.read_csv(csv_file)

aero_dict = {'CL': v3_polar_data['CL'].values, 
             'CD': v3_polar_data['CD'].values+0.075, 
             'alpha': np.radians(v3_polar_data['aoa'].values), 
             'steering_coefficient': 0.2,
             'k_cl_us': 0.0,
             'k_cd_us': 0.0,
             'k_cl_up': 0.0,
             'k_cd_up': 0.0,
             'theta_t_0': np.radians(-5),
             'delta_theta_up': np.radians(-10),
             }
aero_input = ["polars", aero_dict]


# Example Usage
kite = KiteSystem(m=30, A=20, aero_input=aero_input)

alpha_func = ca.Function('alpha', [kite.v_tau, kite.v_w, kite.beta, kite.chi, kite.phi, kite.v_r, kite.u_p], [kite.angle_of_attack], ['v_tau', 'v_w', 'beta', 'chi', 'phi', 'v_r', 'theta_t'], ['alpha'])
residual_func = kite.get_residual_function()

environment = Environment(v_w=10.0, g=9.81, rho=1.225)
residual_func = environment.apply(residual_func)

# Control inputs
v_r = 0.0  # Radial velocity
theta_t = np.radians(-2)  # Tether pitch angle
dot_v_r = 0.0  # Derivative of radial velocity
u_p = 0.0  # Powered angle
control = Control(dot_chi=0.0, v_r=v_r, u_p = u_p, dot_v_r=0.0)
residual_func = control.apply(residual_func)


# Extract the input names from the CasADi function
input_names = residual_func.name_in()
print("Input names:", input_names)
print(residual_func)


# Define the known inputs

# Positional inputs
r = 100.0  # Radius
chi = np.radians(0)  # Kite heading angle

# Quasi-steady assumption
dot_v_tau = 0.0  # Derivative of tether velocity

# Unknown inputs
T = ca.SX.sym('T')  # Tether tension
u_s = ca.SX.sym('u_s')  # Steering input
v_tau = ca.SX.sym('v_tau')  # Tangential velocity

# Define the range of phi and beta
phi_values = np.radians(np.linspace(-90, 90, 100))  # Range for phi in radians
beta_values = np.radians(np.linspace(0, 90, 100))  # Range for beta in radians
# Generate combinations of phi and beta using meshgrid
phi_grid, beta_grid = np.meshgrid(phi_values, beta_values)

# Flatten the grids to create pairwise combinations
phi_combinations = phi_grid.flatten()
beta_combinations = beta_grid.flatten()



# Prepare to store solutions
solutions = []
# Initial guess
T_guess = 1000
u_s_guess = 0.0
v_tau_guess = 200

start = time.time()
for phi, beta in zip(phi_combinations, beta_combinations):
    # Substitute current phi and beta into the residual function
    if phi > 0:
        chi_val = -chi
    else:
        chi_val = chi
    residual = residual_func(
        dot_v_tau=dot_v_tau,
        r=r,
        chi=chi_val, # Kite course angle
        beta=beta,  # Current beta
        phi=phi,  # Current phi
        u_s=u_s,
        T=T,
        v_tau=v_tau,
    )

    # Define partial_residual_func for this combination
    partial_residual_func = ca.Function(
        'partial_residual_func', [T, u_s, v_tau],
        [residual["residual"]], ['T', 'us', 'v_tau'], ['residual']
    )

    # Define the rootfinder
    rf = ca.rootfinder(
        'rf', 'newton',
        {'x': ca.vertcat(T, u_s, v_tau), 'g': partial_residual_func(T, u_s, v_tau)}
    )

    try:
        # Solve the system
        solution = rf([T_guess, u_s_guess, v_tau_guess], [])
        alpha = alpha_func(solution[2], environment.v_w, beta, chi_val, phi, v_r, theta_t)
        solutions.append({
            'phi': np.degrees(phi),
            'beta': np.degrees(beta),
            'T': float(solution[0]),
            'u_s': float(solution[1]),
            'v_tau': float(solution[2]),
            'alpha': float(alpha),
        })
        # T_guess = float(solution[0])
        # u_s_guess = float(solution[1])
        # v_tau_guess = float(solution[2])
    except RuntimeError as e:
        # Handle solver failure
        solutions.append({
            'phi': np.degrees(phi),
            'beta': np.degrees(beta),
            'T': None,
            'u_s': None,
            'v_tau': None
        })

end = time.time()
print(f"Time taken: {end - start} seconds for {len(phi_values) * len(beta_values)} iterations")

# At dt of 0.1, the time taken is:
time_per_iteration = (end - start) / (len(phi_values) * len(beta_values))
print(f"Time per iteration: {time_per_iteration} seconds")


# Display the solutions

solutions_df = pd.DataFrame(solutions)
# Filter out rows where 'T' is None
solutions_df = solutions_df[solutions_df['T'].notna()]
solutions_df = solutions_df[np.degrees(solutions_df['alpha']) < 20]
solutions_df = solutions_df[solutions_df['T']>kite.m*9.81]
solutions_df.reset_index(drop=True, inplace=True)


# Extract data for plotting
phi_values = solutions_df['phi'].values
beta_values = solutions_df['beta'].values
alpha_values = np.degrees(solutions_df['alpha'].values)
tether_tensions = solutions_df['T'].values

# Convert spherical to Cartesian for 3D plotting
x = np.cos(np.radians(beta_values)) * np.sin(np.radians(phi_values)) * 1
y = np.cos(np.radians(beta_values)) * np.cos(np.radians(phi_values)) * 1
z = np.sin(np.radians(beta_values)) * 1

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
plt.show()
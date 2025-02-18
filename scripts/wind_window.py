import numpy as np
from picawe import SystemModel
import pandas as pd
import matplotlib.pyplot as plt
import time
import json


aero_dict = {
    "oswald_efficiency": 0.9,
    "aspect_ratio": 10,
    "steering_coefficient": 0.2,
    "CD0": 0.05,
    "theta_t_0": np.radians(0),
    "delta_theta_up": np.radians(-18.0),
}
aero_input = ["inviscid", aero_dict]

# -----------------------------------------------
# Load data and define aerodynamic model
# -----------------------------------------------

# Define aerodynamic input
file_path = "./data/v3_aero_input.json"
with open(file_path, "r") as file:
    aero_input = json.load(file)


# Example Usage
kite_model = SystemModel(mass_wing=15, area_wing=20, aero_input=aero_input, mass_kcu=10, dof=6, quasi_steady=True, wind_model="uniform")

# Constants
kite_model.speed_wind_ref = 10
kite_model.distance_radial = 200.0
kite_model.angle_course = np.radians(90)
kite_model.speed_radial = 0.0
kite_model.input_depower = 0.0
kite_model.input_steering = 0.0

unknown_vars = [
    "tension_tether_ground",
    "timeder_angle_course",
    "speed_tangential",
    "angle_roll",
    "angle_pitch",
    "angle_yaw",
]
# alpha_func = kite_model.extract_parameter_function('angle_of_attack')
# T_func = kite_model.extract_parameter_function('tension_tether')

# Define the range of phi and beta
phi_values = np.radians(np.linspace(-90, 90, 30))  # Range for phi in radians
beta_values = np.radians(np.linspace(0, 90, 30))  # Range for beta in radians

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
    "ipopt": {
        "print_level": 0,  # Suppresses IPOPT output
        # "max_iter": 200,  # Maximum number of iterations
        "sb": "yes",  # Suppresses more detailed solver information
    },
    "print_time": False,  # Disables CasADi's internal timing output
}
qs_guess = [kite_model.distance_radial, 0, 40, 1e-3, 1e-3, 1e-3]

solve_func, inputs_name = kite_model.solve_quasi_steady_state(
        unknown_vars, solver_options=solver_options
    )
print(solve_func)
# Loop over combinations of phi and beta
for phi, beta in zip(phi_combinations, beta_combinations):

    current_state = {
        "distance_radial": kite_model.distance_radial,
        "angle_elevation": beta,
        "angle_azimuth": phi,
    }

    p = [current_state[name] for name in inputs_name]
    # print(p)
    lbx,ubx,lbg,ubg = kite_model.get_boundaries(unknown_vars)
    # print(lbx,ubx,lbg,ubg)
    sol = solve_func(x0=qs_guess, p=p, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)

    

    if np.linalg.norm(sol["g"]) < 1:
        # qs_guess = sol["x"]
        qs_state = {name: float(sol["x"][i]) for i, name in enumerate(unknown_vars)}
        current_state = qs_state
        current_state["angle_azimuth"] = phi
        current_state["angle_elevation"] = beta
        # current_state["alpha"] = float(alpha_value)
        # qs_guess = [current_state[name] for name in unknown_vars]
        solutions.append(current_state)
    else:
        print("Quasi steady solution not found")


end = time.time()
print(
    f"Time taken: {end - start} seconds for {len(phi_values) * len(beta_values)} iterations"
)

# At dt of 0.1, the time taken is:
time_per_iteration = (end - start) / (len(phi_values) * len(beta_values))
print(f"Time per iteration: {time_per_iteration} seconds")


# Display the solutions
solutions_df = pd.DataFrame(solutions)
# Filter out rows where 'T' is None
solutions_df = solutions_df[solutions_df["tension_tether_ground"].notna()]
# solutions_df = solutions_df[(np.degrees(solutions_df['alpha']) < 20)&(np.degrees(solutions_df['alpha']) > -5)]

solutions_df.reset_index(drop=True, inplace=True)


# Extract data for plotting
phi_values = solutions_df["angle_azimuth"].values
beta_values = solutions_df["angle_elevation"].values
# alpha_values = np.degrees(solutions_df['alpha'].values)
tether_tensions = solutions_df["tension_tether_ground"].values
theta_k_values = np.degrees(solutions_df["angle_pitch"].values)
phi_k_values = np.degrees(solutions_df["angle_roll"].values)
psi_k_values = np.degrees(solutions_df["angle_yaw"].values)
# input_steering = solutions_df['input_steering'].values

# Convert spherical to Cartesian for 3D plotting
x = np.cos(beta_values) * np.sin(phi_values) * 1
y = np.cos(beta_values) * np.cos(phi_values) * 1
z = np.sin(beta_values) * 1

# Create a 3D plot for tether tension and angle of attack
fig = plt.figure(figsize=(16, 8))

# 3D Plot for Tether Tension
ax1 = fig.add_subplot(121, projection="3d")
sc1 = ax1.scatter(x, y, z, c=tether_tensions, cmap="viridis", marker="o")
fig.colorbar(sc1, ax=ax1, label="Tether Tension (N)")
ax1.set_title("Tether Tension (3D)")
ax1.set_xlabel("X Coordinate")
ax1.set_ylabel("Y Coordinate")
ax1.set_zlabel("Z Coordinate")

# 3D Plot for Angle of Attack (alpha)
ax2 = fig.add_subplot(122, projection="3d")
# sc2 = ax2.scatter(x, y, z, c=alpha_values, cmap='plasma', marker='o')
# fig.colorbar(sc2, ax=ax2, label="Angle of Attack (degrees)")
ax2.set_title("Angle of Attack (3D)")
ax2.set_xlabel("X Coordinate")
ax2.set_ylabel("Y Coordinate")
ax2.set_zlabel("Z Coordinate")

plt.tight_layout()


# Create a 3D plot for tether tension and angle of attack
fig = plt.figure(figsize=(18, 6))

# 3D Plot for Tether Tension
ax1 = fig.add_subplot(131, projection="3d")
mean = np.mean(theta_k_values)
sc1 = ax1.scatter(
    x, y, z, c=theta_k_values, cmap="viridis", marker="o", vmin=mean - 5, vmax=mean + 5
)
fig.colorbar(sc1, ax=ax1, label="Pitch Angle (degrees)")
ax1.set_title("Pitch Angle (3D)")
ax1.set_xlabel("X Coordinate")
ax1.set_ylabel("Y Coordinate")
ax1.set_zlabel("Z Coordinate")

# 3D Plot for Angle of Attack (alpha)
ax2 = fig.add_subplot(132, projection="3d")
mean = np.mean(phi_k_values)
sc2 = ax2.scatter(
    x, y, z, c=phi_k_values, cmap="plasma", marker="o", vmin=mean - 5, vmax=mean + 5
)
fig.colorbar(sc2, ax=ax2, label="Roll Angle (degrees)")
ax2.set_title("Roll Angle (3D)")
ax2.set_xlabel("X Coordinate")
ax2.set_ylabel("Y Coordinate")
ax2.set_zlabel("Z Coordinate")

ax3 = fig.add_subplot(133, projection="3d")
mean = np.mean(psi_k_values)
sc3 = ax3.scatter(x, y, z, c=psi_k_values, cmap="plasma", marker="o")
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
ax1 = fig.add_subplot(121, projection="3d")
sc1 = ax1.scatter(x, y, z, c=tether_tensions, cmap="viridis", marker="o")
fig.colorbar(sc1, ax=ax1, label="Tether Tension (N)")
ax1.set_title("Tether Tension (3D)")
ax1.set_xlabel("X Coordinate")
ax1.set_ylabel("Y Coordinate")
ax1.set_zlabel("Z Coordinate")

# 3D Plot for Angle of Attack (alpha)
ax2 = fig.add_subplot(122, projection="3d")
# sc2 = ax2.scatter(x, y, z, c=input_steering, cmap='plasma', marker='o')
fig.colorbar(sc2, ax=ax2, label="Steering Input")
ax2.set_title("Angle of Attack (3D)")
ax2.set_xlabel("X Coordinate")
ax2.set_ylabel("Y Coordinate")
ax2.set_zlabel("Z Coordinate")

plt.tight_layout()
plt.show()

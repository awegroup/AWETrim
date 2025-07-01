import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import json

from picawe import SystemModel
from picawe.system.tether import FlexibleLumpedTether
from picawe.system.kite import Kite
from picawe.system.system_model import State


# ------------------------------
# Load aerodynamic input
# ------------------------------
file_path = "./data/LEI-V3-KITE/v3_aero_input.json"
with open(file_path, "r") as file:
    aero_input = json.load(file)

# ------------------------------
# Setup system model
# ------------------------------
tether = FlexibleLumpedTether()
kite = Kite(
    mass_wing=15,
    area_wing=20,
    aero_input=aero_input,
    mass_kcu=28,
    steering_control="asymmetric",
)
kite_model = SystemModel(dof=3, quasi_steady=True, kite=kite, tether=tether)

# Configure constants
kite_model.wind.speed_wind_ref = 10

# ------------------------------
# Define angle ranges
# ------------------------------
phi_values = np.radians(np.linspace(-80, 80, 30))
beta_values = np.radians(np.linspace(0, 60, 30))
phi_grid, beta_grid = np.meshgrid(phi_values, beta_values)

phi_combinations = phi_grid.flatten()
beta_combinations = beta_grid.flatten()

# Sort combinations by closeness to phi = 0
sorted_indices = np.argsort(np.abs(phi_combinations))
phi_combinations = phi_combinations[sorted_indices]
beta_combinations = beta_combinations[sorted_indices]

# ------------------------------
# Compute solutions
# ------------------------------
solutions = []
start = time.time()
for phi, beta in zip(phi_combinations, beta_combinations):
    state_obj = State(
        distance_radial=200,
        angle_elevation=beta,
        angle_azimuth=phi,
        angle_course=np.radians(90),
        speed_radial=0,
        input_depower=0,
        input_steering=0,
        timeder_angle_course=np.radians(0),
        speed_tangential=30,
        length_tether=199.99,
    )

    new_state = kite_model.solve_quasi_steady(
        state_obj,
        unknown_vars=["speed_tangential", "timeder_angle_course", "length_tether"],
    )
    if new_state is not None:
        new_state_dict = new_state.to_dict()
        solutions.append(new_state_dict)

end = time.time()
print(f"Time taken: {end - start:.2f}s for {len(phi_combinations)} iterations")
print(f"Time per iteration: {(end - start) / len(phi_combinations):.4f}s")


# ------------------------------
# Filter and visualize solutions
# ------------------------------
solutions_df = pd.DataFrame(solutions)
solutions_df = solutions_df[solutions_df["tension_tether_ground"].notna()]
solutions_df = solutions_df[
    (np.degrees(solutions_df["angle_of_attack"]) < 20)
    & (np.degrees(solutions_df["angle_of_attack"]) > -5)
]
solutions_df.reset_index(drop=True, inplace=True)

# Convert to Cartesian for 3D plotting
phi = solutions_df["angle_azimuth"].values
beta = solutions_df["angle_elevation"].values
alpha_values = np.degrees(solutions_df["angle_of_attack"].values)
tether_tensions = solutions_df["tension_tether_ground"].values
x = np.cos(beta) * np.sin(phi)
y = np.cos(beta) * np.cos(phi)
z = np.sin(beta)

# Plotting
fig = plt.figure(figsize=(16, 8))

# Tether Tension
ax1 = fig.add_subplot(121, projection="3d")
sc1 = ax1.scatter(x, y, z, c=tether_tensions, cmap="viridis")
fig.colorbar(sc1, ax=ax1, label="Tether Tension (N)")
ax1.set_title("Tether Tension (3D)")
ax1.set_xlabel("X")
ax1.set_ylabel("Y")
ax1.set_zlabel("Z")

# Angle of Attack
ax2 = fig.add_subplot(122, projection="3d")
sc2 = ax2.scatter(x, y, z, c=alpha_values, cmap="plasma")
fig.colorbar(sc2, ax=ax2, label="Angle of Attack (degrees)")
ax2.set_title("Angle of Attack (3D)")
ax2.set_xlabel("X")
ax2.set_ylabel("Y")
ax2.set_zlabel("Z")

plt.tight_layout()
plt.show()

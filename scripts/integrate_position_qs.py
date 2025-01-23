import numpy as np
import pandas as pd
import casadi as ca
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from picawe import State
import json

# -----------------------------------------------
# Load data and define aerodynamic model
# -----------------------------------------------

# Define aerodynamic input
file_path = "./data/v3_aero_input.json"
with open(file_path, "r") as file:
    aero_input = json.load(file)

# -----------------------------------------------
# Define the system and aerodynamic model
# -----------------------------------------------

state = State(
    mass_wing=15,
    area_wing=20,
    aero_input=aero_input,
    mass_kcu=25,
    dof=3,
)

# Set constant parameters
state.timeder_speed_tangential = 0.0
state.timeder_speed_radial = 0.0
state.speed_wind = 10
state.input_depower = 0.0
state.timeder_length_tether = 0
state.timeder_angle_course = 0.0

# Extract the tension tether function
tension_tether_func = state.extract_function("tension_tether")

# -----------------------------------------------
# Define simulation parameters and initial state
# -----------------------------------------------
unknown_vars = ["length_tether", "input_steering", "speed_tangential"]
current_state = {
    "distance_radial": 200,
    "angle_elevation": 0,
    "angle_azimuth": 0,
    "angle_course": 0,
    "speed_radial": 0,
    "speed_tangential": 10,
    "length_tether": 200,
}
solver_options = {
    "ipopt": {"print_level": 0, "sb": "yes"},
    "print_time": False,
}
time_step = 0.1
time = np.arange(0, 100, time_step)
qs_guess = [200, 0, 40]
states = []
import time as timet
start_time = timet.time()
state.establish_residual()
# -----------------------------------------------
# Time integration loop
# -----------------------------------------------
for t in time:
    # Solve quasi-steady state
    sol, _ = state.solve_quasi_steady_state(
        current_state, unknown_vars, qs_guess, solver_options=solver_options
    )
    qs_guess = sol

    # Construct initial conditions for integration
    x0 = [
        current_state["distance_radial"],
        current_state["angle_elevation"],
        current_state["angle_azimuth"],
        current_state["angle_course"],
        current_state["speed_radial"],
        float(sol[2]),  # speed_tangential
        float(sol[0]),  # length_tether
    ]

    # Integrate the dynamics
    xf = state.integrate(x0, t, time_step, quasi_steady=True)

    # Update the current state
    current_state = {name: float(xf[i]) for i, name in enumerate(current_state.keys())}

    # Evaluate tension tether
    T = tension_tether_func(
        *[current_state[name] for name in tension_tether_func.name_in()]
    )
    states.append({**current_state, "T": float(T)})

    # Stop if the system reaches critical limits
    if current_state["angle_elevation"] < 0 or current_state["distance_radial"] < 20:
        break

print("Elapsed time: ", timet.time() - start_time)
# -----------------------------------------------
# Process and visualize results
# -----------------------------------------------
solution_df = pd.DataFrame(states)

# Plot speed
plt.figure()
plt.plot(solution_df["speed_tangential"], label="Speed Tangential")
plt.xlabel("Time [s]")
plt.ylabel("Speed [m/s]")
plt.legend()

# Plot tether tension
plt.figure()
plt.plot(solution_df["T"], label="Tether Tension")
plt.xlabel("Time [s]")
plt.ylabel("Tether Tension [N]")
plt.legend()

# Extract spherical coordinates
r = solution_df["distance_radial"]
theta = solution_df["angle_azimuth"]
phi = solution_df["angle_elevation"]

# Convert to Cartesian coordinates
x = r * np.cos(phi) * np.cos(theta)
y = r * np.cos(phi) * np.sin(theta)
z = r * np.sin(phi)

# Plot 3D trajectory
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot(x, y, z, label="Trajectory")
ax.set_xlabel("X [m]")
ax.set_ylabel("Y [m]")
ax.set_zlabel("Z [m]")
ax.set_xlim(0, 200)
ax.set_ylim(-100, 100)
ax.set_zlim(0, 200)
ax.legend()

plt.show()

# -----------------------------------------------
# Print final results
# -----------------------------------------------
print("Reel-in elevation angle: ", np.degrees(states[-1]["angle_elevation"]))
print("Reel-in tether force: ", states[-1]["T"])

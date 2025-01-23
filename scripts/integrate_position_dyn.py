import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # For 3D trajectory plot
from picawe import State

# -----------------------------------------------
# Load data and define aerodynamic model
# -----------------------------------------------

# Define aerodynamic input
file_path = "./data/v3_aero_input.json"
with open(file_path, "r") as file:
    aero_input = json.load(file)

# -----------------------------------------------
# Define the system and initial state
# -----------------------------------------------
state = State(
    mass_wing=15,
    area_wing=20,
    aero_input=aero_input,
    mass_kcu=25,
    dof=6,
)

# Set constant parameters
state.speed_wind = 10
state.input_depower = 0.0
state.timeder_length_tether = 0
state.input_steering = 0.0

# Initial conditions
current_state = {
    "distance_radial": 200,
    "angle_elevation": np.radians(0),
    "angle_azimuth": 0,
    "angle_course": 0,
    "speed_radial": 0,
    "speed_tangential": 30,
    "length_tether": 200,
}
accelerations = {
    "timeder_speed_tangential": 0.0,
    "timeder_speed_radial": 0.0,
    "timeder_angle_roll": 0,
    "timeder_angle_pitch": 0,
    "timeder_angle_yaw": 0,
    "acceleration_angle_roll": 0,
    "acceleration_angle_pitch": 0,
    "acceleration_angle_yaw": 0,
}
unknown_vars = ["length_tether", "timeder_angle_course", "speed_tangential", "angle_roll", "angle_pitch", "angle_yaw"]
qs_guess = [200, 0, 40, 0, 0, 0]

# Solver configuration
solver_options = {
    "ipopt": {"print_level": 0, "sb": "yes"},
    "print_time": False,
}
time_step = 0.01
time = np.arange(0, 50, time_step)

# -----------------------------------------------
# Solve the quasi-steady state and initialize variables
# -----------------------------------------------
sol, _ = state.solve_quasi_steady_state(
    {**current_state, **accelerations}, unknown_vars, qs_guess, solver_options
)
current_state["speed_tangential"] = float(sol[2])
current_state["length_tether"] = float(sol[0])
current_state = {**current_state, 
                 'angle_roll': float(sol[3]),
                    'angle_pitch': float(sol[4]),
                    'angle_yaw': float(sol[5]),
                    'timeder_angle_roll': 0,
                    'timeder_angle_pitch': 0,
                    'timeder_angle_yaw': 0,
}
x0 = [x for x in current_state.values()]
states = []

# Extract functions
tension_tether_func = state.extract_function("tension_tether")
aoa_func = state.extract_function("angle_of_attack")

# -----------------------------------------------
# Time integration loop
# -----------------------------------------------
for t in time:
    try:
        # Integrate system dynamics
        xf, zf = state.integrate(x0, t, time_step)
    except Exception as e:
        print("Integration failed at time: ", t)
        print(e)
        break
    # Enforce constraints/reset values (e.g., angles)
    x0 = xf
    # x0[3] = 0  # Reset angle_course (Only to find the reel-in angle)
    # x0[2] = 0  # Reset angle_azimuth (Only to find the reel-in angle)

    # Update the current state
    new_state = {name: float(xf[j]) for j, name in enumerate(current_state.keys())}

    # Evaluate tension and angle of attack
    T = tension_tether_func(
        *[new_state[name] for name in tension_tether_func.name_in()]
    )
    aoa = aoa_func(*[new_state[name] for name in aoa_func.name_in()])

    # Store full state
    full_state = {**new_state, "T": float(T), "aoa": float(aoa)}
    states.append(full_state)

    # Stop if the system reaches critical limits
    if new_state["angle_elevation"] < 0 or new_state["distance_radial"] < 10:
        break

# -----------------------------------------------
# Process and visualize results
# -----------------------------------------------
solution_df = pd.DataFrame(states)

plt.figure()
plt.plot(solution_df["angle_pitch"]*180/np.pi, label="Pitch")
plt.plot(solution_df["angle_roll"]*180/np.pi, label="Roll")
plt.show()

# Plot speeds
plt.figure()
plt.plot(solution_df["speed_tangential"], label="Speed Tangential")
plt.plot(solution_df["speed_radial"], label="Speed Radial")
plt.xlabel("Time [s]")
plt.ylabel("Speed [m/s]")
plt.legend()

# Plot tether tension
plt.figure()
plt.plot(solution_df["T"], label="Tether Tension")
plt.xlabel("Time [s]")
plt.ylabel("Tether Tension [N]")
plt.legend()

# Plot angle of attack
plt.figure()
plt.plot(np.degrees(solution_df["aoa"]), label="Angle of Attack")
plt.xlabel("Time [s]")
plt.ylabel("Angle of Attack [deg]")
plt.legend()

# Extract spherical coordinates
r = solution_df["distance_radial"]
theta = solution_df["angle_azimuth"]
phi = solution_df["angle_elevation"]

# Convert to Cartesian coordinates for 3D trajectory
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



# Print final results
print("Reel-in elevation angle: ", np.degrees(states[-1]["angle_elevation"]))
print("Reel-in tether force: ", states[-1]["T"])

plt.show()
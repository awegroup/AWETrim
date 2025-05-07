import numpy as np
import pandas as pd
import casadi as ca
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from picawe import SystemModel
import json
from picawe.system.kite import Kite

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

kite = Kite(mass_wing=15, area_wing=20, aero_input=aero_input, mass_kcu=25, steering_control="asymmetric")
kite_model = SystemModel(
    dof=3,
    quasi_steady=True,
    wind_model="uniform",
    kite=kite,
)

# Set constant parameters
kite_model.wind.speed_wind_ref = 10
# kite_model.input_depower = 1.0
# kite_model.timeder_angle_course = 0.0

# Extract the tension tether function
aoa_func = kite_model.extract_function("angle_of_attack")

# -----------------------------------------------
# Define simulation parameters and initial state
# -----------------------------------------------
unknown_vars = ["speed_tangential", "timeder_angle_course", "length_tether"]
current_state = {
    "distance_radial": 200,
    "angle_elevation": np.radians(5),
    "angle_azimuth": 0,
    "angle_course": 0,
    "speed_radial": -1,
    "speed_tangential": 10,
    "input_depower": 0.0,
    "input_steering": 0
}
solver_options = {
    "ipopt": {"print_level": 0, "sb": "yes"},
    "print_time": False,
}
time_step = 0.01
time = np.arange(0, 50, time_step)
qs_guess = [200, 0, 40]
states = []
import time as timet
start_time = timet.time()
solve_qs, inputs_name = kite_model.setup_qs_solver(
        unknown_vars, solver_options=solver_options
    )
# Solve quasi-steady state
p = [current_state[name] for name in inputs_name]

lbx,ubx,lbg,ubg = kite_model.get_boundaries(unknown_vars)
sol = solve_qs(x0=qs_guess, p=p, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
z0 = sol['x']
kite_model.establish_ode()
kite_model.establish_algebraic()
# Construct initial conditions for integration
x0 = [
    current_state["distance_radial"],
    current_state["angle_elevation"],
    current_state["angle_azimuth"],
    current_state["angle_course"],
]
    
# -----------------------------------------------
# Time integration loop
# -----------------------------------------------
intg = kite_model.integrator(time_step)
inputs = {"input_steering": 0.0, "input_depower": 0.0, "speed_radial": 0.0}
p = [inputs[name] for name in inputs.keys()]
for t in time:

    # Integrate the dynamics
    sol = intg(x0=x0, z0=z0, p=p)
    xf = sol["xf"]
    zf = sol["zf"]
    x0 = xf
    z0 = zf

    # Update the current state
    # current_state = {name: float(xf[i]) for i, name in enumerate(current_state.keys())}

    full_state = { "distance_radial": float(xf[0]),
                    'angle_elevation': float(xf[1]),
                    'angle_azimuth': float(xf[2]),
                    'angle_course': float(xf[3]),
                    'speed_tangential': float(zf[0]),
                    'timeder_angle_course': float(zf[1]),
                    'length_tether': float(zf[2]),

    }

    # Evaluate tension tether

    # aoa = aoa_func(
    #     *[full_state[name] for name in aoa_func.name_in()]
    # )

    states.append({**full_state})#, "aoa": float(aoa)})

    # Stop if the system reaches critical limits
    if current_state["angle_elevation"] < 0 or current_state["distance_radial"] < 20:
        break

print("Elapsed time: ", timet.time() - start_time)
print("Simulated time: ", time[-1])
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


# # Plot tether tension
# plt.figure()
# plt.plot(solution_df["tension_tether_ground"], label="Tether Tension")
# plt.xlabel("Time [s]")
# plt.ylabel("Tether Tension [N]")
# plt.legend()

#Plot angle of attack
# plt.figure()
# plt.plot(solution_df["aoa"]*180/np.pi, label="Angle of Attack")
# plt.xlabel("Time [s]")
# plt.ylabel("Angle of Attack [deg]")
# plt.legend()

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
# print("Reel-in tether force: ", states[-1]["tension_tether_ground"])

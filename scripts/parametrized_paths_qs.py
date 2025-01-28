import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from picawe.parametrized_patterns import Helix, Lissajous, FigureEight
from picawe.Kinematics import ParametrizedKinematics
from picawe import State
import casadi as ca
import time as timet
import json

# -----------------------------------------------
# Load data and define aerodynamic model
# -----------------------------------------------

# Define aerodynamic input
file_path = "./data/v3_aero_input.json"
with open(file_path, "r") as file:
    aero_input = json.load(file)

# -----------------------------------------------
# Define the parametrized path
# -----------------------------------------------
omega = -1
x0 = 200
rh = 61.7
vr = 0
beta = np.radians(0)
ry = 120
rz = 40
helix = Helix(omega, x0, rh, vr, beta)
lissajous = Lissajous(omega, x0, ry, rz, vr, beta)
figure_eight = FigureEight(omega, x0, 80, 80, vr, beta)

pattern = helix
kinematics = ParametrizedKinematics(pattern)
state = State(mass_wing=15, area_wing=20, aero_input=aero_input, mass_kcu=25, dof=3, quasi_steady=True)

# Substitute the numeric values into the symbolic expressions using CasADi functions
chi_func = ca.Function(
    "chi", [kinematics.t, kinematics.s, kinematics.s_dot], [kinematics.chi]
)

vk_func = ca.Function(
    "vk", [kinematics.t, kinematics.s, kinematics.s_dot], [kinematics.vk]
)
dR_ds_func = ca.Function(
    "dR_ds",
    [kinematics.t, kinematics.s, kinematics.s_dot],
    [ca.norm_2(kinematics.dR_ds)],
)

vr_func = ca.Function("vr", [kinematics.t], [kinematics.vr])
dot_chi_func = ca.Function(
    "dot_chi", [kinematics.t, kinematics.s, kinematics.s_dot], [kinematics.dot_chi]
)
vtau_func = ca.Function(
    "vtau", [kinematics.t, kinematics.s, kinematics.s_dot], [kinematics.vtau]
)


state.speed_wind = 10
state.input_depower = 0

solver_options = {
    "ipopt": {
        "print_level": 0,  # Suppresses IPOPT output
        # 'max_iter': 200,  # Maximum number of iterations
        "sb": "yes",  # Suppresses more detailed solver information
        
    },
    "print_time": False,  # Disables CasADi's internal timing output
    # "allow_free": True,  # Allows free variables
}
time_step = 0.1
s = np.linspace(0, 2*np.pi, 100) + np.pi/2
s_dot = 0.1
s_ddot = 0
vk = 20
states = []
unknown_vars = ["length_tether", "input_steering", "s_dot"]
qs_guess = [200, 0, 10]
s_dot = ca.SX.sym("s_dot")
s_sym = ca.SX.sym("s")
time_sym = ca.SX.sym("time")
state.s_dot = s_dot
start_time = timet.time()
state.timeder_angle_course =  dot_chi_func(time_sym, s_sym, s_dot)
state.speed_tangential = vtau_func(time_sym, s_sym, s_dot)
state.angle_course = chi_func(time_sym, s_sym, s_dot)
state.override_gravity = False
state.override_centripetal = False
state.override_coriolis = False

tension_tether_func = state.extract_function("tension_tether")
solve_func, inputs_name = state.solve_quasi_steady_state(
        unknown_vars, solver_options=solver_options
    )
print(solve_func)
print(ca.symvar(state.residual))
time = 0
power = 0
for i in range(len(s)):

    current_state = {
        "distance_radial": pattern.r(time),
        "angle_elevation": pattern.elevation(time, s[i]),
        "angle_azimuth": pattern.azimuth(time, s[i]),
        "speed_radial": float(vr_func(time)),
        "length_tether": pattern.r(time),
        "s": s[i],
        "time": time,
    }
    p = [current_state[name] for name in inputs_name]

    lbx,ubx,lbg,ubg = state.get_boundaries(current_state)
    sol = solve_func(x0=qs_guess, p=p, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)

    qs_guess = sol["x"]
    qs_state = {name: float(qs_guess[i]) for i, name in enumerate(unknown_vars)}

    current_state["angle_course"] = float(chi_func(time, s[i], sol['x'][2]))
    current_state["speed_tangential"] = float(vtau_func(time, s[i], sol['x'][2]))
    full_state = {**current_state, **qs_state}
    full_state["tension_tether"] = float(
        tension_tether_func(
            *[full_state[name] for name in tension_tether_func.name_in()]
        )
    )
    
    states.append(full_state)
    if i < len(s)-1:
        time_step = (s[i+1]-s[i])/float(sol['x'][2])
        power += full_state["tension_tether"]*time_step*full_state["speed_radial"]
        time += time_step
print(f"Time taken: {timet.time() - start_time}")
states = pd.DataFrame(states)

print("Power: ", power/time, "W")
# states = states[abs(states["s"] * omega) > 2 * np.pi]

# Reflect the override choices in the file name
override_settings = {
    "gravity": state.override_gravity,
    "centripetal": state.override_centripetal,
    "coriolis": state.override_coriolis,
}

# Build a suffix based on active overrides
overrides_active = [key for key, value in override_settings.items() if value]
overrides_suffix = "_".join(overrides_active)
file_name = f"helix_quasi_steady"
if overrides_active:
    file_name += f"_override_{overrides_suffix}"
file_name += ".csv"

# Save the DataFrame
output_path = f"./results/impact_inertial_forces/{file_name}"
states.to_csv(output_path, index=False)

print(f"Saved results to {output_path}")

# Convert angles to radians for plotting if necessary
azimuth = np.array(states["angle_azimuth"])
elevation = np.array(states["angle_elevation"])
speed_tangential = np.array(states["speed_tangential"])
input_steering = np.array(states["input_steering"])
course = np.array(states["angle_course"])
s = np.array(states["s"])

# Find indices of max and min tangential speed
max_idx = np.argmax(speed_tangential)
min_idx = np.argmin(speed_tangential)

# Plot the trajectory with a colorbar for tangential speed
plt.figure(figsize=(8, 6))
scatter = plt.scatter(
    azimuth, elevation, c=speed_tangential, cmap="viridis", s=10
)  # `s` adjusts marker size
cbar = plt.colorbar(scatter)
cbar.set_label("Tangential Speed [m/s]", fontsize=12)

# Plot max and min speed points
plt.scatter(
    azimuth[max_idx],
    elevation[max_idx],
    color="red",
    label="Max Speed",
    edgecolor="black",
    zorder=5,
)
plt.scatter(
    azimuth[min_idx],
    elevation[min_idx],
    color="red",
    label="Min Speed",
    edgecolor="black",
    zorder=5,
)

print(
    f"Max speed: {speed_tangential[max_idx]} m/s at phase {(s[max_idx]*omega*180/np.pi)%360} degrees"
)


# Labels, title, and legend
plt.xlabel("Azimuth [rad]", fontsize=12)
plt.ylabel("Elevation [rad]", fontsize=12)
plt.title("Flown Trajectory with Tangential Speed", fontsize=14)
plt.legend(fontsize=10)
plt.grid()
# plt.show()

# Plot the trajectory with a colorbar for tangential speed
plt.figure(figsize=(8, 6))
scatter = plt.scatter(
    azimuth, elevation, c=course, cmap="viridis", s=10
)  # `s` adjusts marker size
cbar = plt.colorbar(scatter)
cbar.set_label("Course angle", fontsize=12)

# Labels, title, and legend
plt.xlabel("Azimuth [rad]", fontsize=12)
plt.ylabel("Elevation [rad]", fontsize=12)
plt.grid()
plt.show()

# # Plot the trajectory with a colorbar for tangential speed
# plt.figure(figsize=(8, 6))
# scatter = plt.scatter(
#     azimuth, elevation, c=np.array(states["angle_roll"]), cmap="viridis", s=10
# )  # `s` adjusts marker size
# cbar = plt.colorbar(scatter)
# cbar.set_label("Roll angle", fontsize=12)

# # Labels, title, and legend
# plt.xlabel("Azimuth [rad]", fontsize=12)
# plt.ylabel("Elevation [rad]", fontsize=12)
# plt.legend(fontsize=10)
# plt.grid()
# plt.show()


# def angle_with_x(azimuth, elevation):
#     # Convert azimuth and elevation to radians if not already
#     phi = np.radians(azimuth)
#     theta = np.radians(elevation)

#     # Cartesian coordinates of the vector
#     vx = np.cos(theta) * np.cos(phi)
#     vy = np.cos(theta) * np.sin(phi)
#     vz = np.sin(theta)

#     # Magnitude of the vector
#     magnitude = np.sqrt(vx**2 + vy**2 + vz**2)

#     # Angle with the x-axis
#     angle = np.arccos(vx / magnitude)  # Result in radians
#     return np.degrees(angle)  # Convert to degrees if needed


# vatau = (
#     -np.sin(elevation) * np.cos(azimuth) * np.cos(course)
#     - np.sin(azimuth) * np.sin(course)
# ) * 10 - speed_tangential
# van = (
#     -np.sin(elevation) * np.cos(azimuth) * np.sin(course)
#     + np.sin(azimuth) * np.cos(course)
# ) * 10
# var = (np.cos(elevation) * np.cos(azimuth)) * 10
# # Plot the trajectory with a colorbar for tangential speed
# plt.figure(figsize=(8, 6))
# scatter = plt.scatter(
#     azimuth, elevation, c=van / vatau, cmap="viridis", s=10
# )  # `s` adjusts marker size
# cbar = plt.colorbar(scatter)
# cbar.set_label("Tangential Speed [m/s]", fontsize=12)

# # Labels, title, and legend
# plt.xlabel("Azimuth [rad]", fontsize=12)
# plt.ylabel("Elevation [rad]", fontsize=12)
# plt.title("Flown Trajectory with Tangential Speed", fontsize=14)
# plt.legend(fontsize=10)
# plt.grid()

# plt.figure()
# plt.plot(time[: i + 1], input_steering)
# plt.show()

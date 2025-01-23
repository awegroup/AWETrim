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
omega = -0.1
x0 = 200
rh = 60
vr = 0
beta = np.radians(30)
ry = 120
rz = 40
helix = Helix(omega, x0, rh, vr, beta)
lissajous = Lissajous(omega, x0, ry, rz, vr, beta)
figure_eight = FigureEight(omega, x0, 80, 80, vr, beta)

pattern = helix
kinematics = ParametrizedKinematics(pattern)
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

dot_vtau_func = ca.Function(
    "dot_vtau",
    [kinematics.t, kinematics.s, kinematics.s_dot, kinematics.s_ddot],
    [kinematics.dot_vtau],
)
dot_vr_func = ca.Function(
    "dot_vr",
    [kinematics.t, kinematics.s, kinematics.s_dot, kinematics.s_ddot],
    [kinematics.dot_vr],
)

# -----------------------------------------------
# Define the system
# -----------------------------------------------
state = State(mass_wing=15, area_wing=20, aero_input=aero_input, mass_kcu=25, dof=3, quasi_steady=True)

state.speed_wind = 10
state.input_depower = 0


solver_options = {
    "ipopt": {
        "print_level": 0,  # Suppresses IPOPT output
        # 'max_iter': 200,  # Maximum number of iterations
        "sb": "yes",  # Suppresses more detailed solver information
    },
    "print_time": False,  # Disables CasADi's internal timing output
}
time_step = 0.1
time = np.arange(0, 300, time_step)
s = 0
s_dot = 0.1
vk = 20
states = []
unknown_vars = ["length_tether", "input_steering", "s_dot"]
qs_guess = [200, 0, 40]

start_time = timet.time()

current_state = {
    "time": time[0],
    "distance_radial": pattern.r(time[0]),
    "s": s,
    "timeder_speed_tangential": 0,
    "timeder_speed_radial": 0,
}

s_dot_sym = ca.SX.sym("s_dot")
state.s_dot = s_dot_sym
s_sym = ca.SX.sym("s")
time_sym = ca.SX.sym("time")
state.speed_tangential = vtau_func(time_sym, s_sym, s_dot_sym)
state.angle_course = chi_func(time_sym, s_sym, s_dot_sym)
state.timeder_angle_course = dot_chi_func(time_sym, s_sym, s_dot_sym)
state.distance_radial = pattern.r(time_sym)
state.angle_elevation = pattern.elevation(time_sym, s_sym)
state.angle_azimuth = pattern.azimuth(time_sym, s_sym)
state.speed_radial = vr_func(time_sym)


solve_func, inputs_name = state.solve_quasi_steady_state(
        unknown_vars, solver_options=solver_options
    )

# Solve quasi-steady state
p = [current_state[name] for name in inputs_name]

lbx,ubx,lbg,ubg = state.get_boundaries(current_state)

sol = solve_func(x0=qs_guess, p=p, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)

current_state["s_dot"] = float(sol['x'][2])
s_ddot_sym = ca.SX.sym("s_ddot")
state.s_ddot = s_ddot_sym
state.timeder_speed_tangential = dot_vtau_func(time_sym, s_sym, s_dot_sym, s_ddot_sym)
state.timeder_speed_radial = dot_vr_func(time_sym, s_sym, s_dot_sym, s_ddot_sym)
unknown_vars = ["length_tether", "input_steering", "s_ddot"]
solve_func, inputs_name = state.solve_quasi_steady_state(
        unknown_vars, solver_options=solver_options
    )

for i in range(len(time)):
    # print(i)
    # Solve quasi-steady state
    p = [current_state[name] for name in inputs_name]

    lbx,ubx,lbg,ubg = state.get_boundaries(current_state)
    sol = solve_func(x0=qs_guess, p=p, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
    qs_guess = sol['x']
    # print(qs_guess)

    s_dot += float(sol['x'][2]) * time_step
    s += s_dot * time_step
    if np.linalg.norm(s_dot) > 1:
        print("Infeasible solution")
    current_state = {}
    current_state["s"] = s
    current_state["time"] = time[i]
    current_state["s_dot"] = s_dot
    current_state["input_steering"] = float(sol['x'][1])
    current_state["length_tether"] = float(sol['x'][0])
    current_state["angle_course"] = float(chi_func(time[i], s, s_dot))
    current_state["speed_tangential"] = float(vtau_func(time[i], s, s_dot))
    current_state["angle_azimuth"] = float(pattern.azimuth(time[i], s))
    current_state["angle_elevation"] = float(pattern.elevation(time[i], s))
    current_state["speed_radial"] = float(vr_func(time[i]))
    current_state["timeder_angle_course"] = float(dot_chi_func(time[i], s, s_dot))
    current_state["distance_radial"] = float(pattern.r(time[i]))
    states.append(current_state)

    if abs(omega * s) > 6 * np.pi:
        break
print(f"Time taken: {timet.time() - start_time}")
states = pd.DataFrame(states)


states = states[abs(states["s"] * omega) > 4 * np.pi]

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
plt.show()

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
plt.legend(fontsize=10)
plt.grid()
plt.show()

# Plot the trajectory with a colorbar for tangential speed
plt.figure(figsize=(8, 6))
scatter = plt.scatter(
    azimuth, elevation, c=np.array(states["angle_roll"]), cmap="viridis", s=10
)  # `s` adjusts marker size
cbar = plt.colorbar(scatter)
cbar.set_label("Roll angle", fontsize=12)

# Labels, title, and legend
plt.xlabel("Azimuth [rad]", fontsize=12)
plt.ylabel("Elevation [rad]", fontsize=12)
plt.legend(fontsize=10)
plt.grid()
plt.show()


def angle_with_x(azimuth, elevation):
    # Convert azimuth and elevation to radians if not already
    phi = np.radians(azimuth)
    theta = np.radians(elevation)

    # Cartesian coordinates of the vector
    vx = np.cos(theta) * np.cos(phi)
    vy = np.cos(theta) * np.sin(phi)
    vz = np.sin(theta)

    # Magnitude of the vector
    magnitude = np.sqrt(vx**2 + vy**2 + vz**2)

    # Angle with the x-axis
    angle = np.arccos(vx / magnitude)  # Result in radians
    return np.degrees(angle)  # Convert to degrees if needed


vatau = (
    -np.sin(elevation) * np.cos(azimuth) * np.cos(course)
    - np.sin(azimuth) * np.sin(course)
) * 10 - speed_tangential
van = (
    -np.sin(elevation) * np.cos(azimuth) * np.sin(course)
    + np.sin(azimuth) * np.cos(course)
) * 10
var = (np.cos(elevation) * np.cos(azimuth)) * 10
# Plot the trajectory with a colorbar for tangential speed
plt.figure(figsize=(8, 6))
scatter = plt.scatter(
    azimuth, elevation, c=van / vatau, cmap="viridis", s=10
)  # `s` adjusts marker size
cbar = plt.colorbar(scatter)
cbar.set_label("Tangential Speed [m/s]", fontsize=12)

# Labels, title, and legend
plt.xlabel("Azimuth [rad]", fontsize=12)
plt.ylabel("Elevation [rad]", fontsize=12)
plt.title("Flown Trajectory with Tangential Speed", fontsize=14)
plt.legend(fontsize=10)
plt.grid()

plt.figure()
plt.plot(time[: i + 1], input_steering)
plt.show()

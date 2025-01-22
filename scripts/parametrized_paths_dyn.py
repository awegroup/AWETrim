import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from picawe.parametrized_patterns import Helix, Lissajous, FigureEight
from picawe.Kinematics import ParametrizedKinematics
from picawe import State
import casadi as ca
import time as timet

aero_input = {
    "model": "coeffs",
    "params": {
        "CD0": 0.1,
        "CL0": 0.257,
        "angle_pitch_depower_0": np.radians(-8),
        "delta_pitch_depower": np.radians(-9.0),
        "Cn_base": -0.01,
        # Add other aerodynamic parameters
    },
    "dependencies": {
        "alpha": {"k_cl": 4.615, "k_cd": 0.027, "k_cs": 0.0, "k_cn": 0.0},
        "alpha_squared": {"k_cl": -4.68, "k_cd": 1.217, "k_cs": 0.0, "k_cn": 0.0},
        "u_s": {"k_cl": 0, "k_cd": 0.15, "k_cs": 0.23, "k_cn": 0.005},  #
        "yaw_rate": {"k_cl": 0, "k_cd": 0, "k_cs": -0.01, "k_cn": -0.02},  #
        "sideslip": {
            "k_cl": 0,
            "k_cd": 0,
            "k_cs": 0.01,
            "k_cn": -0.05,
        },  # Cn 0.85 from Jelle
        "u_p": {"k_cl": 0, "k_cd": 0.0, "k_cs": 0, "k_cm": 0.01},  #  Cm 0.04 from Jelle
        # Add other dependencies as needed
    },
}
omega = -0.1
x0 = 200
rh = 60
vr = 0
beta = np.radians(30)
ry = 120
rz = 40
helix = Helix(omega, x0, rh, vr, beta)
lissajous = Lissajous(omega, x0, ry,rz, vr, beta)
figure_eight = FigureEight(omega, x0, 80, 80, vr, beta)

pattern = helix
kinematics = ParametrizedKinematics(pattern)
state = State(mass_wing=15, area_wing=20, aero_input=aero_input, mass_kcu = 25, dof = 3)

# Substitute the numeric values into the symbolic expressions using CasADi functions
chi_func = ca.Function('chi', [kinematics.t, kinematics.s, kinematics.s_dot], [kinematics.chi])

vk_func = ca.Function('vk', [kinematics.t, kinematics.s, kinematics.s_dot], [kinematics.vk])
dR_ds_func = ca.Function('dR_ds', [kinematics.t, kinematics.s, kinematics.s_dot], [ca.norm_2(kinematics.dR_ds)])

vr_func = ca.Function('vr', [kinematics.t], [kinematics.vr])
dot_chi_func = ca.Function('dot_chi', [kinematics.t, kinematics.s, kinematics.s_dot], [kinematics.dot_chi])
vtau_func = ca.Function('vtau', [kinematics.t, kinematics.s, kinematics.s_dot], [kinematics.vtau])

dot_vtau_func = ca.Function('dot_vtau', [kinematics.t, kinematics.s, kinematics.s_dot, kinematics.s_ddot], [kinematics.dot_vtau])
dot_vr_func = ca.Function('dot_vr', [kinematics.t, kinematics.s, kinematics.s_dot, kinematics.s_ddot], [kinematics.dot_vr])

state.timeder_speed_tangential = 0.0
state.timeder_speed_radial = 0.0
state.speed_wind = 10
state.input_depower = 0

solver_options = {
    'ipopt': {
        'print_level': 0,  # Suppresses IPOPT output
        # 'max_iter': 200,  # Maximum number of iterations
        'sb': 'yes'        # Suppresses more detailed solver information
    },
    'print_time': False    # Disables CasADi's internal timing output
}
time_step = 0.01
time = np.arange(0, 100, time_step)
s = 0
s_dot = 0.1
vk = 20
states = []
unknown_vars = ['length_tether', 'input_steering', 's_dot']
qs_guess = [200, 0, 40]
s_dot = ca.SX.sym('s_dot')
state.s_dot = s_dot
start_time = timet.time()

current_state = {
    "distance_radial": pattern.r(time[0]),
    "angle_elevation": pattern.elevation(time[0], s),
    "angle_azimuth": pattern.azimuth(time[0], s),
    "angle_course": chi_func(time[0], s, s_dot),
    "speed_radial": vr_func(time[0]),
    "speed_tangential": vtau_func(time[0], s, s_dot),
    "length_tether": pattern.r(time[0]),
    "timeder_angle_course": dot_chi_func(time[0], s, s_dot),
}
sol,_ = state.solve_quasi_steady_state(current_state,unknown_vars, qs_guess, solver_options=solver_options)
print(sol)
s_dot = float(sol[2])
state.s_dot = s_dot
s_ddot = ca.SX.sym('s_ddot')
state.s_ddot = s_ddot
unknown_vars = ['length_tether', 'input_steering', 's_ddot']
for i in range(len(time)):

    current_state = {
        "distance_radial": pattern.r(time[i]),
        "angle_elevation": pattern.elevation(time[i], s),
        "angle_azimuth": pattern.azimuth(time[i], s),
        "angle_course": chi_func(time[i], s, s_dot),
        "speed_radial": vr_func(time[i]),
        "speed_tangential": vtau_func(time[i], s, s_dot),
        "length_tether": pattern.r(time[i]),
        "timeder_angle_course": dot_chi_func(time[i], s, s_dot),
    }
    state.timeder_speed_tangential = dot_vtau_func(time[i], s, s_dot, s_ddot)
    state.timeder_speed_radial = dot_vr_func(time[i], s, s_dot, s_ddot)
    sol,_ = state.solve_quasi_steady_state(current_state,unknown_vars, qs_guess, solver_options=solver_options)

    qs_guess = sol
    # print(s_dot)
    
    s_dot += float(sol[2])*time_step
    s += s_dot*time_step
    state.s_dot = s_dot

    current_state["s"] = s
    current_state["input_steering"] = float(sol[1])
    current_state["length_tether"] = float(sol[0])
    current_state["angle_course"] = float(chi_func(time[i], s, s_dot))
    current_state["speed_tangential"] = float(vtau_func(time[i], s, s_dot))
    states.append(current_state)

    if abs(omega*s) > 6*np.pi:
        break
print(f"Time taken: {timet.time() - start_time}")
states = pd.DataFrame(states)

states = states[abs(states['s']*omega )> 4*np.pi]
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
scatter = plt.scatter(azimuth, elevation, c=speed_tangential, cmap='viridis', s=10)  # `s` adjusts marker size
cbar = plt.colorbar(scatter)
cbar.set_label('Tangential Speed [m/s]', fontsize=12)

# Plot max and min speed points
plt.scatter(azimuth[max_idx], elevation[max_idx], color='red', label='Max Speed', edgecolor='black', zorder=5)
plt.scatter(azimuth[min_idx], elevation[min_idx], color='red', label='Min Speed', edgecolor='black', zorder=5)

print(f"Max speed: {speed_tangential[max_idx]} m/s at phase {(s[max_idx]*omega*180/np.pi)%360} degrees")


# Labels, title, and legend
plt.xlabel('Azimuth [rad]', fontsize=12)
plt.ylabel('Elevation [rad]', fontsize=12)
plt.title('Flown Trajectory with Tangential Speed', fontsize=14)
plt.legend(fontsize=10)
plt.grid()
plt.show()

# Plot the trajectory with a colorbar for tangential speed
plt.figure(figsize=(8, 6))
scatter = plt.scatter(azimuth, elevation, c=course, cmap='viridis', s=10)  # `s` adjusts marker size
cbar = plt.colorbar(scatter)
cbar.set_label('Course angle', fontsize=12)

# Labels, title, and legend
plt.xlabel('Azimuth [rad]', fontsize=12)
plt.ylabel('Elevation [rad]', fontsize=12)
plt.legend(fontsize=10)
plt.grid()
plt.show()

# Plot the trajectory with a colorbar for tangential speed
plt.figure(figsize=(8, 6))
scatter = plt.scatter(azimuth, elevation, c=np.array(states['angle_roll']), cmap='viridis', s=10)  # `s` adjusts marker size
cbar = plt.colorbar(scatter)
cbar.set_label('Roll angle', fontsize=12)

# Labels, title, and legend
plt.xlabel('Azimuth [rad]', fontsize=12)
plt.ylabel('Elevation [rad]', fontsize=12)
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

vatau = (-np.sin(elevation)*np.cos(azimuth)*np.cos(course) - np.sin(azimuth)*np.sin(course) )*10 -speed_tangential
van = (-np.sin(elevation)*np.cos(azimuth)*np.sin(course) + np.sin(azimuth)*np.cos(course))*10
var = (np.cos(elevation)*np.cos(azimuth))*10
# Plot the trajectory with a colorbar for tangential speed
plt.figure(figsize=(8, 6))
scatter = plt.scatter(azimuth, elevation, c=van/vatau, cmap='viridis', s=10)  # `s` adjusts marker size
cbar = plt.colorbar(scatter)
cbar.set_label('Tangential Speed [m/s]', fontsize=12)

# Labels, title, and legend
plt.xlabel('Azimuth [rad]', fontsize=12)
plt.ylabel('Elevation [rad]', fontsize=12)
plt.title('Flown Trajectory with Tangential Speed', fontsize=14)
plt.legend(fontsize=10)
plt.grid()

plt.figure()
plt.plot(time[:i+1],input_steering)
plt.show()
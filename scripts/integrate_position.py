
from picawe import State
import numpy as np
import pandas as pd
import casadi as ca
import matplotlib.pyplot as plt

csv_file = './processed_data/VSM_results_alpha_sweep.csv'
v3_polar_data = pd.read_csv(csv_file)

#-----------------------------------------
# Define the system
#-----------------------------------------

aero_input = {
    "model": "polars",
    "params": {
        "CD0": 0.075,
        'CL': v3_polar_data['CL'].values, 
        'CD': v3_polar_data['CD'].values, 
        'alpha': np.radians(v3_polar_data['aoa'].values), 
        'angle_pitch_depower_0': np.radians(-10),
        'delta_pitch_depower': np.radians(-10.0),
        "Cn_base": -0.01,
        # Add other aerodynamic parameters
    },
    "dependencies": {
        "alpha": {},
        "u_s": {"k_cl": 0, "k_cd": 0.15, "k_cs": 0.23, "k_cm": 0.005},
        "yaw_rate": {"k_cl": 0, "k_cd": 0, "k_cs": -0.01, "k_cm": -0.02},
        "sideslip": {"k_cl": 0, "k_cd": 0, "k_cs": 0.01, "k_cm": -0.05},
        "u_p": {"k_cl": 0, "k_cd": 0., "k_cs": 0, "k_cn": 0.04},
        # Add other dependencies as needed
    },
}

# Example Usage
state = State(mass_wing=15, area_wing=20, aero_input=aero_input, mass_kcu = 10)

residual = state.rb_residual
ode = state.ode

dot_beta = state.extract_parameter_function('timeder_angle_elevation')
dot_theta = state.extract_parameter_function('timeder_angle_azimuth')

known_state = {
    'timeder_speed_tangential': 0.0,
    'timeder_speed_radial': 0.0,
    'input_steering': 0.0,
    'speed_radial': -3,
    'input_depower': 0,
    'speed_wind': 10,
}
for name, value in known_state.items():
    variable = getattr(state, name)
    residual = ca.substitute(residual, variable, value)
    ode = ca.substitute(ode, variable, value)
unknown_vars = ['length_tether', 'timeder_angle_course', 'speed_tangential', 'angle_roll', 'angle_pitch', 'angle_yaw']
sym_list = [getattr(state, name) for name in unknown_vars]
current_state = {
        'distance_radial': 100,
        'angle_course': np.radians(0),
        'angle_azimuth': 0,
        'angle_elevation': 0,
    }
solver_options = {
    'ipopt': {
        'print_level': 0,  # Suppresses IPOPT output
        # 'max_iter': 200,  # Maximum number of iterations
        'sb': 'yes'        # Suppresses more detailed solver information
    },
    'print_time': False    # Disables CasADi's internal timing output
}
time = np.arange(0, 100, 0.1)
vtau_guess = 40
states = []
ode_states = ['distance_radial', 'speed_radial', 'speed_tangential', 'angle_course', 'angle_azimuth', 'angle_elevation']
for i in range(len(time)):

    current_residual = residual
    current_ode = ode
    # print(current_state)
    # Substitute known values into the residual function
    for name, value in current_state.items():
        variable = getattr(state, name)
        current_residual = ca.substitute(current_residual, variable, value)

    # Solve for the unknown variables
    g = current_residual  # This is the vector of equations to solve
    # Bounds for the variables
    lbx = [current_state["distance_radial"]-5, -10, 0, -np.pi / 4, -np.pi / 4, -np.pi / 4]  # Lower bounds for T, u_s, speed_tangential, phi_k, theta_k
    ubx = [current_state["distance_radial"]+5, 10, 500, np.pi / 4, np.pi / 4, np.pi / 4]  # Upper bounds for T, u_s, speed_tangential, phi_k, theta_k
    
    # NLP problem definition
    nlp = {'x': ca.vertcat(*sym_list), 'f': 0, 'g': g}  # 'f' is set to 0 for root-finding

    # Define the NLP solver
    solver = ca.nlpsol('solver', 'ipopt', nlp, solver_options)

    # Bounds for the constraints
    lbg = [0] * g.size1()  # Lower bounds (0 for residuals)
    ubg = [0] * g.size1()  # Upper bounds (0 for residuals)

    # try:
    # Solve the system
    sol = solver(
        x0=[current_state["distance_radial"], 0, vtau_guess, 0, 0, 0],  # Initial guess
        lbg=lbg,
        ubg=ubg,
        lbx=lbx,
        ubx=ubx
    )
    qs_state = {
        'length_tether': float(sol['x'][0]),
        'timeder_angle_course': float(sol['x'][1]),
        'speed_tangential': float(sol['x'][2]),
        'angle_roll': float(sol['x'][3]),
        'angle_pitch': float(sol['x'][4]),
        'angle_yaw': float(sol['x'][5]),
    }
    vtau_guess = qs_state['speed_tangential']
    # print(qs_state['speed_tangential'])
    # Substitute known values into the ode function
    for name, value in current_state.items():
        if name in ode_states:
            continue
        else:
            variable = getattr(state, name)
            current_ode = ca.substitute(current_ode, variable, value)
    for name, value in qs_state.items():
        if name in ode_states:
            continue
        else:
            variable = getattr(state, name)
            current_ode = ca.substitute(current_ode, variable, value)

    # x = ca.vertcat(*[getattr(state, name) for name in state.state_vars])
    x = ca.vertcat(state.distance_radial, state.speed_radial, state.speed_tangential, state.angle_course, state.angle_azimuth, state.angle_elevation)
    # Integrate the system
    intg = ca.integrator('intg','cvodes',{'x':x,'ode':current_ode},time[i],time[i]+0.1)

    state_combined = {**current_state, **known_state, **qs_state}
    current_dot_beta = float(dot_beta(*[state_combined[name] for name in dot_beta.name_in()]))
    current_dot_theta = float(dot_theta(*[state_combined[name] for name in dot_theta.name_in()]))
    x0 = [state_combined["distance_radial"],
            state_combined["speed_radial"],
            state_combined["speed_tangential"],
            state_combined["angle_course"],
            state_combined["angle_azimuth"],
            state_combined["angle_elevation"]]
    res = intg(x0 = x0)
    current_state = {
        'distance_radial': float(res['xf'][0]),
        'angle_course': float(res['xf'][3]),
        'angle_azimuth': float(res['xf'][4]),
        'angle_elevation': float(res['xf'][5]),
    }
    if current_state["angle_elevation"] < 0 or current_state["distance_radial"] < 10:
        break
    states.append(state_combined)
    # print(current_state["distance_radial"])
    

solution_df = pd.DataFrame(states)

plt.figure()
plt.plot(solution_df['speed_tangential'])
plt.xlabel('Time [s]')
plt.ylabel('Speed Tangential [m/s]')

# Extract spherical coordinates
r = solution_df['distance_radial']  # Radial distance
theta = solution_df["angle_azimuth"]  # Azimuth angle in radians
phi = solution_df['angle_elevation']  # Elevation angle in radians


# Convert to Cartesian coordinates
x = r * np.cos(phi) * np.cos(theta)
y = r * np.cos(phi) * np.sin(theta)
z = r * np.sin(phi)

from mpl_toolkits.mplot3d import Axes3D
# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the trajectory
ax.plot(x, y, z, label='Trajectory in Cartesian Coordinates')
ax.set_xlabel('X [m]')
ax.set_ylabel('Y [m]')
ax.set_zlabel('Z [m]')
ax.set_ylim(-100, 100)
ax.legend()

plt.show()

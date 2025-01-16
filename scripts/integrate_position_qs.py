
from picawe import State
import numpy as np
import pandas as pd
import casadi as ca
import matplotlib.pyplot as plt

csv_file = './processed_data/VSM_results_alpha_sweep.csv'
v3_polar_data = pd.read_csv(csv_file)

#-----------------------------------------------
# Define the system
#-----------------------------------------------
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
state = State(mass_wing=15, area_wing=20, aero_input=aero_input, mass_kcu = 25)

residual = state.rb_residual
ode = state.ode

T_func = state.extract_parameter_function('tension_tether')
aoa_func = state.extract_parameter_function('angle_of_attack')

known_state = {
    'timeder_speed_tangential': 0.0,
    'timeder_speed_radial': 0.0,
    'timeder_angle_course': 0.0,
    'speed_radial': -2,
    'input_depower': 0,
    'speed_wind': 10,
}
for name, value in known_state.items():
    variable = getattr(state, name)
    residual = ca.substitute(residual, variable, value)
    ode = ca.substitute(ode, variable, value)
unknown_vars = ['length_tether', 'input_steering', 'speed_tangential', 'angle_roll', 'angle_pitch', 'angle_yaw']

current_state = {
        'distance_radial': 200,
        'angle_course': np.radians(0),
        'angle_azimuth': 0,
        'angle_elevation': np.radians(0),
    }
solver_options = {
    'ipopt': {
        'print_level': 0,  # Suppresses IPOPT output
        # 'max_iter': 200,  # Maximum number of iterations
        'sb': 'yes'        # Suppresses more detailed solver information
    },
    'print_time': False    # Disables CasADi's internal timing output
}
time_step = 0.1
time = np.arange(0, 50, time_step)
qs_guess = [100, 0, 40,0,0,0]
states = []
for i in range(len(time)):
    
    known_state.update(current_state)

    current_state = state.solve_quasi_steady_state(known_state, unknown_vars, qs_guess, solver_options= solver_options, dof = 6)

    qs_guess = [current_state[name] for name in unknown_vars]
    
    new_state = state.integrate(current_state, time[i], time_step)

    T = T_func(*[current_state[name] for name in T_func.name_in()])
    aoa = aoa_func(*[current_state[name] for name in aoa_func.name_in()])
    current_state["T"] = float(T)
    current_state["aoa"] = float(aoa)
    states.append(current_state)
    # print(current_state["input_steering"])
    current_state = {
        'distance_radial': float(new_state[0]),
        'angle_course': float(new_state[3]),
        'angle_azimuth': float(new_state[4]),
        'angle_elevation': float(new_state[5]),
    }

    if current_state["angle_elevation"] < 0 or current_state["distance_radial"] < 10:
        break


    # current_state["timeder_angle_course"] = qs_guess[2]/40
    # print(float(new_state[2]))    
    



print("Reel-in elevation angle: ", np.degrees(states[-1]["angle_elevation"]))
print("Reel-in tether force: ", states[-1]["T"])
solution_df = pd.DataFrame(states)

plt.figure()
plt.plot(solution_df['speed_tangential'])
plt.xlabel('Time [s]')
plt.ylabel('Speed Tangential [m/s]')

plt.figure()
plt.plot(solution_df['T'])
plt.xlabel('Time [s]')
plt.ylabel('Tether Tension [N]')

plt.figure()
plt.plot(np.degrees(solution_df['aoa']))
plt.xlabel('Time [s]')
plt.ylabel('Angle of Attack [deg]')
plt.show()


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
ax.set_xlim(0,200)
ax.set_zlim(0,200)
ax.legend()

plt.show()

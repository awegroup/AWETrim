import h5py
import pandas as pd
import numpy as np
from picawe import State
import casadi as ca
import time
def read_results(year, month, day, kite_model, addition="", path_to_main=""):
    path = "data/flight_logs/"
    date = str(year) + "-" + str(month) + "-" + str(day)
    file_name = str(kite_model) + "_" + date
    hdf5_path = path_to_main + path + file_name + addition + ".h5"
    ekf_output_df, flight_data_df, config_data = read_results_from_hdf5(hdf5_path)
    return ekf_output_df, flight_data_df, config_data

def read_results_from_hdf5(hdf5_path):
    with h5py.File(hdf5_path, 'r') as hf:
        # Read the ekf_output_df DataFrame
        ekf_group = hf['ekf_output']
        ekf_output_df = pd.DataFrame({col: ekf_group[col][:].astype(str) if ekf_group[col].dtype.kind == 'S' else ekf_group[col][:] 
                                      for col in ekf_group.keys()})
        
        # Read the flight_data DataFrame
        flight_group = hf['flight_data']
        flight_data_df = pd.DataFrame({col: flight_group[col][:].astype(str) if flight_group[col].dtype.kind == 'S' else flight_group[col][:] 
                                       for col in flight_group if isinstance(flight_group[col], h5py.Dataset)})
        
        # Read config_data
        config_group = hf['config_data']
        config_data = read_dict_from_group(config_group)

    return ekf_output_df, flight_data_df, config_data

def read_dict_from_group(group):
    config_dict = {}
    for key, value in group.attrs.items():
        if isinstance(value, bytes):
            value = value.decode('utf-8')  # Decode byte strings back to regular strings
        config_dict[key] = value

    for subgroup_name in group:
        subgroup = group[subgroup_name]
        config_dict[subgroup_name] = read_dict_from_group(subgroup)
    
    return config_dict

results, flight_data,config_data = read_results("2019", "10", "08", "v3",addition='')
mask = (flight_data.cycle.isin([62,63,64,65])) 

flight_data = flight_data[mask]
results = results[mask]
results = results.reset_index(drop=True)
flight_data = flight_data.reset_index(drop=True)

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
        "u_s": {"k_cl": 0, "k_cd": 0.15, "k_cs": 0.23, "k_cm": 0.005},    # 
        "yaw_rate": {"k_cl": 0, "k_cd": 0, "k_cs": -0.01, "k_cm": -0.02}, # 
        "sideslip": {"k_cl": 0, "k_cd": 0, "k_cs": 0.01, "k_cm": -0.05}, # Cm 0.85 from Jelle
        "u_p": {"k_cl": 0, "k_cd": 0., "k_cs": 0, "k_cn": 0.04},     #  Cn 0.04 from Jelle
        # Add other dependencies as needed
    },
}

# Example Usage
state = State(mass_wing=15, area_wing=20, mass_kcu = 25, aero_input=aero_input)

sideslip_func = state.extract_parameter_function('angle_sideslip')
cl_func = state.extract_parameter_function('CL')
cd_func = state.extract_parameter_function('CD')
T_func = state.extract_parameter_function('tension_tether')

speed_tangential = []
# Unknown inputs
solutions = []
u_s_guess = 0
speed_tangential_guess = 20
theta_guess = 1e-3
phi_guess = 1e-3
psi_guess = 1e-3
start = time.time()
vw_window = []
vw_averaged = []
wdir_window = []

residual = state.rb_residual
qs_state = {
    'timeder_speed_tangential': 0.0,
    'timeder_speed_radial': 0.0,
}
for name, value in qs_state.items():
    variable = getattr(state, name)
    residual = ca.substitute(residual, variable, value)
unknown_vars = ['length_tether', 'input_steering', 'speed_tangential', 'angle_roll', 'angle_pitch', 'angle_yaw']
# Define variables to solve for
sym_list = [getattr(state, name) for name in unknown_vars]
solver_options = {
    'ipopt': {
        'print_level': 0,  # Suppresses IPOPT output
        # 'max_iter': 200,  # Maximum number of iterations
        'sb': 'yes'        # Suppresses more detailed solver information
    },
    'print_time': False    # Disables CasADi's internal timing output
}
# Sliding window size
window_size = 5
for i, row in flight_data.iterrows():
    position = np.array([results.kite_position_x[i], results.kite_position_y[i], results.kite_position_z[i]])
    velocity = np.array([results.kite_velocity_x[i], results.kite_velocity_y[i], results.kite_velocity_z[i]])
    distance_radial = np.linalg.norm(position)
    speed_radial = np.dot(velocity, position)/distance_radial
    speed_radial = row.tether_reelout_speed
    speed_tangential.append(np.linalg.norm(velocity - np.dot(speed_radial,position/distance_radial)*position/distance_radial))
    
    # Wind speed (vw) sliding window average
    vw_window.append(results.wind_speed_horizontal[i])
    wdir_window.append(results.wind_direction[i])
    if len(vw_window) > window_size:
        vw_window.pop(0)  # Keep the window size constant
        wdir_window.pop(0)
    
    vw = np.mean(vw_window)  # Compute the average of the current window

    angle_azimuth = np.arctan2(position[1], position[0])-np.mean(wdir_window)

    current_state = {
        'speed_wind': vw,
        'distance_radial': distance_radial,
        'angle_course': row.kite_course,
        'timeder_angle_course': row.kite_yaw_rate_1,
        'speed_radial': speed_radial,
        'input_depower': row.up,
        'angle_azimuth': row.kite_azimuth,
        'angle_elevation': row.kite_elevation,
    }
    current_residual = residual
    # Substitute known values into the residual function
    for name, value in current_state.items():
        variable = getattr(state, name)
        current_residual = ca.substitute(current_residual, variable, value)


    g = current_residual  # This is the vector of equations to solve
    # Bounds for the variables
    lbx = [distance_radial-5, -1, 0, -np.pi / 4, -np.pi / 4, -np.pi / 4]  # Lower bounds for T, u_s, speed_tangential, phi_k, theta_k
    ubx = [distance_radial+5, 1, 500, np.pi / 4, np.pi / 4, np.pi / 4]  # Upper bounds for T, u_s, speed_tangential, phi_k, theta_k
    
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
        x0=[distance_radial, u_s_guess, speed_tangential_guess, theta_guess, phi_guess, psi_guess],  # Initial guess
        lbg=lbg,
        ubg=ubg,
        lbx=lbx,
        ubx=ubx
    )
    solution = sol['x']
    solution_dict = {
        'length_tether': float(solution[0]),
        'input_steering': float(solution[1]),
        'speed_tangential': float(solution[2]),
        'speed_radial': speed_radial,
        'angle_roll': float(solution[3]),
        'angle_pitch': float(solution[4]),
        'angle_yaw': float(solution[5]),
    }
    state_combined = {**current_state, **qs_state, **solution_dict}
    sideslip = sideslip_func(*[state_combined[name] for name in sideslip_func.name_in()])
    CL = cl_func(*[state_combined[name] for name in cl_func.name_in()])
    CD = cd_func(*[state_combined[name] for name in cd_func.name_in()])
    T = T_func(*[state_combined[name] for name in T_func.name_in()])
    solution_dict['sideslip'] = float(sideslip)
    solution_dict['CL'] = float(CL)
    solution_dict['CD'] = float(CD)
    solution_dict['tension_tether'] = float(T)
    solutions.append(solution_dict)
    # print(f"Solution: {solution}")
    T_guess = float(solution[0])
    u_s_guess = float(solution[1])
    speed_tangential_guess = float(solution[2])
    theta_guess = float(solution[3])
    phi_guess = float(solution[4])
    psi_guess = float(solution[5])
    print(i)
    # except RuntimeError as e: 
    #     # Handle solver failure
    #     solutions.append({
    #         'T': None,
    #         'u_s': None,
    #         'speed_tangential': None
    #     })


end = time.time()
print(f"Time taken: {end - start} seconds for {len(flight_data)} iterations")

# At dt of 0.1, the time taken is:
time_per_iteration = (end - start) / (len(flight_data))
print(f"Time per iteration: {time_per_iteration} seconds")

# Display the solutions

solutions_df = pd.DataFrame(solutions)
# Filter out rows where 'T' is None
solutions_df = solutions_df[solutions_df['length_tether'].notna()]

dt = 0.1
total_time = len(flight_data)*dt
# print('Estimated power: ', sum(solutions_df['T']*solutions_df['v_r']*dt)/total_time, 'W')
# print('Measured power: ', sum(flight_data['ground_tether_force']*flight_data['tether_reelout_speed']*dt)/total_time, 'W')



import matplotlib.pyplot as plt
plt.figure()
plt.plot(speed_tangential, label='tangential speed')
plt.plot(solutions_df['speed_tangential'], label='speed_tangential')
plt.legend()

plt.figure()
plt.plot(flight_data["ground_tether_force"], label='ground_tether_force')
plt.plot(solutions_df['tension_tether'], label='estimated tension_tether')
plt.legend()

plt.figure()
plt.plot(np.degrees(solutions_df['angle_roll']), label='roll tether-kite')
plt.plot(np.degrees(solutions_df['angle_pitch']), label='pitch tether-kite')
plt.plot(np.degrees(solutions_df['angle_yaw']), label='yaw tether-kite')
plt.legend()
# plt.show()

#Plot sideslip
plt.figure()
plt.plot(np.degrees(solutions_df['sideslip']), label='sideslip')
plt.legend()


# PLot steering input
plt.figure()
plt.plot(flight_data["us"], label='us measured')
plt.plot(solutions_df['input_steering'], label='u_s')
plt.legend()
# plt.show()


plt.figure()
plt.plot(results["wing_lift_coefficient"], label='wing_lift_coefficient')
plt.plot(solutions_df['CL'], label='CL')
plt.legend()
# plt.show()

plt.figure()
plt.plot(results["wing_drag_coefficient"], label='wing_drag_coefficient')
plt.plot(solutions_df['CD'], label='CD')
plt.legend()
# plt.show()

total_power = sum(solutions_df['tension_tether']*solutions_df['speed_radial']*dt)/total_time
measured_power = sum(flight_data['ground_tether_force']*flight_data['tether_reelout_speed']*dt)/total_time
print('Estimated power: ', total_power, 'W')
print('Measured power: ', measured_power, 'W')
plt.show()
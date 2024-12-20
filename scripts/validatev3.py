import h5py
import pandas as pd
import numpy as np
from picawe import KiteSystem, Environment, Control
import casadi as ca
import time
def read_results(year, month, day, kite_model, addition="", path_to_main=""):
    path = "data/"
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
mask = (flight_data.cycle.isin([60,61,62,63,64,65])) 

flight_data = flight_data[mask]
results = results[mask]
results = results.reset_index(drop=True)
flight_data = flight_data.reset_index(drop=True)

csv_file = './processed_data/VSM_results_alpha_sweep.csv'
v3_polar_data = pd.read_csv(csv_file)

aero_dict = {'CL': v3_polar_data['CL'].values, 
             'CD': v3_polar_data['CD'].values, 
             'alpha': np.radians(v3_polar_data['aoa'].values), 
             'steering_coefficient': 0.23,
             'k_cl_us': 0.2,
             'k_cd_us': 0.07,
             'k_cl_up': -0.12,
             'k_cd_up': 0.02,
             'theta_t_0': np.radians(-10),
             'delta_theta_up': np.radians(-18.0),
             "CD0": 0.075,
             }
aero_input = ["polars", aero_dict]


# Example Usage
kite = KiteSystem(m=15, A=20, m_kcu = 26, aero_input=aero_input)

# alpha_func = ca.Function('alpha', [kite.v_tau, kite.v_w, kite.beta, kite.chi, kite.phi, kite.v_r, kite.u_p, kite.r, kite.T], [kite.angle_of_attack], 
#                          ['v_tau', 'v_w', 'beta', 'chi', 'phi', 'v_r', 'u_p', 'r', 'T'], ['alpha'])
# aero_coeff_func = ca.Function('aero_coeffs', [kite.v_tau, kite.v_w, kite.beta, kite.chi, kite.phi, kite.v_r, kite.u_p, kite.u_s,kite.r, kite.T], kite.aerodynamic_coeffs, 
#                               ['v_tau', 'v_w', 'beta', 'chi', 'phi', 'v_r', 'u_p', 'u_s', 'r', 'T'], ['CL', 'CD', 'CS'])
residual_func = kite.get_residual_function()

environment = Environment(g=9.81, rho=1.225)
residual_func = environment.apply(residual_func)

speed_tangential = []
# Unknown inputs
T = ca.SX.sym('T')  # Tether tension
u_s = ca.SX.sym('u_s')  # Steering input
v_tau = ca.SX.sym('v_tau')  # Tangential velocity
solutions = []
T_guess = 10000
u_s_guess = 0
v_tau_guess = 20
start = time.time()
vw_window = []
vw_averaged = []
wdir_window = []

# Sliding window size
window_size = 10
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

    residual = residual_func(
        dot_v_tau=0,
        r=distance_radial,
        chi=row.kite_course, # Kite course angle
        beta=row.kite_elevation,  # Current beta
        phi=row.kite_azimuth,  # Current phi
        u_s=u_s,
        T=T,
        v_tau=v_tau,
        v_w=vw,
        dot_chi = row.kite_yaw_rate_1,
        v_r = speed_radial,
        u_p = row.up,
    )

        # Define partial_residual_func for this combination
    partial_residual_func = ca.Function(
        'partial_residual_func', [T, u_s, v_tau],
        [residual["residual"]], ['T', 'us', 'v_tau'], ['residual']
    )

    # Define the rootfinder
    rf = ca.rootfinder(
        'rf', 'newton',
        {'x': ca.vertcat(T, u_s, v_tau), 'g': partial_residual_func(T, u_s, v_tau)}
    )

    try:
        # Solve the system
        solution = rf([T_guess, u_s_guess, v_tau_guess], [])
        # alpha = alpha_func(solution[2], vw, row.kite_elevation, row.kite_course, row.kite_azimuth, speed_radial, row.up, distance_radial, solution[0])
        # CL,CD,CS = aero_coeff_func(solution[2], vw, row.kite_elevation, row.kite_course, row.kite_azimuth, speed_radial, row.up, solution[1])
        solutions.append({
            'T': float(solution[0]),
            'u_s': float(solution[1]),
            'v_tau': float(solution[2]),
            'v_r': speed_radial,
            # 'alpha': float(alpha),
            # 'CL': float(CL),
            # 'CD': float(CD),
            # 'CS': float(CS),
        })
        # T_guess = float(solution[0])
        # u_s_guess = float(solution[1])
        # v_tau_guess = float(solution[2])
    except RuntimeError as e:
        # Handle solver failure
        solutions.append({
            'T': None,
            'u_s': None,
            'v_tau': None
        })


end = time.time()
print(f"Time taken: {end - start} seconds for {len(flight_data)} iterations")

# At dt of 0.1, the time taken is:
time_per_iteration = (end - start) / (len(flight_data))
print(f"Time per iteration: {time_per_iteration} seconds")

# Display the solutions

solutions_df = pd.DataFrame(solutions)
# Filter out rows where 'T' is None
solutions_df = solutions_df[solutions_df['T'].notna()]

dt = 0.1
total_time = len(flight_data)*dt
# print('Estimated power: ', sum(solutions_df['T']*solutions_df['v_r']*dt)/total_time, 'W')
# print('Measured power: ', sum(flight_data['ground_tether_force']*flight_data['tether_reelout_speed']*dt)/total_time, 'W')



import matplotlib.pyplot as plt
plt.figure()
plt.plot(speed_tangential, label='tangential speed')
plt.plot(solutions_df['v_tau'], label='v_tau')
plt.legend()

plt.figure()
plt.plot(flight_data["ground_tether_force"], label='ground_tether_force')
plt.plot(solutions_df['T'], label='T')
plt.legend()

plt.show()

# plt.figure()
# plt.plot(solutions_df['alpha']*180/np.pi, label='alpha')


# plt.figure()
# plt.plot(solutions_df['CL'], label='CL')
# plt.plot(results["wing_lift_coefficient"], label='kite_lift_coefficient')

# plt.figure()
# plt.plot(solutions_df['CD'], label='CD')
# plt.plot(results["wing_drag_coefficient"], label='kite_drag_coefficient')
# plt.show()


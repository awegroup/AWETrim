import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
mask = (flight_data.cycle.isin([65])) 

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
             }
aero_input = ["polars", aero_dict]


# Example Usage
kite = KiteSystem(m=45, A=20, aero_input=aero_input)


residual_func = kite.get_residual_function_aero_iden()

environment = Environment(g=9.81, rho=1.225)
residual_func = environment.apply(residual_func)

speed_tangential = []
# Unknown inputs
T = ca.SX.sym('T')  # Tether tension
u_s = ca.SX.sym('u_s')  # Steering input
v_tau = ca.SX.sym('v_tau')  # Tangential velocity
delta_theta_up = ca.SX.sym('delta_theta_up')  # Upwind angle
theta_t_0 = ca.SX.sym('theta_t_0')  # Tether angle
# alpha_func = ca.Function('alpha', [kite.v_tau, kite.v_w, kite.beta, kite.chi, kite.phi, kite.v_r, kite.u_p],
#                          [kite.angle_of_attack], ['v_tau', 'v_w', 'beta', 'chi', 'phi', 'v_r', 'u_p'], ['alpha'])
aero_coeff_func = ca.Function('aero_coeffs', [kite.v_tau, kite.v_w, kite.beta, kite.chi, kite.phi, kite.v_r, kite.u_p, kite.u_s, kite.theta_t_0, kite.delta_theta_up,
                                              kite.k_cl_us, kite.k_cd_us, kite.k_cl_up, kite.k_cd_up, kite.CD0],
                              kite.aerodynamic_coeffs, ['v_tau', 'v_w', 'beta', 'chi', 'phi', 'v_r', 'u_p', 'u_s', 'theta_t_0', 'delta_theta_up', 'k_cl_us', 'k_cd_us', 'k_cl_up', 'k_cd_up', 'CD0'],
                              ['CL', 'CD', 'CS'])

CD0 = ca.SX.sym('CD0')  # 0 drag coeff
k_cd_us = ca.SX.sym('k_cd_us')  # Drag coeff us
k_cl_us = ca.SX.sym('k_cl_us')  # Lift coeff us
k_cd_up = ca.SX.sym('k_cd_up')  # Drag coeff up
k_cl_up = ca.SX.sym('k_cl_up')  # Lift coeff up
T_guess = 1000
u_s_guess = 0
v_tau_guess = 20
N = len(flight_data)
opti = ca.Opti()
T_var = opti.variable(N)
u_s_var = opti.variable(N)
v_tau_var = opti.variable(N)
delta_theta_var = opti.variable()
theta_t_0_var = opti.variable()
CD0_var = opti.variable()
k_cd_us_var = opti.variable()
k_cl_us_var = opti.variable()
k_cd_up_var = opti.variable()
k_cl_up_var = opti.variable()
start = time.time()
res = 0
vw_window = []
vw_averaged = []

# Sliding window size
window_size = 10
for i, row in flight_data.iterrows():
    if i == N:
        break
    position = np.array([results.kite_position_x[i], results.kite_position_y[i], results.kite_position_z[i]])
    velocity = np.array([results.kite_velocity_x[i], results.kite_velocity_y[i], results.kite_velocity_z[i]])
    distance_radial = np.linalg.norm(position)
    speed_radial = np.dot(velocity, position)/distance_radial
    speed_radial = row.tether_reelout_speed
    speed_tangential.append(np.linalg.norm(velocity - np.dot(speed_radial,position/distance_radial)*position/distance_radial))
 
    angle_azimuth = np.arctan2(position[1], position[0])-results.wind_direction[i]

    # Wind speed (vw) sliding window average
    vw_window.append(results.wind_speed_horizontal[i])
    if len(vw_window) > window_size:
        vw_window.pop(0)  # Keep the window size constant
    
    vw = np.mean(vw_window)  # Compute the average of the current window

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
        delta_theta_up = delta_theta_up,
        theta_t_0 = theta_t_0,
        CD0 = CD0,
        k_cd_us = k_cd_us,
        k_cl_us = k_cl_us,
        k_cd_up = k_cd_up,
        k_cl_up = k_cl_up
    )

        # Define partial_residual_func for this combination
    partial_residual_func = ca.Function(
        'partial_residual_func', [T, u_s, v_tau, delta_theta_up, theta_t_0, CD0, k_cd_us, k_cl_us, k_cd_up, k_cl_up], 
        [residual["residual"]], ['T', 'us', 'v_tau','delta_theta_up', 'theta_t_0', 'CD0', 'k_cd_us', 'k_cl_us', 'k_cd_up', 'k_cl_up'], ['residual']
    )
    opti.subject_to(partial_residual_func(T_var[i], u_s_var[i], v_tau_var[i], delta_theta_var, theta_t_0_var,CD0_var, k_cd_us_var, k_cl_us_var, k_cd_up_var, k_cl_up_var) == 0)
    CL, CD, CS = aero_coeff_func(v_tau_var[i], vw, row.kite_elevation, row.kite_course, row.kite_azimuth, speed_radial, row.up, u_s_var[i], theta_t_0_var, delta_theta_var, k_cl_us_var, k_cd_us_var, k_cl_up_var, k_cd_up_var, CD0_var)
    res += (T_var[i] - row.ground_tether_force)**2/4000
    res += (v_tau_var[i] - speed_tangential[-1])**2/20
    res += (CL - results.wing_lift_coefficient[i])**2/0.5

end = time.time()
print(f"Time taken: {end - start} seconds for {len(flight_data)} iterations")

# At dt of 0.1, the time taken is:
time_per_iteration = (end - start) / (len(flight_data))
print(f"Time per iteration: {time_per_iteration} seconds")


# Display the solutions


opti.minimize(res)

# Initialize
opti.set_initial(T_var, flight_data.ground_tether_force.iloc[:N])
opti.set_initial(u_s_var, flight_data.us.iloc[:N])
opti.set_initial(v_tau_var, speed_tangential)
opti.set_initial(delta_theta_var, -np.radians(12))
opti.set_initial(theta_t_0_var, -np.radians(10))
opti.set_initial(CD0_var, 0.05)
opti.set_initial(k_cd_us_var, 0.1)

# BOundaries
opti.subject_to(opti.bounded(500, T_var, 10000))
opti.subject_to(opti.bounded(5, v_tau_var, 40))
# opti.subject_to(v_tau_var > 0)
# opti.subject_to(opti.bounded(-1, u_s_var, 1))
opti.subject_to(opti.bounded(-np.radians(12), delta_theta_var, -np.radians(8)))
opti.subject_to(opti.bounded(-np.radians(15), theta_t_0_var, -np.radians(5)))
opti.subject_to(opti.bounded(0.01, CD0_var, 0.1))
opti.subject_to(opti.bounded(0, k_cd_us_var, 0.2))
opti.subject_to(opti.bounded(-0.2, k_cl_us_var, 0.4))
opti.subject_to(opti.bounded(-0.2, k_cd_up_var, 0.2))
opti.subject_to(opti.bounded(-0.2, k_cl_up_var, 0.2))


opti.solver('ipopt')
try:
    sol = opti.solve()
    # If the solver converges, retrieve the solution
    T_sol = sol.value(T_var)
    # u_s_sol = sol.value(u_s_var)
    v_tau_sol = sol.value(v_tau_var)
    print(sol.value(delta_theta_var)*180/np.pi)
    print(sol.value(theta_t_0_var)*180/np.pi)
    print(sol.value(CD0_var))
    print(sol.value(k_cd_us_var))
    print(sol.value(k_cl_us_var))
    print(sol.value(k_cd_up_var))
    print(sol.value(k_cl_up_var))
except RuntimeError as e:
    # If the solver fails, retrieve the last computed values
    print("Solver failed to converge:", e)
    T_sol = opti.debug.value(T_var)
    # u_s_sol = opti.debug.value(u_s_var)
    v_tau_sol = opti.debug.value(v_tau_var)
    print(opti.debug.value(delta_theta_var)*180/np.pi)
    print(opti.debug.value(theta_t_0_var)*180/np.pi)
    print(opti.debug.value(CD0_var))
    print(opti.debug.value(k_cd_us_var))
    print(opti.debug.value(k_cl_us_var))
    print(opti.debug.value(k_cd_up_var))
    print(opti.debug.value(k_cl_up_var))


# print(sol.value(T_var), row.ground_tether_force)
# print(sol.value(u_s_var), row.us)
# print(sol.value(v_tau_var), speed_tangential[-1])
# print(a_sol*180/np.pi)

# Usol_SS = sol.value(a) # should be [-2.7038;-0.5430;0.2613;0.5840]
# Filter out rows where 'T' is None
# solutions_df = pd.DataFrame(solutions)
# solutions_df = solutions_df[solutions_df['T'].notna()]


plt.figure()
plt.plot(speed_tangential, label='tangential speed')
plt.plot(v_tau_sol, label='v_tau')
plt.legend()

plt.figure()
plt.plot(flight_data["ground_tether_force"], label='ground_tether_force')
plt.plot(T_sol, label='T')
plt.legend()
# plt.show()

plt.figure()
plt.plot(flight_data["us"], label='us')
# plt.plot(u_s_sol, label='u_s')
plt.legend()
plt.show()

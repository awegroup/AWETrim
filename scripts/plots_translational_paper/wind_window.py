import numpy as np
from picawe import SystemModel
import pandas as pd
import matplotlib.pyplot as plt
import time
from picawe.system.kite import Kite
import json


aero_dict = {
    "oswald_efficiency": 0.9,
    "aspect_ratio": 10,
    "steering_coefficient": 0.2,
    "CD0": 0.05,
    "theta_t_0": np.radians(0),
    "delta_theta_up": np.radians(-18.0),
}
aero_input = ["inviscid", aero_dict]

# -----------------------------------------------
# Load data and define aerodynamic model
# -----------------------------------------------

# Define aerodynamic input
file_path = "./data/v3_aero_input.json"
with open(file_path, "r") as file:
    aero_input = json.load(file)


aero_input =    {
        "model": "inviscid",
        "params": {
            "CD0": 0.05,
            "aspect_ratio": 10,
            "oswald_efficiency": 1,
            "angle_pitch_depower_0": 0,
        },
       "dependencies": {
        # "u_s": { "k_cl": 0, "k_cd": 0.0, "k_cs": 0.23, "k_cn": 0.005 },
    } 

    }
# -----------------------------------------------
# Define the state
# -----------------------------------------------
kite = Kite(mass_wing=80, area_wing=20, aero_input=aero_input, mass_kcu=0, steering_control="roll")
state = SystemModel( dof=3, quasi_steady=True, wind_model="uniform", kite=kite)

speed_wind = 10
state.speed_wind_ref = speed_wind
state.input_depower = 0


# Constants

state.distance_radial = 200.0
state.speed_radial = 0.0
state.input_depower = 0.0
state.timeder_angle_course = 0.0

unknown_vars = [
    "tension_tether_ground",
    "input_steering",
    "speed_tangential",
]

# Define the range of phi and beta
phi_values = np.radians(np.linspace(-90, 90, 30))  # Range for phi in radians
beta_values = np.radians(np.linspace(0, 90, 30))  # Range for beta in radians

# Generate combinations of phi and beta using meshgrid
phi_grid, beta_grid = np.meshgrid(phi_values, beta_values)

# Flatten the grids to create pairwise combinations
phi_combinations = phi_grid.flatten()
beta_combinations = beta_grid.flatten()

# Compute the absolute distance of phi from 0
phi_distances = np.abs(phi_combinations)

# Sort the indices based on the distance from 0
sorted_indices = np.argsort(phi_distances)

# Reorder the combinations
phi_combinations = phi_combinations[sorted_indices]
beta_combinations = beta_combinations[sorted_indices]


qs_guess = [1000, 0, 40]

solve_func, inputs_name = state.solve_quasi_steady_state(
        unknown_vars
    )
print(solve_func)

angles_course = [np.pi/2, -np.pi/2, 0, np.pi]
ww_course = []
for angle_course in angles_course:
    # Loop over combinations of phi and beta
    wind_window = []
    for phi, beta in zip(phi_combinations, beta_combinations):

        current_state = {
            "distance_radial": state.distance_radial,
            "angle_elevation": beta,
            "angle_azimuth": phi,
            "angle_course": angle_course,
        }

        p = [current_state[name] for name in inputs_name]
        # print(p)
        lbx,ubx,lbg,ubg = state.get_boundaries(unknown_vars)
        # print(lbx,ubx,lbg,ubg)
        sol = solve_func(x0=qs_guess, p=p, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)

        

        if np.linalg.norm(sol["g"]) < 1e-6 and sol["x"][0] > 0 and sol["x"][2] >0:
            # qs_guess = sol["x"]
            qs_state = {name: float(sol["x"][i]) for i, name in enumerate(unknown_vars)}
            current_state = qs_state
            current_state["angle_azimuth"] = phi
            current_state["angle_elevation"] = beta
            # current_state["alpha"] = float(alpha_value)
            # qs_guess = [current_state[name] for name in unknown_vars]
            wind_window.append(current_state)
        else:
            print("Quasi steady solution not found")
    ww_course.append(pd.DataFrame(wind_window))


for i in range(len(angles_course)):
    sol = ww_course[i]
    az = []
    el = []
    for beta in beta_combinations:
        try:
            idx = np.argmax(sol[sol.angle_elevation == beta].angle_azimuth)
            az.append(sol[sol.angle_elevation == beta].angle_azimuth.iloc[idx])
            el.append(beta)
            idx = np.argmin(sol[sol.angle_elevation == beta].angle_azimuth)
            az.append(sol[sol.angle_elevation == beta].angle_azimuth.iloc[idx])
            el.append(beta)
        except:
            pass
    
    y = 1*np.cos(el)*np.sin(az)
    z = 1*np.sin(el)
    sorted_indices = np.argsort(y)
    y = np.array(y)[sorted_indices]
    z = np.array(z)[sorted_indices]
    plt.plot(y, z, label=f"Course: {angles_course[i]}")

plt.xlabel("Azimuth [rad]")
plt.ylabel("Elevation [rad]")
plt.legend()
plt.show()
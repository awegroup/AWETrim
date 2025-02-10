import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from picawe.parametrized_patterns import Helix, Lissajous, FigureEight
from picawe.Kinematics import ParametrizedKinematics
from picawe import State
import casadi as ca
import time as timet
import json
import os
from picawe.color_palette import get_color_list, set_plot_style, set_plot_style_no_latex

save_folder = "./results/plots_point_mass/"
# -----------------------------------------------
# Load data and define aerodynamic model
# -----------------------------------------------
#Load initial state

file_path = "./results/impact_inertial_forces/"
file_name = "helix_quasi_steady.csv"
results = pd.read_csv(os.path.join(file_path, file_name))

set_plot_style()
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
# State.__bases__ = (KiteKinematics, Tether, Wind, RigidKite)
mass_wing = 80
state = State(mass_wing=mass_wing, area_wing=20, aero_input=aero_input, mass_kcu=0, dof=3, quasi_steady=True, steering_control="roll")

speed_wind = 18
state.speed_wind = speed_wind
state.input_depower = 0
state.timeder_angle_course = 0
state.distance_radial = 200

solver_options = {
    "ipopt": {"print_level": 0, "sb": "yes","tol": 1e-8,},
    "print_time": False,
}
# state.override_gravity = True
state.establish_residual()
variables = ca.symvar(state.residual)
name_vars = [var.name() for var in variables]
residual = ca.Function("residual", variables, [state.residual], name_vars, ["residual"])
print(residual)
angles_elevation = np.linspace(0, 90, 10)/180*np.pi
angles_course = [np.pi/2, 0, np.pi]
f_el = np.zeros((len(angles_elevation), len(angles_course)))
ft_el = np.zeros((len(angles_elevation), len(angles_course)))
for i, beta in enumerate(angles_elevation):
        speed_tangential_i = 30
        tension_tether_i = 1000
        for k,angle_course in enumerate(angles_course):
            opti = ca.Opti()  
            reeling_factor = opti.variable()   
            speed_tangential = opti.variable()
            angle_roll = opti.variable()
            tension_tether = opti.variable()
            opti.subject_to(residual(angle_course = angle_course,
                                    angle_azimuth = 0,
                                        angle_elevation = beta,
                                        speed_tangential = speed_tangential,
                                        angle_roll = angle_roll,
                                        # speed_wind = speed_wind,
                                        speed_radial = reeling_factor*speed_wind,
                                        tension_tether_ground = tension_tether)["residual"] == 0)
            
            opti.subject_to(0 <= (reeling_factor <= 0.4))
            opti.subject_to(mass_wing*9.81 <= (tension_tether))
            opti.subject_to(0 <= (speed_tangential))
            opti.set_initial(reeling_factor, 0.3)
            opti.set_initial(speed_tangential, 50)
            opti.set_initial(angle_roll, 0)
            opti.set_initial(tension_tether, 1e4)
            opti.minimize(-tension_tether*reeling_factor*speed_wind)
            opti.solver("ipopt", solver_options)
            try:
                sol = opti.solve()
                f_el[i,k] = sol.value(reeling_factor)
                ft_el[i,k] = sol.value(tension_tether)
                speed_tangential_i = sol.value(speed_tangential)
                tension_tether_i = sol.value(tension_tether)
            except:
                f_el[i,k] = np.nan
                ft_el[i,k] = np.nan
                print("Failed at elevation angle: ", beta, " and course angle: ", angle_course)

angles_azimuth = np.linspace(0, 90, 10)/180*np.pi
f_az = np.zeros((len(angles_azimuth), len(angles_course)))
ft_az = np.zeros((len(angles_azimuth), len(angles_course)))
for i, phi in enumerate(angles_azimuth):
        speed_tangential_i = 30
        tension_tether_i = 1000
        for k,angle_course in enumerate(angles_course):
            opti = ca.Opti()  
            reeling_factor = opti.variable()   
            speed_tangential = opti.variable()
            angle_roll = opti.variable()
            tension_tether = opti.variable()
            opti.subject_to(residual(angle_course = angle_course,
                                    angle_azimuth = phi,
                                        angle_elevation = 0,
                                        speed_tangential = speed_tangential,
                                        angle_roll = angle_roll,
                                        # speed_wind = speed_wind,
                                        speed_radial = reeling_factor*speed_wind,
                                        tension_tether_ground = tension_tether)["residual"] == 0)
            
            opti.subject_to(0 <= (reeling_factor <= 0.4))
            opti.subject_to(mass_wing*9.81 <= (tension_tether))
            opti.subject_to(0 <= (speed_tangential))
            opti.set_initial(reeling_factor, 0.3)
            opti.set_initial(speed_tangential, 50)
            opti.set_initial(angle_roll, 0)
            opti.set_initial(tension_tether, 1e4)
            opti.minimize(-tension_tether*reeling_factor*speed_wind)
            opti.solver("ipopt", solver_options)
            try:
                sol = opti.solve()
                f_az[i,k] = sol.value(reeling_factor)
                ft_az[i,k] = sol.value(tension_tether)
                speed_tangential_i = sol.value(speed_tangential)
                tension_tether_i = sol.value(tension_tether)
            except:
                f_az[i,k] = np.nan
                ft_az[i,k] = np.nan
                print("Failed at azimuth angle: ", phi, " and course angle: ", angle_course)

# -----------------------------------------------
# Plot the results
# -----------------------------------------------
colors = get_color_list()
plt.figure(figsize=(5,4))
for i, chi in enumerate(angles_course):
    x = f_el[:,i]*speed_wind
    y = ft_el[:,i]
    plt.plot(x,y, label = r"$\chi$ = "+str(chi*180/np.pi)+r"$^\circ$, $\phi$ = 0$^\circ$", color = colors[i])

for i, chi in enumerate(angles_course):
    x = f_az[:,i]*speed_wind
    y = ft_az[:,i]
    plt.plot(x,y, linestyle = '--', label = r"$\chi$ = "+str(chi*180/np.pi)+r"$^\circ$, $\beta$ = 0$^\circ$", color = colors[i])


plt.ylabel("Tension Tether (N)")
plt.xlabel("Reeling Speed (m/s)")
plt.legend()
plt.tight_layout()
# plt.savefig(save_folder + "optimal_reeling_speed_force.pdf")
plt.show()

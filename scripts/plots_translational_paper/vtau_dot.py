import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from picawe.system.kite import Kite
from picawe import SystemModel
import casadi as ca
import time as timet
import json
from picawe.utils.color_palette import get_color_list, set_plot_style, set_plot_style_no_latex

save_folder = "./results/figures/translational_paper/"
# -----------------------------------------------
# Load data and define aerodynamic model
# -----------------------------------------------
set_plot_style()
# Define aerodynamic input
file_path = "./data/v3_aero_input.json"
# file_path = "./data/rigid_kite.json"
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
        "u_s": { "k_cl": 0, "k_cd": 0.0, "k_cs": 0.23, "k_cn": 0.005 },
    } 

    }

# -----------------------------------------------
# Define the state
# -----------------------------------------------
# State.__bases__ = (KiteKinematics, Tether, Wind, RigidKite)
kite = Kite(mass_wing=80, area_wing=20, aero_input=aero_input, mass_kcu=0, steering_control="roll")
state = SystemModel(dof=3, quasi_steady=True, wind_model="uniform", kite=kite)

speed_wind = 10
state.speed_wind_ref = speed_wind
state.input_depower = 0
state.timeder_angle_course = 0
state.angle_course = np.pi/2
state.angle_elevation = 0
state.angle_azimuth = 0
state.input_steering = 0
state.angle_roll = 0
state.speed_radial = 2
state.distance_radial = 200

courses = [np.pi/2,0, np.pi]
speed_tangential = np.linspace(30,90,100)
fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
axs[0].axhline(y=0, color='gray', linewidth=0.5)
axs[1].axhline(y=0, color='gray', linewidth=0.5)
print(state.speed_wind)
for course in courses:
    vtau_dot = []
    aoa = []
    state.angle_course = course
    for vtau in speed_tangential:

        state.speed_tangential = vtau
        vtau_dot.append(float(state.force_residual[0]))
        aoa.append(float(state.angle_of_attack))

    axs[0].plot(speed_tangential, vtau_dot, label=f"$\chi$: {np.degrees(course)}$^\circ$")
    axs[1].plot(np.degrees(aoa), vtau_dot)


axs[0].set_xlabel(r"$v_{\tau}$ [$m s^{-1}$]")
axs[0].set_ylabel(r"$\dot{v}_\tau$ [$m s^{-2}$]")
fig.legend(loc="upper center", bbox_to_anchor=(0.5, 0.8), ncol=len(courses)//2, frameon=True)
axs[1].set_xlabel(r"$\alpha$ [$^\circ$]")
axs[0].set_xlim([min(speed_tangential), max(speed_tangential)])
axs[1].set_xlim([min(np.degrees(aoa)), max(np.degrees(aoa))])


#Save the figure as pdf
plt.tight_layout()
plt.savefig(save_folder + "vtau_dot_loyd.pdf")
plt.show()



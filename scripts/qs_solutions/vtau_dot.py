import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import casadi as ca

from picawe.system.kite import Kite
from picawe import SystemModel
from picawe.utils.color_palette import get_color_list, set_plot_style_no_latex

# ------------------------------
# Setup and configuration
# ------------------------------
set_plot_style_no_latex()
save_folder = "./results/figures/translational_paper/"

# Load aerodynamic input
file_path = "./data/LEI-V3-KITE/v3_aero_input.json"
with open(file_path, "r") as file:
    aero_input = json.load(file)

# ------------------------------
# Initialize system model
# ------------------------------
kite = Kite(
    mass_wing=80,
    area_wing=20,
    aero_input=aero_input,
    mass_kcu=0,
    steering_control="roll",
)
state = SystemModel(dof=3, quasi_steady=True, kite=kite)

# Fixed parameters
state.wind.speed_wind_ref = 10
state.input_depower = 0
state.timeder_angle_course = 0
state.input_steering = 0
state.angle_elevation = 0
state.angle_azimuth = 0
state.speed_radial = 2
state.distance_radial = 200

# ------------------------------
# Sweep over tangential speeds
# ------------------------------
courses = [np.pi / 2, 0, np.pi]  # course angles in radians
speed_tangential = np.linspace(0, 60, 100)

fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
axs[0].axhline(y=0, color="gray", linewidth=0.5)
axs[1].axhline(y=0, color="gray", linewidth=0.5)

# Evaluate force residual and angle of attack across speeds
for course in courses:
    vtau_dot = []
    aoa = []
    state.angle_course = course

    for vtau in speed_tangential:
        state.speed_tangential = vtau
        vtau_dot.append(float(state.force_residual[0]))
        aoa.append(float(state.angle_of_attack))

    axs[0].plot(speed_tangential, vtau_dot, label=f"$\\chi$: {np.degrees(course)}°")
    axs[1].plot(np.degrees(aoa), vtau_dot)

# ------------------------------
# Plot formatting
# ------------------------------
axs[0].set_xlabel(r"$v_{\tau}$ [m/s]")
axs[0].set_ylabel(r"$\dot{v}_\tau$ [m/s²]")
axs[1].set_xlabel(r"$\alpha$ [°]")

fig.legend(
    loc="upper center", bbox_to_anchor=(0.5, 0.8), ncol=len(courses) // 2, frameon=True
)

plt.tight_layout()
plt.savefig(save_folder + "vtau_dot_loyd.pdf")
plt.show()

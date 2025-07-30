import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import casadi as ca

from picawe.system.kite import Kite
from picawe import SystemModel
from picawe.utils.color_palette import get_color_list, set_plot_style

# ------------------------------
# Setup and configuration
# ------------------------------
set_plot_style()
save_folder = "./results/figures/translational_paper/"

# Load aerodynamic input
file_path = "./data/LEI-V3-KITE/v3_aero_input.json"
with open(file_path, "r") as file:
    aero_input = json.load(file)

# ------------------------------
# Initialize system model
# ------------------------------
kite = Kite(
    mass_wing=120,
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
state.speed_radial = 0
state.distance_radial = 200

# ------------------------------
# Sweep over tangential speeds
# ------------------------------
courses = [np.pi / 2, 0, np.pi]  # course angles in radians
speed_tangential = np.linspace(20, 60, 100)
colors = get_color_list()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3), sharey=True)

ax1.axhline(y=0, color="gray", linewidth=0.5)
ax2.axhline(y=0, color="gray", linewidth=0.5)

# Evaluate force residual and angle of attack across speeds
for i, course in enumerate(courses):
    vtau_dot = []
    aoa = []
    state.angle_course = course
    color = colors[i]

    for vtau in speed_tangential:
        state.speed_tangential = vtau
        vtau_dot.append(float(state.force_residual[0]))
        aoa.append(float(state.angle_of_attack))

    # Plot vtau_dot vs speed_tangential on left subplot
    ax1.plot(
        speed_tangential,
        vtau_dot,
        color=color,
        linestyle="-",
        label=f"$\\chi$: {np.degrees(course)}°",
    )

    # Plot vtau_dot vs angle of attack on right subplot
    ax2.plot(np.degrees(aoa), vtau_dot, color=color, linestyle="--")

# ------------------------------
# Plot formatting
# ------------------------------
ax1.set_xlabel(r"$v_{\tau}$ [m/s]")
ax1.set_ylabel(r"$\dot{v}_\tau$ [m/s²]")
ax2.set_xlabel(r"$\alpha$ [°]")

ax1.grid(True, linestyle="-", alpha=0.3)
ax2.grid(True, linestyle="--", alpha=0.3)

# Only show legend for course angles on the left subplot
ax1.legend(loc="lower center", frameon=True)

plt.tight_layout()
plt.savefig(save_folder + "vtau_dot_loyd.pdf")
plt.show()

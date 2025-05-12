import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
from picawe.kinematics.parametrized_patterns import Helix
from picawe import SystemModel, State
from picawe.utils.color_palette import set_plot_style, get_color_list
from picawe.timeseries.phase_parametrized import PhaseParameterized
from picawe.system.kite import Kite
from picawe.system.tether import FlexibleLumpedTether
from picawe.utils.defaults import PLOT_LABELS

# -------------------- Configuration --------------------
file_path = "./data/v3_aero_input.json"
# file_path = "./data/ap2_aero_input.json"
with open(file_path, "r") as file:
    aero_input = json.load(file)

pattern_config = {
    "pattern_type": "helix",
    "initial_parameters": {
        "omega": -1.0,
        "r0": 200.0,
        "d0": 82.0,
        "vr": 0.2,
        "beta": 0.35,
        "kappa": 0
    },
    "optimization_parameters": {
        # Add any optimization-related parameters here if needed as list of names
        "d0",
        # "kappa",
        # "beta",
    }
}
# pattern_config = {
#     "pattern_type": "figure_eight",
#     "initial_parameters": {
#         "omega": -1.0,
#         "r0": 200.0,
#         "ry": 100,
#         "rz": 100,
#         "ky": 0.5,
#         "kz": 0.5,
#         "vr": 1,
#         "beta": 0.45,
#         "kappa": 0
#     },
#     "optimization_parameters": {
#         # Add any optimization-related parameters here if needed as list of names
#         "d0",
#         # "kappa",
#         # "beta",
#     }
# }


start_state = State(
    t=0,
    s=np.pi/2,
    s_dot=2,
    s_ddot=0,
    length_tether=200,
    input_steering=0,
    angle_roll=0,
    angle_pitch=0,
    angle_yaw=0,
)

s_array = np.linspace(np.pi/2, 9*np.pi/2, 400)
colors = get_color_list()
save_folder = "./results/figures/translational_paper/"

# -------------------- Setup and Simulation --------------------
mass_ratio = 2
dof = 3
area_wing = 20
mass_wing = mass_ratio * area_wing

tether = FlexibleLumpedTether()
kite = Kite(mass_wing=mass_wing, area_wing=area_wing, aero_input=aero_input, steering_control="asymmetric")

phases = {}
for quasi_steady in [True, False]:
    model = SystemModel(dof=dof, quasi_steady=quasi_steady, kite=kite, wind_model="uniform", tether=tether)
    model.wind.speed_wind_ref = 9
    model.input_depower = 0
    model.speed_radial = 0

    phase = PhaseParameterized(model, quasi_steady=quasi_steady, pattern_config=pattern_config)
    phase.run_simulation(start_state=start_state, s_array=s_array)

    if quasi_steady:
        start_state = phase.states[0]
        start_state["s_dot"] = phase.return_variable("s_dot")[0]

    phases[quasi_steady] = phase

# -------------------- Plot Layout --------------------
fig = plt.figure(figsize=(12, 6))
gs = fig.add_gridspec(8, 3, width_ratios=[1, 0.25, 2], height_ratios=[1, 1, 1, 1, 1, 1, 1, 1])

ax1 = fig.add_subplot(gs[:4, 0])
ax2 = fig.add_subplot(gs[4:, 0])
ax3 = fig.add_subplot(gs[:2, 2])
ax4 = fig.add_subplot(gs[2:4, 2])
ax5 = fig.add_subplot(gs[4:6, 2])
ax6 = fig.add_subplot(gs[6:, 2])

for i, quasi_steady in enumerate([True, False]):
    linestyle = "--" if quasi_steady else "-"
    label = "Quasi-steady" if quasi_steady else "Dynamic"
    color = colors[i]

    phase = phases[quasi_steady]
    s = np.degrees(phase.return_variable("s") - 5*np.pi/2)
    vtau = phase.return_variable("speed_tangential")
    tension = phase.return_variable("tension_tether_ground") / 1000
    roll = np.degrees(phase.return_variable("input_steering"))
    aoa = np.degrees(phase.return_variable("angle_of_attack"))
    azimuth = np.degrees(phase.return_variable("angle_azimuth"))
    elevation = np.degrees(phase.return_variable("angle_elevation"))

    print(np.sum(phase.return_variable("tension_tether_ground")) / 1000)

    ax3.plot(s, vtau, linestyle=linestyle, color=color, label=label)
    ax4.plot(s, tension, linestyle=linestyle, color=color)
    ax5.plot(s, roll, linestyle=linestyle, color=color)
    ax6.plot(s, aoa, linestyle=linestyle, color=color)

    ax = ax2 if quasi_steady else ax1
    scatter = ax.scatter(azimuth, elevation, c=vtau, cmap="viridis", s=10)

cbar_ax = fig.add_axes([0.35, 0.3, 0.02, 0.4])
cbar = fig.colorbar(scatter, cax=cbar_ax)
cbar.set_label(PLOT_LABELS["speed_tangential"])

ax1.set_ylabel(PLOT_LABELS["angle_elevation"])
ax2.set_xlabel(PLOT_LABELS["angle_azimuth"])
ax2.set_ylabel(PLOT_LABELS["angle_elevation"])
ax3.set_ylabel(PLOT_LABELS["speed_tangential"])
ax4.set_ylabel("Tension [kN]")
ax5.set_ylabel(PLOT_LABELS["angle_roll"])
ax6.set_ylabel(PLOT_LABELS["angle_of_attack"])
ax6.set_xlabel(PLOT_LABELS["phase"])

ax3.set_xlim([0, 360])
ax4.set_xlim([0, 360])
ax5.set_xlim([0, 360])
ax6.set_xlim([0, 360])

ax3.legend()
set_plot_style()
plt.tight_layout()
plt.savefig(save_folder + "parametrized_circle_results_combined.pdf", bbox_inches='tight')
plt.show()

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
from picawe.kinematics.parametrized_patterns import Helix
from picawe import SystemModel, State
from picawe.utils.color_palette import set_plot_style, get_color_list
from picawe.timeseries.phase_parametrized import PhaseParameterized
from picawe.system.kite import Kite
from picawe.system.tether import FlexibleLumpedTether, RigidLumpedTether
from picawe.utils.defaults import PLOT_LABELS

# -------------------- Configuration --------------------
# file_path = "./data/LEI-V3-KITE/v3_aero_input.json"
file_path = "./data/AP2/ap2_aero_input.json"
file_path = "./data/LEI-V9-KITE/v9_aero_input.json"
with open(file_path, "r") as file:
    aero_input = json.load(file)

# aero_input["params"]["angle_pitch_depower_0"] = 0.05
wind_speed = 15  # m/s
pattern_config = {
    "pattern_type": "helix",
    "parameters": {
        "omega": -1.0,
        "r0": 300.0,
        "d0": 110.0,
        "vr": 1.5,
        "beta0": 25 / 180 * np.pi,  # Convert degrees to radians
        "kappa": 1,
        "kbeta": 0,
    },
    "start_path_angle": -np.pi / 2,
    "end_path_angle": 6 * np.pi + np.pi / 2,
    "n_points": 400,
    "optimization_parameters": {
        "d0",
    },
}

# pattern_config = {
#     "pattern_type": "figure_eight",
#     "parameters": {
#         "omega": -1.0,
#         "r0": 200.0,
#         "ry": 70,
#         "rz": 60,
#         "ky": 0.8,
#         "kz": 0.6,
#         "vr": 1,
#         "beta0": 0.35,
#         "kappa": 0,
#     },
#     "start_path_angle": -np.pi / 2,
#     "end_path_angle": 6 * np.pi + np.pi / 2,
#     "n_points": 400,
#     "optimization_parameters": {
#         # Add any optimization-related parameters here if needed as list of names
#         # "ry",
#         # "rz",
#         # "ky",
#         # "kz",
#         "kappa",
#         # "beta",
#         "vr",
#     },
# }

mass_ratio = 10
area_wing = 20
s_array = np.linspace(-np.pi / 2, 2 * 2 * np.pi + np.pi / 2, 800)
save_folder = "./results/figures/"
colors = get_color_list()

# -------------------- Plot Layout --------------------
fig = plt.figure(figsize=(14, 8))
gs = fig.add_gridspec(8, 3, width_ratios=[1, 0.25, 2], height_ratios=[1] * 8)

ax1 = fig.add_subplot(gs[:4, 0])
ax2 = fig.add_subplot(gs[4:, 0])
ax3 = fig.add_subplot(gs[:2, 2])
ax4 = fig.add_subplot(gs[2:4, 2])
ax5 = fig.add_subplot(gs[4:6, 2])
ax6 = fig.add_subplot(gs[6:, 2])

scatter = None
results = {}

mass_wing = mass_ratio * area_wing
tether = RigidLumpedTether(diameter=0.01)
kite = Kite(
    mass_wing=mass_wing,
    area_wing=area_wing,
    aero_input=aero_input,
    steering_control="roll",
)

start_state = State(
    t=0,
    s=np.pi / 2,
    s_dot=2,
    s_ddot=0,
    length_tether=199.6,
    input_steering=0,
    angle_roll=0,
    angle_pitch=0,
    angle_yaw=0,
    tension_tether_ground=1e8,
)

for j, quasi_steady in enumerate([True, False]):
    linestyle = "--" if quasi_steady else "-"
    label = f"{'Quasi-Steady' if quasi_steady else 'Dynamic'}"
    color = colors[j % len(colors)]

    model = SystemModel(dof=3, quasi_steady=quasi_steady, kite=kite)  # , tether=tether)
    model.wind.speed_wind_ref = wind_speed
    model.input_depower = 0
    # model.speed_radial = 0

    phase = PhaseParameterized(
        model, quasi_steady=quasi_steady, pattern_config=pattern_config
    )
    phase.run_simulation(start_state=start_state)

    if quasi_steady:
        start_state = phase.states[0]
        start_state["s_dot"] = phase.return_variable("s_dot")[0]

    s = np.degrees(phase.return_variable("s") - 5 * np.pi / 2)
    vtau = phase.return_variable("speed_tangential")
    tension = phase.return_variable("tension_tether_ground") / 1000
    roll = np.degrees(phase.return_variable("input_steering"))
    aoa = np.degrees(phase.return_variable("angle_of_attack"))
    azimuth = np.degrees(phase.return_variable("angle_azimuth"))
    elevation = np.degrees(phase.return_variable("angle_elevation"))

    results["qs" if quasi_steady else "dyn"] = {
        "s": s,
        "vtau": vtau,
        "tension": tension,
        "vr": phase.return_variable("speed_radial"),
        "t": phase.return_variable("t"),
    }

    ax3.plot(s, vtau, linestyle=linestyle, color=color, label=label)
    ax4.plot(s, tension, linestyle=linestyle, color=color)
    ax5.plot(s, roll, linestyle=linestyle, color=color)
    ax6.plot(s, aoa, linestyle=linestyle, color=color)

    ax = ax2 if quasi_steady else ax1
    scatter = ax.scatter(azimuth, elevation, c=vtau, cmap="viridis", s=10)

# -------------------- Finalize Plot --------------------
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

for a in [ax1, ax2]:
    a.set_ylim(0, 60)
    a.set_xlim(-30, 30)
for a in [ax3, ax4, ax5, ax6]:
    a.set_xlim([0, 360])

ax3.legend()


# -------------------- Difference Analysis --------------------
s_qs = results["qs"]["s"]
s_dyn = results["dyn"]["s"]
mask_qs = s_qs > 0
mask_dyn = s_dyn > 0
# mask_qs = (s_qs > 360) & (s_qs < 360+360)
# mask_dyn = (s_dyn > 360) & (s_dyn < 360+360)
vtau_qs = results["qs"]["vtau"][mask_qs]
vtau_dyn = results["dyn"]["vtau"][mask_dyn]
tension_qs = results["qs"]["tension"][mask_qs]
tension_dyn = results["dyn"]["tension"][mask_dyn]
vr_qs = results["qs"]["vr"][mask_qs]
vr_dyn = results["dyn"]["vr"][mask_dyn]
s_qs = s_qs[mask_qs]
s_dyn = s_dyn[mask_dyn]
t_qs = results["qs"]["t"][mask_qs]
t_dyn = results["dyn"]["t"][mask_dyn]

# Phase difference in maxima and minima
s_max_qs = s_qs[np.argmax(vtau_qs)]
s_max_dyn = s_qs[np.argmax(vtau_dyn)]
s_min_qs = s_dyn[np.argmin(vtau_qs)]
s_min_dyn = s_dyn[np.argmin(vtau_dyn)]

phase_diff_max = s_max_dyn - s_max_qs
phase_diff_min = s_min_dyn - s_min_qs

mean_vtau_diff = (np.max(vtau_dyn) - np.max(vtau_qs)) / np.max(vtau_dyn) * 100

mean_tension_diff = (
    (np.mean(tension_dyn) - np.mean(tension_qs)) / np.mean(tension_dyn) * 100
)

# Calculate the difference in sum(tension * vr*dt)
sum_energy_qs = np.sum(tension_qs * vr_qs * np.diff(t_qs, prepend=t_qs[0]))
sum_energy_dyn = np.sum(tension_dyn * vr_dyn * np.diff(t_dyn, prepend=t_dyn[0]))
sum_force_work_diff = (sum_energy_dyn - sum_energy_qs) / sum_energy_dyn * 100

sum_pow_qs = sum_energy_qs / (t_qs[-1] - t_qs[0])
sum_pow_dyn = sum_energy_dyn / (t_dyn[-1] - t_dyn[0])
print("Power QS:", sum_pow_qs)
print("Power Dyn:", sum_pow_dyn)

print("Phase difference at max vtau:", phase_diff_max)
print("Phase difference at min vtau:", phase_diff_min)
print(
    "Max vtau difference:",
    (np.max(vtau_dyn) - np.max(vtau_qs)) / np.max(vtau_dyn) * 100,
)
print(
    "Max tether tension difference:",
    (np.max(tension_dyn) - np.max(tension_qs)) / np.max(tension_dyn) * 100,
)
print("Mean tether tension difference:", mean_tension_diff)
print("Difference in energy:", sum_force_work_diff)
print("Power difference:", (sum_pow_dyn - sum_pow_qs) / sum_pow_dyn * 100)


# Mark max and min vtau on left-hand plots
def find_angle_at_extreme(phase_obj, vtau, kind="max"):
    idx = np.argmax(vtau) if kind == "max" else np.argmin(vtau)
    az = np.degrees(phase_obj.return_variable("angle_azimuth")[idx])
    el = np.degrees(phase_obj.return_variable("angle_elevation")[idx])
    return az, el


az_max_dyn, el_max_dyn = find_angle_at_extreme(phase, vtau_dyn, kind="max")
az_min_dyn, el_min_dyn = find_angle_at_extreme(phase, vtau_dyn, kind="min")
az_max_qs, el_max_qs = find_angle_at_extreme(phase, vtau_qs, kind="max")
az_min_qs, el_min_qs = find_angle_at_extreme(phase, vtau_qs, kind="min")

ax1.plot(az_max_dyn, el_max_dyn, "ro", label="Max $v_\\tau$")
ax1.plot(az_min_dyn, el_min_dyn, "bo", label="Min $v_\\tau$")
ax2.plot(az_max_qs, el_max_qs, "ro", label="Max $v_\\tau$")
ax2.plot(az_min_qs, el_min_qs, "bo", label="Min $v_\\tau$")

for ax in [ax1, ax2]:
    ax.legend(loc="lower right", fontsize=10)
ax1.text(
    0.95,
    0.95,
    "Dynamic",
    transform=ax1.transAxes,
    ha="right",
    va="top",
    fontsize=12,
    weight="bold",
    bbox=dict(facecolor="white", edgecolor="gray", alpha=0.8),
)
ax2.text(
    0.95,
    0.95,
    "Quasi-Steady",
    transform=ax2.transAxes,
    ha="right",
    va="top",
    fontsize=12,
    weight="bold",
    bbox=dict(facecolor="white", edgecolor="gray", alpha=0.8),
)

set_plot_style()
# plt.tight_layout()
# fig.suptitle(f"Trajectory Comparison for Mass Ratio = {mass_ratio}", fontsize=16)
# plt.savefig(
#     save_folder + "parametrized_circle_results_single_mr.png",
#     bbox_inches="tight",
#     dpi=300,
# )
plt.show()

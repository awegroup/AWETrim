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
file_path = "./data/LEI-V9-KITE/v9_aero_input.json"
with open(file_path, "r") as file:
    aero_input = json.load(file)

wind_speed = 15  # m/s
save_folder = "./results/figures/"
colors = get_color_list()

# Define two pattern configs
pattern_configs = [
    {
        "pattern_type": "helix",
        "parameters": {
            "omega": 1.0,
            "r0": 300.0,
            "d0": 70 * 2,
            "vr": 1.3,
            "beta0": 25 / 180 * np.pi,
            "kappa": 0,
            "kbeta": 0,
        },
        "start_path_angle": -np.pi / 2,
        "end_path_angle": 4 * np.pi + np.pi / 2,
        "n_points": 500,
    },
    {
        "pattern_type": "helix",
        "parameters": {
            "omega": 1,
            "r0": 300.0,
            "d0": 70 * 2,
            "vr": 1.3,
            "beta0": 25 / 180 * np.pi,
            "kappa": 0,
            "kbeta": 0,
        },
        "start_path_angle": -np.pi / 2,
        "end_path_angle": 4 * np.pi + np.pi / 2,
        "n_points": 500,
    },
]

# Define two kites
kite_params = [
    {"mass_ratio": 2, "area_wing": 20},
    {"mass_ratio": 10, "area_wing": 20},
]

# -------------------- Plot Layout --------------------
fig = plt.figure(figsize=(12, 6))
gs = fig.add_gridspec(6, 3, width_ratios=[1, 0.25, 2], height_ratios=[1] * 6)

ax1 = fig.add_subplot(gs[:3, 0])
ax2 = fig.add_subplot(gs[3:, 0])
ax3 = fig.add_subplot(gs[:2, 2])
ax4 = fig.add_subplot(gs[2:4, 2])
ax5 = fig.add_subplot(gs[4:6, 2])

scatter = None
results = []

for i, (kite_param, pattern_config) in enumerate(zip(kite_params, pattern_configs)):
    mass_wing = kite_param["mass_ratio"] * kite_param["area_wing"]
    kite = Kite(
        mass_wing=mass_wing,
        area_wing=kite_param["area_wing"],
        aero_input=aero_input,
        steering_control="roll",
    )
    # tether = RigidLumpedTether(diameter=0.01)
    start_state = State(
        t=0,
        s=-np.pi / 2,
        s_dot=2,
        s_ddot=0,
        length_tether=199.6,
        input_steering=0,
        angle_roll=0,
        angle_pitch=0,
        angle_yaw=0,
        tension_tether_ground=1e8,
    )

    sim_results = {}
    for j, quasi_steady in enumerate([True, False]):
        linestyle = "--" if quasi_steady else "-"
        label = (
            f"{'QS' if quasi_steady else 'Dyn'} ($m/S$ = {kite_param['mass_ratio']})"
        )
        color = colors[i]

        model = SystemModel(dof=3, quasi_steady=quasi_steady, kite=kite)
        model.wind.speed_wind_ref = wind_speed
        model.input_depower = 0

        phase = PhaseParameterized(
            model, quasi_steady=quasi_steady, pattern_config=pattern_config
        )
        phase.run_simulation(start_state=start_state)

        s = np.degrees(phase.return_variable("s")) - 450
        vtau = phase.return_variable("speed_tangential")
        tension = phase.return_variable("tension_tether_ground") / 1000
        azimuth = np.degrees(phase.return_variable("angle_azimuth"))
        elevation = np.degrees(phase.return_variable("angle_elevation"))
        us = phase.return_variable("input_steering")

        sim_results["qs" if quasi_steady else "dyn"] = {
            "s": s,
            "vtau": vtau,
            "tension": tension,
            "azimuth": azimuth,
            "elevation": elevation,
            "phase": phase,
            "vr": phase.return_variable("speed_radial"),
            "t": phase.return_variable("t"),
        }
        if quasi_steady:
            start_state = phase.states[0]
            start_state["s_dot"] = phase.return_variable("s_dot")[0]
        mask_last_circle = s > s[-1] - 360
        # Plot vtau and tension for both kites
        ax3.plot(
            s[mask_last_circle],
            vtau[mask_last_circle],
            linestyle=linestyle,
            color=color,
            label=label,
        )
        ax4.plot(
            s[mask_last_circle],
            tension[mask_last_circle],
            linestyle=linestyle,
            color=color,
        )

        ax5.plot(
            s[mask_last_circle], us[mask_last_circle], linestyle=linestyle, color=color
        )
    # -------------------- Difference Analysis --------------------
    s_qs = sim_results["qs"]["s"]
    s_dyn = sim_results["dyn"]["s"]
    mask_qs = s_qs > 0
    mask_dyn = s_dyn > 0
    # mask_qs = (s_qs > 360) & (s_qs < 360+360)
    # mask_dyn = (s_dyn > 360) & (s_dyn < 360+360)
    vtau_qs = sim_results["qs"]["vtau"][mask_qs]
    vtau_dyn = sim_results["dyn"]["vtau"][mask_dyn]
    tension_qs = sim_results["qs"]["tension"][mask_qs]
    tension_dyn = sim_results["dyn"]["tension"][mask_dyn]
    vr_qs = sim_results["qs"]["vr"][mask_qs]
    vr_dyn = sim_results["dyn"]["vr"][mask_dyn]
    s_qs = s_qs[mask_qs]
    s_dyn = s_dyn[mask_dyn]
    t_qs = sim_results["qs"]["t"][mask_qs]
    t_dyn = sim_results["dyn"]["t"][mask_dyn]

    # Phase difference in maxima and minima
    s_max_qs = s_qs[np.argmax(vtau_qs)]
    s_max_dyn = s_dyn[np.argmax(vtau_dyn)]
    s_min_qs = s_qs[np.argmin(vtau_qs)]
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
    results.append(sim_results)

    el_dyn = sim_results["dyn"]["elevation"][mask_dyn]
    az_dyn = sim_results["dyn"]["azimuth"][mask_dyn]
    el_qs = sim_results["qs"]["elevation"][mask_qs]
    az_qs = sim_results["qs"]["azimuth"][mask_qs]
    ax = ax1 if i == 0 else ax2
    # Ensure consistent color range for all scatter plots
    vmin = 35  # Set to your expected minimum vtau value
    vmax = 75  # Set to your expected maximum vtau value

    scatter = ax.scatter(
        az_dyn,
        el_dyn,
        c=vtau_dyn,
        cmap="viridis",
        s=10,
        vmin=vmin,
        vmax=vmax,
    )

    def find_angle_at_extreme(el, az, vtau, kind="max"):
        idx = np.argmax(vtau) if kind == "max" else np.argmin(vtau)
        az = az[idx]
        el = el[idx]
        return az, el

    az_max_dyn, el_max_dyn = find_angle_at_extreme(el_dyn, az_dyn, vtau_dyn, kind="max")
    az_min_dyn, el_min_dyn = find_angle_at_extreme(el_dyn, az_dyn, vtau_dyn, kind="min")
    az_max_qs, el_max_qs = find_angle_at_extreme(el_qs, az_qs, vtau_qs, kind="max")
    az_min_qs, el_min_qs = find_angle_at_extreme(el_qs, az_qs, vtau_qs, kind="min")

    ax.plot(
        az_max_dyn,
        el_max_dyn,
        "o",
        color=colors[i],
        markerfacecolor="none",
        label="Max $v_\\tau$ (Dyn)",
    )
    ax.plot(
        az_min_dyn,
        el_min_dyn,
        "o",
        color=colors[i],
        markerfacecolor=colors[i],
        label="Min $v_\\tau$ (Dyn)",
    )
    ax.plot(
        az_max_qs,
        el_max_qs,
        "s",
        color=colors[i],
        markerfacecolor="none",
        label="Max $v_\\tau$ (QS)",
    )
    ax.plot(
        az_min_qs,
        el_min_qs,
        "s",
        color=colors[i],
        markerfacecolor=colors[i],
        label="Min $v_\\tau$ (QS)",
    )

for a in [ax1, ax2]:
    a.set_ylim(10, 40)
    a.set_xlim(-20, 20)
for a in [ax3, ax4, ax5]:
    a.set_xlim([0, 360])

ax1.legend(frameon=False)
ax3.legend(frameon=False)

cbar_ax = fig.add_axes([0.35, 0.3, 0.02, 0.4])
cbar = fig.colorbar(scatter, cax=cbar_ax)
cbar.set_label(PLOT_LABELS["speed_tangential"])

ax1.set_ylabel(PLOT_LABELS["angle_elevation"])
ax2.set_xlabel(PLOT_LABELS["angle_azimuth"])
ax2.set_ylabel(PLOT_LABELS["angle_elevation"])
ax3.set_ylabel(PLOT_LABELS["speed_tangential"])
ax4.set_ylabel("Tension [kN]")
ax5.set_ylabel(PLOT_LABELS["angle_roll"])

ax1.text(
    0.95,
    0.95,
    "$m/S = 2$",
    transform=ax1.transAxes,
    ha="right",
    va="top",
    fontsize=12,
    bbox=dict(facecolor="white", edgecolor="none", alpha=0),
)
ax2.text(
    0.95,
    0.95,
    "$m/S = 10$",
    transform=ax2.transAxes,
    ha="right",
    va="top",
    fontsize=12,
    bbox=dict(facecolor="white", edgecolor="none", alpha=0),
)

set_plot_style()
plt.tight_layout()

# Save the figure
fig.savefig(save_folder + "circular_motion_analysis.pdf", bbox_inches="tight")
plt.show()

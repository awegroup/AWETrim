import numpy as np
import matplotlib.pyplot as plt
import json
from scipy.interpolate import interp1d
from picawe.kinematics.parametrized_patterns import Helix
from picawe import SystemModel, State
from picawe.utils.color_palette import set_plot_style, get_color_list, custom_cmap
from picawe.timeseries.phase_parametrized import PhaseParameterized
from picawe.system.kite import Kite
from picawe.system.tether import RigidLumpedTether
from picawe.utils.defaults import PLOT_LABELS

# ---------- Config ----------
wind_speed = 15  # m/s
colors = get_color_list()

# ---------- Paths ----------
with open("./data/Megawes/100kw_aero_input.json", "r") as file:
    aero_input_100kw = json.load(file)
with open("./data/AP2/ap2_aero_input.json", "r") as file:
    aero_input_ap2 = json.load(file)

# ---------- Pattern configs ----------
mass_100kw = 443.67
area_100kw = 15.44
tether_diameter_100kw = 0.01
pattern_config_100kw = {
    "pattern_type": "helix",
    "parameters": {
        "omega": -1.0,
        "r0": 600.0,
        "d0": 245.0,
        "vr": 3.14,
        "beta0": 25 / 180 * np.pi,  # Convert degrees to radians
        "kappa": 1,
        "kbeta": 0,
    },
    "start_path_angle": -np.pi / 2,
    "end_path_angle": 2 * np.pi + np.pi / 2,
    "n_points": 600,
    "optimization_parameters": {
        "d0",
    },
}
mass_ap2 = 35
area_ap2 = 3
tether_diameter_ap2 = 0.0025
pattern_config_ap2 = {
    "pattern_type": "helix",
    "parameters": {
        "omega": -1.0,
        "r0": 400.0,
        "d0": 145.0,
        "vr": 3.6,
        "beta0": 25 / 180 * np.pi,  # Convert degrees to radians
        "kappa": 1,
        "kbeta": 0,
    },
    "start_path_angle": -np.pi / 2,
    "end_path_angle": 2 * np.pi + np.pi / 2,
    "n_points": 600,
    "optimization_parameters": {
        "d0",
    },
}
# ---------- Starting state ----------
base_start_state = State(
    t=0,
    s=np.pi / 2,
    s_dot=5,
    s_ddot=0,
    length_tether=199.6,
    input_steering=0,
    angle_roll=0,
    angle_pitch=0,
    angle_yaw=0,
    tension_tether_ground=1e8,
)

# ---------- Plot layout ----------
fig = plt.figure(figsize=(14, 8))
gs = fig.add_gridspec(8, 3, width_ratios=[1, 0.25, 2], height_ratios=[1] * 8)
ax1 = fig.add_subplot(gs[:4, 0])
ax2 = fig.add_subplot(gs[4:, 0])
ax3 = fig.add_subplot(gs[:2, 2])
ax4 = fig.add_subplot(gs[2:4, 2])
ax5 = fig.add_subplot(gs[4:6, 2])
ax6 = fig.add_subplot(gs[6:, 2])


def run_sim(
    aero_input,
    pattern_config,
    label_prefix,
    mass_wing,
    area_wing,
    tether_diameter,
    color_base,
    marker="o",
):
    result = {}
    start_state = base_start_state
    for j, quasi_steady in enumerate([True, False]):
        label = f"{label_prefix} {'QS' if quasi_steady else 'Dyn'}"
        linestyle = "--" if quasi_steady else "-"
        color = colors[color_base]
        tether = RigidLumpedTether(
            diameter=tether_diameter,
        )
        kite = Kite(
            mass_wing=mass_wing,
            area_wing=area_wing,
            aero_input=aero_input,
            steering_control="roll",
        )
        model = SystemModel(dof=3, quasi_steady=quasi_steady, kite=kite, tether=tether)
        model.wind.speed_wind_ref = wind_speed
        model.input_depower = 0

        phase = PhaseParameterized(
            model, quasi_steady=quasi_steady, pattern_config=pattern_config
        )
        phase.run_simulation(start_state=start_state)

        if quasi_steady:
            start_state = phase.states[0]
            start_state["s_dot"] = phase.return_variable("s_dot")[0]

        s = np.degrees(phase.return_variable("s") - np.pi / 2)
        result["qs" if quasi_steady else "dyn"] = {
            "s": s,
            "vtau": phase.return_variable("speed_tangential"),
            "tension": phase.return_variable("tension_tether_ground") / 1000,
            "roll": np.degrees(phase.return_variable("input_steering")),
            "input_steering": phase.return_variable("input_steering"),
            "aoa": np.degrees(phase.return_variable("angle_of_attack")),
            "az": np.degrees(phase.return_variable("angle_azimuth")),
            "el": np.degrees(phase.return_variable("angle_elevation")),
            "vr": phase.return_variable("speed_radial"),
            "t": phase.return_variable("t"),
            "phase": phase,
        }

        ax = ax2 if quasi_steady else ax1
        scatter = ax.scatter(
            result["qs" if quasi_steady else "dyn"]["az"],
            result["qs" if quasi_steady else "dyn"]["el"],
            c=result["qs" if quasi_steady else "dyn"]["vtau"],
            cmap=custom_cmap,
            s=10,
            vmin=30,
            vmax=110,
        )

        ax3.plot(
            s,
            result["qs" if quasi_steady else "dyn"]["vtau"],
            linestyle=linestyle,
            color=color,
            label=label,
        )
        ax4.plot(
            s,
            result["qs" if quasi_steady else "dyn"]["tension"],
            linestyle=linestyle,
            color=color,
        )
        ax5.plot(
            s,
            result["qs" if quasi_steady else "dyn"]["roll"],
            linestyle=linestyle,
            color=color,
        )
        ax6.plot(
            s,
            result["qs" if quasi_steady else "dyn"]["aoa"],
            linestyle=linestyle,
            color=color,
        )
    for i, quasi_steady in enumerate([True, False]):
        s = result["qs" if quasi_steady else "dyn"]["s"]
        vtau = result["qs" if quasi_steady else "dyn"]["vtau"][s > 0]
        max_speed_idx = np.argmax(vtau)
        min_speed_idx = np.argmin(vtau)
        # Plot into ax a point at the maximum and minimum speed
        ax = ax2 if quasi_steady else ax1
        ax.plot(
            result["qs" if quasi_steady else "dyn"]["az"][s > 0][max_speed_idx],
            result["qs" if quasi_steady else "dyn"]["el"][s > 0][max_speed_idx],
            marker,
            color=colors[6],
            label=f"{label_prefix} Max Speed",
        )
        ax.plot(
            result["qs" if quasi_steady else "dyn"]["az"][s > 0][min_speed_idx],
            result["qs" if quasi_steady else "dyn"]["el"][s > 0][min_speed_idx],
            marker,
            color=colors[5],
            label=f"{label_prefix} Min Speed",
        )
    return result, scatter


results_100kw, scatter_100kw = run_sim(
    aero_input_100kw,
    pattern_config_100kw,
    "100kW",
    mass_100kw,
    area_100kw,
    tether_diameter_100kw,
    1,
)
results_ap2, scatter_ap2 = run_sim(
    aero_input_ap2,
    pattern_config_ap2,
    "AP2",
    mass_ap2,
    area_ap2,
    tether_diameter_ap2,
    2,
    marker="^",
)

# ---------- Final plot formatting ----------
cbar_ax = fig.add_axes([0.35, 0.3, 0.02, 0.4])
cbar = fig.colorbar(scatter_ap2, cax=cbar_ax)
cbar.set_label(PLOT_LABELS["speed_tangential"])

for ax in [ax1, ax2]:
    ax.set_ylim(10, 40)
    ax.set_xlim(-20, 20)
    ax.legend(loc="lower right", fontsize=9)

# Add text labels to identify Dynamic and Quasi-Steady plots
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

for ax in [ax3, ax4, ax5, ax6]:
    ax.set_xlim(0, 360)

ax1.set_ylabel(PLOT_LABELS["angle_elevation"])
ax2.set_xlabel(PLOT_LABELS["angle_azimuth"])
ax2.set_ylabel(PLOT_LABELS["angle_elevation"])
ax3.set_ylabel(PLOT_LABELS["speed_tangential"])
ax4.set_ylabel("Tension [kN]")
ax5.set_ylabel(PLOT_LABELS["angle_roll"])
ax6.set_ylabel(PLOT_LABELS["angle_of_attack"])
ax6.set_xlabel(PLOT_LABELS["phase"])
ax3.legend()

set_plot_style()
plt.tight_layout()
# Save the figure as pdf
plt.savefig(
    "./results/figures/translational_paper/comparison_ap2_megawes100kw.pdf",
    bbox_inches="tight",
)
plt.show()


# ---------- Energy, power and phase comparison ----------
def compute_energy_metrics(results, label=""):
    s_qs = results["qs"]["s"]
    s_dyn = results["dyn"]["s"]
    mask_qs = s_qs > 0
    mask_dyn = s_dyn > 0
    vtau_qs = results["qs"]["vtau"][mask_qs]
    vtau_dyn = results["dyn"]["vtau"][mask_dyn]
    tension_qs = results["qs"]["tension"][mask_qs]
    tension_dyn = results["dyn"]["tension"][mask_dyn]
    vr_qs = results["qs"]["vr"][mask_qs]
    vr_dyn = results["dyn"]["vr"][mask_dyn]
    t_qs = results["qs"]["t"][mask_qs]
    t_dyn = results["dyn"]["t"][mask_dyn]

    sum_energy_qs = np.sum(tension_qs * vr_qs * np.diff(t_qs, prepend=t_qs[0]))
    sum_energy_dyn = np.sum(tension_dyn * vr_dyn * np.diff(t_dyn, prepend=t_dyn[0]))
    sum_pow_qs = sum_energy_qs / (t_qs[-1] - t_qs[0])
    sum_pow_dyn = sum_energy_dyn / (t_dyn[-1] - t_dyn[0])
    power_diff = (sum_pow_dyn - sum_pow_qs) / sum_pow_dyn * 100

    print(f"\n--- {label} ---")
    print(f"Power QS: {sum_pow_qs:.2f}, Power Dyn: {sum_pow_dyn:.2f}")
    print(f"Δ Power: {power_diff:.2f}%")

    # Cross-correlation
    t_common = np.linspace(max(t_qs[0], t_dyn[0]), min(t_qs[-1], t_dyn[-1]), 1000)
    v1 = interp1d(t_qs, vtau_qs, kind="linear")(t_common) - np.mean(vtau_qs)
    v2 = interp1d(t_dyn, vtau_dyn, kind="linear")(t_common) - np.mean(vtau_dyn)
    corr = np.correlate(v1, v2, mode="full")
    lags = np.arange(-len(v1) + 1, len(v1))
    time_lags = lags * (t_common[1] - t_common[0])
    best_lag = time_lags[np.argmax(corr)]
    print(f"Estimated time lag: {best_lag:.3f} s")


compute_energy_metrics(results_100kw, "100kW")
compute_energy_metrics(results_ap2, "AP2")

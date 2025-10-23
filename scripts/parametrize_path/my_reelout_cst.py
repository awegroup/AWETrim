import numpy as np
import json
import matplotlib.pyplot as plt
from awetrim import SystemModel, State
from awetrim.timeseries.phase_parametrized import PhaseParameterized
from awetrim.system.kite import Kite
from awetrim.system.tether import RigidLumpedTether
from awetrim.environment.Wind import Wind
import pickle
from awetrim.utils.color_palette import set_plot_style, get_color_list
from awetrim.utils.defaults import PLOT_LABELS
from my_reel_in import init_conditions_QS as Single_Spline_final_state_QS
from my_reel_in import init_conditions_Dyn as Single_Spline_final_state_Dyn
from awetrim.kinematics.find_Lissajous_RO_start_end_angles import (
    find_Lissajous_RO_start_end_angles,
)

# ---------- Config ----------
mass_wing = 61
mass_kcu = 30
area_wing = 46.85
tether_diameter = 0.01

speed_wind_at_100 = (
    7.6374  # m/s (6 m/s at reference height of 6 m) got from KP software for LOG
)
wind = Wind(
    wind_model="logarithmic",
    z0=0.0002,
)
speed_friction = 0.41 * speed_wind_at_100 / np.log(100 / wind.z0)
wind.speed_friction = speed_friction
# wind.speed_wind_ref = speed_wind_at_100

# color palette available via get_color_list() as needed

with open("./data/LEI-V9-KITE/v9_aero_input.json", "r") as file:
    aero_input_v9 = json.load(file)

# ---------- Load precomputed Lissajous fit data ----------
segment_name = "LISSAJOUS"

filename = f"fit_results_{segment_name}.pkl"
with open(filename, "rb") as f:
    fit_data = pickle.load(f)

r0 = fit_data["r0"]
duration = fit_data["duration"]
az_amp0 = fit_data["best_params"]["az_amp0"]
beta_amp0 = fit_data["best_params"]["beta_amp0"]
beta_coeffs = fit_data["best_params"]["beta_coeffs"]
az_coeffs = fit_data["best_params"]["az_coeffs"]
beta0 = fit_data["best_params"]["beta0"]

pattern_type = "cst_lissajous"
parameters = {
    "omega": 1.0,
    "r0": r0,
    "az_amp0": az_amp0,
    "beta_amp0": beta_amp0,
    "width_phi": 0.5,
    "width_beta": 0.5,
    "left_first": True,
    "normalize_bumps": False,
    "repeat_phi": True,
    "repeat_beta": True,
    "beta_coeffs": np.array(beta_coeffs),
    "az_coeffs": az_coeffs,
    "kbeta": 0,
    "beta0": beta0,
    "kappa": 0,
}

s_start_opt, range_opt, cycles = find_Lissajous_RO_start_end_angles(
    pattern_type, parameters
)
# cycles is by default 1

# ---------Load winch and depower data ----------

with open("fit_winch_results_RO_phase_settings.pkl", "rb") as f:
    winch_depower_data = pickle.load(f)

f_max = winch_depower_data[0]["max_tether_force"]
f_min = winch_depower_data[0]["min_tether_force"]
beta_plus = winch_depower_data[0]["softplus_beta"]
beta_minus = winch_depower_data[0]["softminus_beta"]
slope = winch_depower_data[0]["slope"]
offset = winch_depower_data[0]["offset"]
# s_start = winch_depower_data[0]["s"]
depower = winch_depower_data[0]["depower"]
# s_end = 2*np.pi

depower_norm = (
    (depower / 100) - 0.4
) / 0.28  # normalize depower between 0 and 1 for V9

Realistic_RO_eg = {
    "reeling_strategy": "force",  # "force" or "constant"
    "force_model": "quadratic",  # "linear" or "quadratic"
    "reeling_speed": 0,  # m/s, only for constant reeling
    "max_tether_force": f_max,  # N, only for force reeling
    "min_tether_force": f_min,  # N, only for force reeling
    "softplus": True,
    "softplus_beta": beta_plus,
    "softminus": True,
    "softminus_beta": beta_minus,
    "slope": slope,  # N/(m/s)^2 for quadratic, N/(m/s) for linear
    "offset": offset,  # m/s
}

pattern_config = {
    "pattern_type": pattern_type,
    "path_parameters": parameters,
    "radial_parameters": Realistic_RO_eg,
    "start_time": 0,
    "end_time": duration + 1,
    "start_angle": s_start_opt,
    "end_angle": s_start_opt + range_opt + cycles * (2 * np.pi),
    "n_points": 500,
    "optimization_parameters": [],
}

# ---------- Starting state ----------
Single_Spline_final_state_QS["s"] = s_start_opt
Single_Spline_final_state_QS["s_dot"] = 2
Single_Spline_final_state_QS["tension_tether_ground"] = 1e8

print("\n")
for key in Single_Spline_final_state_QS.keys():
    print(f"{key}:              {Single_Spline_final_state_QS[key]}")
print("\n")

base_start_state_QS = State(**Single_Spline_final_state_QS)


def run_sim(
    aero_input,
    pattern_config,
    label_prefix,
    mass_wing,
    area_wing,
    mass_kcu,
    tether_diameter,
    depower,
    start_state,
    wind,
    sim_type,
):

    if sim_type == "quasi_steady":
        quasi_steady = True
        inertia_free = False
    elif sim_type == "dynamic":
        quasi_steady = False
        inertia_free = False
    elif sim_type == "inertia_free":
        quasi_steady = True
        inertia_free = True
    elif sim_type == "no_mass":
        quasi_steady = True
        inertia_free = True

    print(f"Running simulation for {sim_type} with label: {label_prefix}")
    tether = RigidLumpedTether(
        diameter=tether_diameter,
    )
    kite = Kite(
        mass_wing=mass_wing,
        mass_kcu=mass_kcu,
        area_wing=area_wing,
        aero_input=aero_input,
        steering_control="asymmetric",
    )
    if inertia_free:
        kite.override_centripetal = True
        kite.override_coriolis = True

    model = SystemModel(
        dof=3, quasi_steady=quasi_steady, kite=kite, tether=tether, wind_model=wind
    )

    model.input_depower = depower
    if sim_type == "no_mass":
        model.mass_wing = 0
        start_state["input_steering"] = 0
    phase = PhaseParameterized(
        model, quasi_steady=quasi_steady, pattern_config=pattern_config
    )
    states = phase.run_simulation_phase(start_state=start_state, return_states=True)

    return phase, states


phaseQS, stateQS = run_sim(
    aero_input_v9,
    pattern_config,
    "V9",
    mass_wing,
    area_wing,
    mass_kcu,
    tether_diameter,
    depower_norm,
    base_start_state_QS,
    wind,
    "quasi_steady",
)
base_start_state_Dyn = stateQS[0]
phaseDyn, stateDyn = run_sim(
    aero_input_v9,
    pattern_config,
    "V9",
    mass_wing,
    area_wing,
    mass_kcu,
    tether_diameter,
    depower_norm,
    base_start_state_Dyn,
    wind,
    "dynamic",
)

QS_tension = []
for i in stateQS:
    QS_tension.append(i["tension_tether_ground"])

Dyn_tension = []
for i in stateDyn:
    Dyn_tension.append(i["tension_tether_ground"])

plt.figure()
plt.plot(QS_tension, label="Quasi-Steady")
plt.plot(Dyn_tension, label="Dynamic")
plt.legend()
plt.show()

dynamic_phase = phaseDyn
qs_phase = phaseQS

# First series creates the overview figure
fig, axes_map, scatter = dynamic_phase.plot_overview_3d(
    label="V9 Dynamic",
    color=get_color_list()[2],
    linestyle="-",
    variables=[
        "speed_tangential",
        "tension_tether_ground",
        "input_steering",
        "speed_radial",
    ],
    x_param="t",
)

# Second series overlays on the same axes
qs_phase.plot_overview_3d(
    label="V9 Quasi-Steady",
    color=get_color_list()[1],
    linestyle="--",
    variables=[
        "speed_tangential",
        "tension_tether_ground",
        "input_steering",
        "speed_radial",
    ],
    x_param="t",
    axes=axes_map,
)

fig.legend(loc="upper center", bbox_to_anchor=(0.5, 0.95), ncol=2)
set_plot_style()
plt.tight_layout()
# # Save the figure as pdf
# plt.savefig("./results/figures/reelout_cst.pdf", bbox_inches="tight")
plt.show()


metrics = dynamic_phase.energy_metrics(qs_phase)
print("\n--- V9 ---")
print(
    f"Power QS: {metrics['avg_power_other']:.2f}, Power Dyn: {metrics['avg_power_self']:.2f}."
)
print(
    f"Mean power QS: {metrics['mean_power_other']:.2f}, Mean power Dyn: {metrics['mean_power_self']:.2f}"
)
print(f"Δ Power: {metrics['power_diff_percent']:.2f}%")
print(f"Estimated time lag: {metrics['best_time_lag']:.3f} s")
print(f"ΔF_t,mean: {metrics['delta_ft_mean_percent']:.2f}%")
print(f"ΔF_t,max: {metrics['delta_ft_max_percent']:.2f}%")
print(f"ΔF_t,min: {metrics['delta_ft_min_percent']:.2f}%")
print(f"Δv_tau,max: {metrics['delta_vtau_max_percent']:.2f}%")
print(f"Δv_tau,min: {metrics['delta_vtau_min_percent']:.2f}%")
print(f"Δs_v_tau,max: {metrics['s_lag_vtau_max_deg']:.2f} deg")
print(f"Δs_v_tau,min: {metrics['s_lag_vtau_min_deg']:.2f} deg")
plt.show()

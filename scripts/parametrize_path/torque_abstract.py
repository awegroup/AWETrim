# uniform{
#     "omega": 1.0,
#     "r0": 200.0,
#     "az_amp0": 0.8726646359971639,
#     "beta_amp0": 0.25,
#     "width_phi": 0.5,
#     "width_beta": 0.5,
#     "left_first": True,
#     "normalize_bumps": False,
#     "repeat_phi": True,
#     "repeat_beta": True,
#     "beta_coeffs": array(
#         [0.82913554, -1.00000001, -0.04212424, -0.62406752, -1.00000001]
#     ),
#     "az_coeffs": [0, 0, 0, 0, 0],|
#     "kbeta": 0,
#     "beta0": 0.6,
#     "kappa": 0,
#     "k_vr": 2240,
# }
# z0 = 0.0002{
#     "omega": 1.0,
#     "r0": 200.0,
#     "az_amp0": 0.8726646359951026,
#     "beta_amp0": 0.25,
#     "width_phi": 0.5,
#     "width_beta": 0.5,
#     "left_first": True,
#     "normalize_bumps": False,
#     "repeat_phi": True,
#     "repeat_beta": True,
#     "beta_coeffs": array(
#         [0.5391955, -0.99999893, 0.25072315, -0.99931578, -0.36642241]
#     ),
#     "az_coeffs": [0, 0, 0, 0, 0],
#     "kbeta": 0,
#     "beta0": 0.6,
#     "kappa": 0,
#     "k_vr": 2240,
# }
# z0 = 0.1{'omega': 1.0, 'r0': 200.0, 'az_amp0': 0.8714565216364403, 'beta_amp0': 0.25, 'width_phi': 0.5, 'width_beta': 0.5, 'left_first': True, 'normalize_bumps': False, 'repeat_phi': True, 'repeat_beta': True, 'beta_coeffs': array([-0.21894066, -0.84928545,  0.52667231, -0.99983684,  0.59125753]), 'az_coeffs': [0, 0, 0, 0, 0], 'kbeta': 0, 'beta0': 0.6, 'kappa': 0, 'k_vr': 2240}

# z0 = 0.25{'omega': 1.0, 'r0': 200.0, 'az_amp0': 0.8717771192334758, 'beta_amp0': 0.25, 'width_phi': 0.5, 'width_beta': 0.5, 'left_first': True, 'normalize_bumps': False, 'repeat_phi': True, 'repeat_beta': True, 'beta_coeffs': array([ 0.73619873, -0.99999999,  0.63068818, -0.94615698,  0.05227672]), 'az_coeffs': [0, 0, 0, 0, 0], 'kbeta': 0, 'beta0': 0.6, 'kappa': 0, 'k_vr': 2240}

import json
from picawe.utils.color_palette import set_plot_style, get_color_list, custom_cmap
import matplotlib.pyplot as plt
from picawe.system.tether import RigidLumpedTether
from picawe.system.kite import Kite
from picawe.system import SystemModel, State
import numpy as np
from picawe.environment import Wind
from picawe.timeseries.phase_parametrized import PhaseParameterized

mass_wing = 90
area_wing = 47
tether_diameter = 0.01
with open("./data/LEI-V9-KITE/v9_aero_input.json", "r") as file:
    aero_input = json.load(file)
tether = RigidLumpedTether(
    diameter=tether_diameter,
)
kite = Kite(
    mass_wing=mass_wing,
    area_wing=area_wing,
    aero_input=aero_input,
    steering_control="asymmetric",
)
base_start_state = State(
    t=0,
    s=np.pi / 2,
    s_dot=2,
    input_steering=0,
    tension_tether_ground=1e8,
    distance_radial=230,
    speed_radial=2,
)


# -----------------------------------------------

# --- Define your cases (pattern_config per wind profile) ---
pattern_cases = {
    "uniform": {
        "omega": 1.0,
        "r0": 230.0,
        "az_amp0": 0.48777050522375176,
        "beta_amp0": 0.25,
        "width_phi": 0.5,
        "width_beta": 0.5,
        "left_first": True,
        "normalize_bumps": False,
        "repeat_phi": True,
        "repeat_beta": True,
        "beta_coeffs": np.array(
            [0.00793318, -1.00000001, -0.09528962, -1.00000001, -0.43694973]
        ),
        "az_coeffs": [0, 0, 0, 0, 0],
        "kbeta": 0,
        "beta0": 0.6936021090152621,
        "kappa": 0,
        "k_vr": 2716,
    },
    0.0002: {
        "omega": 1.0,
        "r0": 200.0,
        "az_amp0": 0.4940902360596809,
        "beta_amp0": 0.25,
        "width_phi": 0.5,
        "width_beta": 0.5,
        "left_first": True,
        "normalize_bumps": False,
        "repeat_phi": True,
        "repeat_beta": True,
        "beta_coeffs": np.array([-0.15163525, -1.0, -0.18504036, -1.0, -0.38562817]),
        "az_coeffs": [0, 0, 0, 0, 0],
        "kbeta": 0,
        "beta0": 0.7547877603670499,
        "kappa": 0,
        "k_vr": 2716,
    },
    0.1: {
        "omega": 1.0,
        "r0": 200.0,
        "az_amp0": 0.489640280042086,
        "beta_amp0": 0.25,
        "width_phi": 0.5,
        "width_beta": 0.5,
        "left_first": True,
        "normalize_bumps": False,
        "repeat_phi": True,
        "repeat_beta": True,
        "beta_coeffs": np.array(
            [-0.08900137, -1.00000001, -0.13256639, -1.00000001, -0.38476704]
        ),
        "az_coeffs": [0, 0, 0, 0, 0],
        "kbeta": 0,
        "beta0": 0.8153751785032307,
        "kappa": 0,
        "k_vr": 2716,
    },
    0.25: {
        "omega": 1.0,
        "r0": 200.0,
        "az_amp0": 0.48654764986996457,
        "beta_amp0": 0.25,
        "width_phi": 0.5,
        "width_beta": 0.5,
        "left_first": True,
        "normalize_bumps": False,
        "repeat_phi": True,
        "repeat_beta": True,
        "beta_coeffs": np.array(
            [-0.04715123, -1.00000001, -0.10246252, -1.00000001, -0.38830766]
        ),
        "az_coeffs": [0, 0, 0, 0, 0],
        "kbeta": 0,
        "beta0": 0.8333553153994843,
        "kappa": 0,
        "k_vr": 2716,
    },
}

default_params = {
    "omega": 1.0,
    "r0": 200.0,
    "az_amp0": 0.6191628957199365,
    "beta_amp0": 0.17039404884183,
    "width_phi": 0.5,
    "width_beta": 0.5,
    "left_first": True,
    "normalize_bumps": False,
    "repeat_phi": True,
    "repeat_beta": True,
    "beta_coeffs": np.array([0, 0, 0, 0, 0]),
    "az_coeffs": [0, 0, 0, 0, 0],
    "kbeta": 0,
    "beta0": 0.3844981096025447,
    "kappa": 0,
    "k_vr": 2716,
}
# --- Choose your wind magnitude reference ---
# If you want 12 m/s at 100 m, set:
speed_wind_at_100 = 12.0  # change as needed


def make_wind(case_key):
    """Return a Wind() configured as uniform or logarithmic depending on case_key."""
    if case_key == "uniform":
        w = Wind(wind_model="uniform")
        # For uniform, set the reference speed directly (your comment hints at this):
        w.speed_wind_ref = speed_wind_at_100
        return w
    else:
        z0 = float(case_key)
        w = Wind(wind_model="logarithmic", z0=z0)
        # Match your original friction-velocity calibration:
        # u_* = kappa * U(z_ref) / ln(z_ref / z0) with z_ref = 100 m
        w.speed_friction = 0.41 * speed_wind_at_100 / np.log(100.0 / w.z0)
        return w


def make_pattern_config(parameters):
    """Return the pattern_config for the given case_key."""
    pattern_config = {
        "pattern_type": "cst_lissajous",
        "parameters": parameters,
        "start_time": 0,
        "end_time": 60,
        "start_angle": np.pi / 2,
        "end_angle": 2 * np.pi + np.pi / 2,
        "n_points": 600,
        "optimization_parameters": [],
    }
    return pattern_config


results = {}  # store whatever PhaseParameterized returns (or attributes you need)

for case_key, parameters in pattern_cases.items():
    # 1) Wind for this case
    wind = make_wind(case_key)

    # 2) Model
    model = SystemModel(
        dof=3, quasi_steady=True, kite=kite, tether=tether, wind_model=wind
    )
    model.input_depower = 0

    # 3) Phase runner
    phase = PhaseParameterized(
        model, quasi_steady=True, pattern_config=make_pattern_config(parameters)
    )
    wind = make_wind(case_key)
    model = SystemModel(
        dof=3, quasi_steady=True, kite=kite, tether=tether, wind_model=wind
    )
    model.input_depower = 0
    phase_default = PhaseParameterized(
        model, quasi_steady=True, pattern_config=make_pattern_config(default_params)
    )

    # 4) Start state (use the one you defined)
    start_state = base_start_state

    # 5) Run
    phase.run_simulation_phase(start_state=start_state)
    start_state = base_start_state
    phase_default.run_simulation_phase(start_state=start_state)

    s = phase.return_variable("s")
    s_default = phase_default.return_variable("s")
    power = phase.return_variable("mechanical_power")
    power_default = phase_default.return_variable("mechanical_power")
    power = np.mean(power[s < s[0] + 2 * np.pi])
    power_default = np.mean(power_default[s_default < s_default[0] + 2 * np.pi])
    print(f"Case {case_key}: Power = {power:.1f} W, Default = {power_default:.1f} W")
    print(f"  Gain = {100*(power/power_default - 1):.1f} %")
    # plt.plot(np.diff(phase.return_variable("speed_radial")))
    # plt.show()
    # 6) Keep anything useful (change the key if you prefer labels like "z0=0.1")
    results[case_key] = phase

set_plot_style()
fig, axs = plt.subplots(1, 2, figsize=(10, 4))
heights_wind = np.linspace(1, 300, 100)
colors = get_color_list()
i = -1
for case_key, sim_out in results.items():
    i += 1
    wind = sim_out.kite_model.wind
    heights_flown = sim_out.return_variable("z")
    max_height = np.max(heights_flown)
    min_height = np.min(heights_flown)
    if wind.wind_model == "uniform":

        wind_speeds = wind.speed_wind_ref * np.ones_like(heights_wind)

        label = "Uniform"
    else:
        wind_speeds = np.array(wind.speed_wind_at_height(heights_wind))
        label = f"log(z0={wind.z0})"
    axs[0].plot(
        np.degrees(sim_out.return_variable("angle_azimuth")),
        np.degrees(sim_out.return_variable("angle_elevation")),
        # label=label,
    )

    axs[1].plot(wind_speeds, heights_wind, label=label, color=colors[i])
    mask = (heights_wind >= min_height) & (heights_wind <= max_height)
    axs[1].plot(
        wind_speeds[mask],
        heights_wind[mask],
        color=colors[i],
        linewidth=4,
        alpha=0.3,
        # label=label,
    )

axs[0].plot(
    np.degrees(phase_default.return_variable("angle_azimuth")),
    np.degrees(phase_default.return_variable("angle_elevation")),
    linestyle="--",
    color="black",
    alpha=0.7,
    label="Baseline pattern",
)

axs[1].set_xlabel("Wind Speed (m/s)")
axs[1].set_ylabel("Height (m)")

axs[0].set_xlabel("Azimuth (deg)")
axs[0].set_ylabel("Elevation (deg)")
axs[0].set_ylim([0, 40])
axs[0].set_xlim([-40, 40])
axs[1].set_ylim([0, 300])
axs[1].set_xlim([0, 15])
axs[1].legend()
axs[0].legend()
plt.tight_layout()
plt.savefig("v9_parametrization_cases.pdf")
plt.show()
# 'results' now maps each case ('uniform', 0.0002, 0.1, 0.25) to its simulation output.

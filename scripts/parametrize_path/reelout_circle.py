import numpy as np
import matplotlib.pyplot as plt
import json

# from awetrim.kinematics.parametrized_patterns import Helix  # unused
from awetrim import SystemModel, State
from awetrim.utils.color_palette import set_plot_style, get_color_list
from awetrim.timeseries.phase_parametrized import PhaseParameterized
from awetrim.system.kite import Kite
from awetrim.system.tether import RigidLumpedTether
from awetrim.utils.defaults import PLOT_LABELS
from awetrim.environment.Wind import Wind

# ---------- Config ----------
mass_wing = 14.2
mass_kcu = 10
area_wing = 19.75
tether_diameter = 0.01

speed_wind_at_100 = 10
wind = Wind(
    wind_model="logarithmic",
    z0=0.0002,
)
speed_friction = 0.41 * speed_wind_at_100 / np.log(100 / wind.z0)
wind.speed_friction = speed_friction
# wind.speed_wind_ref = speed_wind_at_100

# color palette available via get_color_list() as needed


with open("./data/LEI-V3-KITE/v3_aero_input.json", "r") as file:
    aero_input_v3 = json.load(file)

pattern_config = {
    "pattern_type": "cst_helix",
    "path_parameters": {
        "omega": 1.0,
        "r0": 230.0,
        "az_amp0": np.deg2rad(10),
        "beta_amp0": np.deg2rad(10),
        "width_phi": 0.5,
        "width_beta": 0.5,
        "left_first": True,
        "normalize_bumps": False,
        "repeat_phi": True,
        "repeat_beta": True,
        "beta_coeffs": np.array([0, 0, 0, 0, 0]),
        "az_coeffs": [0, 0, 0, 0, 0],
        "kbeta": 0,
        "beta0": np.deg2rad(30),
        "kappa": 0,
    },
    "radial_parameters": {
        "reeling_strategy": "force",  # "force" or "constant"
        "force_model": "quadratic",  # "linear" or "quadratic"
        "reeling_speed": 0.0,  # m/s, only for constant reeling
        "max_tether_force": 2e4,  # N, only for force reeling
        "min_tether_force": 2000.0,  # N, only for force reeling
        "softplus": True,
        "softplus_beta": 1e-4,
        "softminus": True,
        "softminus_beta": 1e-3,
        "slope": 2716,  # N/(m/s)^2 for quadratic, N/(m/s) for linear
        "offset": 0,  # m/s
    },
    "start_time": 0,
    "end_time": 35,
    "start_angle": np.pi / 2,
    "end_angle": 2 * np.pi + np.pi / 2,
    "n_points": 600,
    "optimization_parameters": [],
}

# ---------- Starting state ----------
base_start_state = State(
    t=0,
    s=np.pi / 2,
    s_dot=1,
    s_ddot=0,
    length_tether=199.6,
    input_steering=0,
    tension_tether_ground=1e8,
    distance_radial=230,
    speed_radial=speed_wind_at_100 / 5,
)


def run_sim(
    aero_input,
    pattern_config,
    label_prefix,
    mass_wing,
    mass_kcu,
    area_wing,
    tether_diameter,
):
    result = {}
    phases = {}
    start_state = base_start_state
    simulation_types = ["quasi_steady", "dynamic"]
    for sim_type in simulation_types:
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
        else:
            continue
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

        model.input_depower = 0
        if sim_type == "no_mass":
            model.mass_wing = 0
            start_state["input_steering"] = 0
        phase = PhaseParameterized(
            model, quasi_steady=quasi_steady, pattern_config=pattern_config
        )
        phase.run_simulation(start_state=start_state)
        s_dot = phase.return_variable("s_dot")
        start_state.s_dot = s_dot[0]
        phases[sim_type] = phase

    return phases


phases = run_sim(
    aero_input_v3, pattern_config, "V3", mass_wing, mass_kcu, area_wing, tether_diameter
)

dynamic_phase = phases["dynamic"]
qs_phase = phases["quasi_steady"]

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
    x_param="s",
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
    x_param="s",
    axes=axes_map,
)

fig.legend(loc="upper center", bbox_to_anchor=(0.5, 0.95), ncol=2)
set_plot_style()
plt.tight_layout()
# Save the figure as pdf
plt.savefig("./results/figures/reelout_cst.pdf", bbox_inches="tight")
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

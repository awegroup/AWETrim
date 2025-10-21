import numpy as np
import matplotlib.pyplot as plt
import os
from awetrim.kinematics.parametrized_patterns import Helix, Lissajous, FigureEight
from awetrim.system.kite import Kite
from awetrim.system.tether import RigidLinkTether, RigidLumpedTether
from awetrim.kinematics.Kinematics import ParametrizedKinematics, KiteKinematics
from awetrim import SystemModel, State
from awetrim.utils.color_palette import set_plot_style_no_latex, get_color_list
from awetrim.timeseries.phase_parametrized import PhaseParameterized
from awetrim.environment import Wind
import json
import copy

file_path = "./data/LEI-V9-KITE/v9_aero_input.json"
with open(file_path, "r") as file:
    aero_input = json.load(file)

speed_wind_at_100 = 10  # m/s

wind = Wind(
    wind_model="uniform",  # logarithmic
    z0=0.1,  # roughness length
)
speed_friction = 0.41 * speed_wind_at_100 / np.log(100 / wind.z0)
wind.speed_friction = speed_friction
wind.speed_wind_ref = speed_wind_at_100

print("Friction speed:", wind.speed_friction)

# -----------------------------------------------
# Define the parametrized path
# -----------------------------------------------

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
    speed_radial=2,
    distance_radial=230,
)
time = np.arange(0, 50, 0.1)
s_array = np.linspace(5 * np.pi / 2, 9 * np.pi / 2, 200)
dof = 3
# -----------------------------------------------
# Define the system and aerodynamic model
# -----------------------------------------------
colors = get_color_list()
tension_tether_results = {}
phases = {}
parameters = ["speed_tangential", "tension_tether_ground", "angle_roll"]
x_param = "s"

N = 1

mass_wing = 60
mass_kcu = 30
area_wing = 46.85
tension_min = 3000
tension_max = 25000
tether_diameter = 0.01
tether = RigidLumpedTether(diameter=tether_diameter)
phases_qs = []
for i in range(N):

    pattern_config = {
        "pattern_type": "cst_lissajous",
        "path_parameters": {
            "omega": 1.0,
            "r0": 230.0,
            "az_amp0": 0.698,
            "beta_amp0": 0.2,
            "width_phi": 0.5,
            "width_beta": 0.5,
            "left_first": True,
            "normalize_bumps": False,
            "repeat_phi": True,
            "repeat_beta": True,
            "beta_coeffs": np.array([0, 0, 0, 0, 0]),
            "az_coeffs": [0, 0, 0, 0, 0],
            "kbeta": 0,
            "beta0": 0.6,
            "kappa": 0,
        },
        "radial_parameters": {
            "reeling_strategy": "force",  # "force" or "constant"
            "force_model": "quadratic",  # "linear" or "quadratic"
            "reeling_speed": 0.0,  # m/s, only for constant reeling
            "max_tether_force": tension_max,  # N, only for force reeling
            "min_tether_force": tension_min,  # N, only for force reeling
            "softplus": True,
            "softplus_beta": 1e-4,
            "softminus": True,
            "softminus_beta": 1e-3,
            "slope": 2700,  # N/(m/s)^2 for quadratic, N/(m/s) for linear
            "offset": 0,  # m/s
        },
        "start_time": 0,
        "end_time": 35,
        "start_angle": np.pi / 2,
        "end_angle": 2 * np.pi + np.pi / 2,
        "n_points": 600,
        "optimization_parameters": [
            "az_amp0",
            "beta_amp0",
            "beta0",
            "beta_coeffs",
            # "kappa",
            # "slope",
            # "offset",
        ],
    }

    # aero_input["params"]["angle_pitch_depower_0"] = thetat
    # Define kite model with current parameters
    kite = Kite(
        mass_wing=mass_wing,
        area_wing=area_wing,
        aero_input=aero_input,
        mass_kcu=mass_kcu,
        steering_control="asymmetric",
    )

    kite_model = SystemModel(
        dof=dof,
        quasi_steady=True,
        kite=kite,
        tether=tether,
        wind_model=wind,
    )
    kite_model.input_depower = 0

    # Run simulation

    phase = PhaseParameterized(
        kite_model,
        quasi_steady=True,
        pattern_config=pattern_config,
        tension_min=tension_min,
        tension_max=tension_max,
    )

    phase_base = PhaseParameterized(
        kite_model,
        quasi_steady=True,
        pattern_config=copy.deepcopy(pattern_config),
        tension_min=tension_min,
        tension_max=tension_max,
    )
    phase_base.run_simulation_phase(start_state=start_state)
    phases_qs.append(copy.deepcopy(phase_base))

    # Optimized
    phase.run_simulation_opti_phase(start_state=start_state)
    pattern_config = phase.pattern_config
    print("Optimized pattern configuration:")
    print(pattern_config)

    phase.run_simulation_phase(start_state=start_state)
    start_state = phase.states[0]

    s = phase.return_variable("s")
    s_dot = phase.return_variable("s_dot")

    phases_qs.append(copy.deepcopy(phase))

    print(s_dot[0])
    start_state["s_dot"] = s_dot[0]
    start_state["s"] = s[0]
    start_state["tension_tether_ground"] = phase.return_variable(
        "tension_tether_ground"
    )[0]

# -----------------------------------------------
# Plot results
# -----------------------------------------------
set_plot_style_no_latex()
save_folder = "./results/figures/"

# Plot baseline and optimized on the same overview figure
colors = get_color_list()
fig, axes_map, _ = phases_qs[0].plot_overview_3d(
    label="QS baseline",
    color=colors[0],
    linestyle="--",
    variables=[
        "speed_tangential",
        "tension_tether_ground",
        "input_steering",
        "mechanical_power",
    ],
    x_param="s",
)
phases_qs[-1].plot_overview_3d(
    label="QS opti",
    color=colors[1] if len(colors) > 1 else None,
    linestyle="-",
    variables=[
        "speed_tangential",
        "tension_tether_ground",
        "input_steering",
        "mechanical_power",
    ],
    x_param="s",
    axes=axes_map,
)

plt.tight_layout()
os.makedirs(save_folder, exist_ok=True)
plt.savefig(save_folder + "reelout_cst_qs_overview.pdf", bbox_inches="tight")
fig.legend(loc="upper left", bbox_to_anchor=(0.5, 0.95), ncol=2)
plt.show()

# Energy metrics
metrics = phases_qs[-1].energy_metrics(phases_qs[0])
print("\n--- Energy Metrics (QS opti vs QS baseline) ---")
print(
    f"Avg power baseline: {metrics['avg_power_other']:.2f} W, opti: {metrics['avg_power_self']:.2f} W"
)
print(f"Δ Power: {metrics['power_diff_percent']:.2f}%")

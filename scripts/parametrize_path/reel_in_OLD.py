import numpy as np
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from awetrim import SystemModel, State
from awetrim.timeseries.my_reelin_phase import ReelinPhase
from awetrim.system.kite import Kite
from awetrim.system.tether import RigidLumpedTether
from awetrim.environment.Wind import Wind
from awetrim.kinematics.parametrized_patterns import create_pattern_from_dict
import pickle
from awetrim.utils.color_palette import set_plot_style, get_color_list
from awetrim.utils.defaults import PLOT_LABELS

# ---------- Load precomputed fit data ----------
with open("fit_results.pkl", "rb") as f:
    fit_data = pickle.load(f)

C_sph = fit_data["C_sph"]
crs0 = fit_data["crs0"]
crsf = fit_data["crsf"]
phi0 = fit_data["phi0"]
phif = fit_data["phif"]
beta0 = fit_data["beta0"]
betaf = fit_data["betaf"]
C_interior = fit_data["C_interior"]
u_vals = fit_data["u_vals"]
U_interior = fit_data["U_interior"]
v0 = float(
    np.sqrt(fit_data["v0"][0] ** 2 + fit_data["v0"][1] ** 2 + fit_data["v0"][2] ** 2)
)

# ---------- Config ----------
wind = Wind(
    wind_model="uniform",
    z0=0.1,
)
wind.speed_wind_ref = 8

with open("./data/LEI-V9-KITE/v9_aero_input.json", "r") as file:
    aero_input_v9 = json.load(file)

pattern_config_v9 = {
    "pattern_type": "spline",
    "parameters": {
        "p": 3,
        "n_ctrl": 8,
        "r0": 300,
        "r1": 200,
        "crs0": crs0,
        "crsf": crsf,
        "phi0": phi0,
        "phif": phif,
        "beta0": beta0,
        "betaf": betaf,
        "C_interior": C_interior,
        "u_vals": u_vals,
        "U_interior": U_interior,
    },
    "start_time": 0,
    "end_time": 30,
    "n_points": 300,
    "optimization_parameters": [],
}

# ---------- Config ----------
speed_wind_at_100 = 7.6374
wind = Wind(
    wind_model="uniform",
    z0=0.1,
)
speed_friction = 0.41 * speed_wind_at_100 / np.log(100 / wind.z0)
# wind.speed_friction = speed_friction
wind.speed_wind_ref = speed_wind_at_100

colors = get_color_list()


with open("./data/LEI-V9-KITE/v9_aero_input.json", "r") as file:
    aero_input_v9 = json.load(file)

# ---------- Starting state ----------
base_start_state = State(
    t=0,
    s=0.01,
    s_dot=2,
    s_ddot=0,
    length_tether=199.6,
    input_steering=0,
    tension_tether_ground=1e8,
    distance_radial=330,
    speed_radial=2,
    timeder_speed_radial=0,
    input_depower=0,
)

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

        tether = RigidLumpedTether(diameter=tether_diameter)
        kite = Kite(
            mass_wing=mass_wing,
            area_wing=area_wing,
            aero_input=aero_input,
            steering_control="asymmetric",
        )

        model = SystemModel(
            dof=3,
            quasi_steady=quasi_steady,
            kite=kite,
            tether=tether,
            wind_model=wind,
            neglect_radial_acceleration=False,
        )

        phase = ReelinPhase(
            model, 
            quasi_steady=quasi_steady, 
            pattern_config=pattern_config
        )

        phase.run_simulation(start_state=start_state)
        
        # Store results like in reel-in-old.py
        result[sim_type] = {
            "x": phase.return_variable("x"),
            "y": phase.return_variable("y"),
            "z": phase.return_variable("z"),
            "phase": phase,
        }

    return result

# Run simulation
results_v9 = run_sim(
    aero_input_v9, pattern_config_v9, "V9", 90, 47, 0.01, 2, marker="^"
)

# 3D plot exactly like in reel-in-old.py
fig_3d = plt.figure()
ax_3d = fig_3d.add_subplot(111, projection="3d")
ax_3d.plot(
    results_v9["quasi_steady"]["x"],
    results_v9["quasi_steady"]["y"],
    results_v9["quasi_steady"]["z"],
    label="Quasi-Steady Trajectory",
)
ax_3d.set_xlabel("X")
ax_3d.set_ylabel("Y")
ax_3d.set_zlabel("Z")
ax_3d.legend()
plt.show()
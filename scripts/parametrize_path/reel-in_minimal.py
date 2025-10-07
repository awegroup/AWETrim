import numpy as np
import matplotlib.pyplot as plt
import json
import casadi as ca
from picawe import SystemModel, State
from picawe.utils.color_palette import set_plot_style, get_color_list, custom_cmap
from picawe.timeseries.reelin_phase_minimal import ReelinPhase
from picawe.system.kite import Kite
from picawe.system.tether import RigidLumpedTether
from picawe.utils.defaults import PLOT_LABELS
from picawe.environment.Wind import Wind
from picawe.kinematics.parametrized_patterns import create_pattern_from_dict
import pickle

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
v0 = float(np.sqrt(fit_data["v0"][0]**2 + fit_data["v0"][1]**2 + fit_data["v0"][2]**2))

# ---------- Config ----------
speed_wind_at_100 = 10
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
    "end_time": 150,
    "start_angle": 0,
    "end_angle": 1,
    "n_points": 400,
    "optimization_parameters": [],
}

# Calculate realistic s_dot from your fitted data and kinematics
def calculate_consistent_speeds(s_current, pattern_config, v0_target, tether_length=330):
    """
    Calculate consistent speeds based on spline derivatives and target physical speed
    """
    test_pattern = create_pattern_from_dict(pattern_config)
    result = test_pattern.evaluate_spline(tether_length, s_current)
    
    # Get path derivatives 
    dphi_ds = float(result["dS"][0])
    dbeta_ds = float(result["dS"][1])
    
    # Angular speed magnitude per unit s
    angular_speed_magnitude = np.sqrt(dphi_ds**2 + dbeta_ds**2)  # rad per unit s
    
    if angular_speed_magnitude < 1e-6:
        print("WARNING: Very small angular derivatives")
        return 0.001, 1.0  # Very slow fallback
    
    target_tangential_speed = v0_target  # Your fitted speed
    calculated_s_dot = target_tangential_speed / (tether_length * angular_speed_magnitude)
    
    # Check resulting speeds
    implied_angular_vel = angular_speed_magnitude * calculated_s_dot
    implied_tangential_speed = tether_length * implied_angular_vel
    
    # Use much smaller radial speed (mostly tangential motion)
    realistic_speed_radial = target_tangential_speed * 0.05  # 5% of tangential
    
    return calculated_s_dot, realistic_speed_radial

# Add this calculation before updating base_start_state
print(f"\n=== Calculating Realistic Speeds ===")
print(f"Fitted speed magnitude v0: {v0:.2f} m/s")

# Calculate realistic speeds
s_dot_realistic, speed_radial_realistic = calculate_consistent_speeds(0.2, pattern_config_v9, v0)

# Update the starting state with physically consistent values
base_start_state = State(
    t=0,
    s=0.2,
    s_dot=s_dot_realistic,  # Use calculated realistic s_dot
    s_ddot=0,
    length_tether=330,
    input_steering=0,
    tension_tether_ground=3000,
    distance_radial=330,
    speed_radial=speed_radial_realistic,  # Use calculated realistic radial speed
    timeder_speed_radial=0,
    input_depower=0.1,  # Reduce depower
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
        else:
            continue

        label = f"{label_prefix} {sim_type.replace('_', ' ').title()}"
        print(f"Running simulation for {sim_type} with label: {label}")
    
        try:
            linestyle = {
                "quasi_steady": "--",
                "dynamic": "-",
                "inertia_free": ":",
                "no_mass": "-.",
            }[sim_type]
            color = colors[color_base]
            tether = RigidLumpedTether(
                diameter=tether_diameter,
            )
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
                model, quasi_steady=quasi_steady, pattern_config=pattern_config
            )

            phase.run_simulation(start_state=start_state)
            if len(phase.states) > 1:
                print(f"Successfully completed {len(phase.states)} simulation steps")
            else:
                print(f"Simulation failed or produced no results")
                
        except Exception as e:
            print(f"Error in {sim_type} simulation: {e}")
            continue

    return result

results_v9 = run_sim(
    aero_input_v9, pattern_config_v9, "V9", 90, 47, 0.01, 2, marker="^"
)
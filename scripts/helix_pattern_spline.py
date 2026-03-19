import json
from pathlib import Path

import numpy as np

from awetrim import SystemModel
from awetrim.environment.Wind import Wind
from awetrim.system.kite import Kite
from awetrim.system.tether import RigidLumpedTether
from awetrim.timeseries.reelout_phase import Reelout
from awetrim.system.system_model import create_system_model_from_yaml
from awetrim.utils.utils import load_cycle_config_from_yaml

"""
Reelout pattern simulation using YAML configuration files.

Configuration is split into two YAML files:
- Kite/physical properties: data/LEI-V3-KITE/v3_kite_input.yaml
- Cycle parameters (reelout/reelin/wind): data/LEI-V3-KITE/v3_cycle_config.yaml

To modify parameters:
1. Edit the YAML files directly, or
2. Load configs and override specific values in this script
"""

# ---------------------------------------------------------------------------
# Configuration files
# ---------------------------------------------------------------------------
KITE_CONFIG_PATH = Path("data/LEI-V3-KITE/v3_kite_input.yaml")
# CYCLE_CONFIG_PATH = Path("data/LEI-V3-KITE/v3_helix_config_example.yaml")
CYCLE_CONFIG_PATH = Path("data/LEI-V3-KITE/v3_helix_config_example_spline.yaml")
CYCLE_CONFIG_PATH = Path(
    "results/optimized_configs/helix/depower_helix_optimized_config_wind_6_z0_0.03_logarithmic_spline.yaml"
)

# Load configurations from YAML
REELOUT_CONFIG, REELIN_CONFIG = load_cycle_config_from_yaml(CYCLE_CONFIG_PATH)
# REELOUT_CONFIG["path_parameters"]["beta0"] += 0.1
# REELOUT_CONFIG["path_parameters"]["beta_amp0"] = 0.18
# REELOUT_CONFIG["path_parameters"]["az_amp0"] = 0.3536
# # REELOUT_CONFIG["path_parameters"]["kappa"] = 1
# REELOUT_CONFIG["path_parameters"]["kbeta"] = 0
REELOUT_CONFIG["sim_parameters"]["input_depower"] += 0.05
WIND_CONFIG = {
    "speed_wind_at_100": 5,
    "z0": 0.03,
    "model_type": "logarithmic",
}
START_STATE = {
    "t": 0,
    "s": 0,
    "s_dot": 2,
    "input_steering": 0,
    "tension_tether_ground": 8.4e3,  # Initial guess for tension (N)
    "speed_radial": 0,  # Positive for reel-out
    "distance_radial": REELOUT_CONFIG["path_parameters"][
        "r0"
    ],  # Start at the specified radius
}


def build_wind_model(speed_wind_at_100=8, z0=0.01, model_type="uniform"):
    """Create a wind model using the supplied parameters."""
    wind_model = Wind(
        wind_model=model_type,
        z0=z0,
    )
    speed_friction = 0.41 * speed_wind_at_100 / np.log(100 / wind_model.z0)
    if model_type == "logarithmic":
        wind_model.speed_friction = speed_friction
    elif model_type == "uniform":
        wind_model.speed_wind_ref = speed_wind_at_100
    return wind_model


# TODO: Make sure it is the number of figure eights that I want
def main(run_plots=False):

    system_model = create_system_model_from_yaml(yaml_path=KITE_CONFIG_PATH)

    wind_model = build_wind_model(
        speed_wind_at_100=WIND_CONFIG["speed_wind_at_100"],
        z0=WIND_CONFIG["z0"],
        model_type=WIND_CONFIG["model_type"],
    )
    system_model.wind = wind_model
    reelout = Reelout(
        system_model=system_model,
        pattern_config=REELOUT_CONFIG,
        start_state=START_STATE,
    )
    optimization_params = [
        "C_phi",
        "C_beta",
        "input_depower",
    ]
    phase, axes = reelout.run_simulation(run_plots=run_plots, phase_sim=True)
    us = phase.return_variable("input_steering")
    t = phase.return_variable("t")
    # us_rate = np.gradient(us, t)
    # plt.figure()
    # plt.plot(t, us_rate)
    # plt.show()
    lift = phase.return_variable("lift_coefficient")
    print("Average lift coefficient:", np.mean(lift))
    drag = phase.return_variable("drag_coefficient")
    print("Average drag coefficient:", np.mean(drag))
    aoa = phase.return_variable("angle_of_attack")
    print("Average angle of attack (deg):", np.degrees(np.mean(aoa)))
    print("Max angle of attack (deg):", np.degrees(np.max(aoa)))
    print(phase.energy_metrics())
    # plt.figure()
    # plt.plot(t, np.degrees(aoa))
    # plt.show()
    solution = reelout.run_simulation_opti(optimization_params=optimization_params)
    depower_prefix = "depower_" if "input_depower" in optimization_params else ""
    filename = f"{depower_prefix}helix_optimized_config_wind_{WIND_CONFIG['speed_wind_at_100']}_z0_{WIND_CONFIG['z0']}_{WIND_CONFIG['model_type']}_spline.yaml"

    solution.save_config_to_yaml(Path("results/optimized_configs/helix") / filename)
    phase, _ = reelout.run_simulation(run_plots=run_plots, axes=axes, phase_sim=True)

    # reelout.pattern_config["path_parameters"]["kappa"] = 1
    # phase, _ = reelout.run_simulation(run_plots=run_plots, axes=axes)
    print(phase.energy_metrics())

    plt.show()

    phase.plot_overview(x_param="t")
    plt.show()
    return reelout


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    main(run_plots=True)
    plt.show()

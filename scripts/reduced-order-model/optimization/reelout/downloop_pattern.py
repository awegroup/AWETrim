import json
from pathlib import Path

import numpy as np

from awetrim import SystemModel
from awetrim.environment.Wind import Wind
from awetrim.system.kite import Kite
from awetrim.system.tether import RigidLumpedTether
from awetrim.timeseries.phase import Phase
from awetrim.system.factory import create_system_model_from_yaml
from awetrim.kinematics.parametrized_patterns import (
    make_bspline_path_parameters_from_named_curve,
)
from awetrim.identification.controls import ROM_POWERED_INPUT_DEPOWER
from awetrim.utils.config_paths import (
    LEI_V3_DOWNLOOP_SPLINE_CONFIG,
    LEI_V3_SYSTEM_CONFIG,
)
from awetrim.utils.utils import load_cycle_config_from_yaml

"""
Reelout pattern simulation using YAML configuration files.

Configuration is split into two YAML files:
- System properties: data/LEI-V3-KITE/lei_v3_system_config.yaml
- Cycle parameters: data/LEI-V3-KITE/cycle_configs/downloop_spline.yaml

To modify parameters:
1. Edit the YAML files directly, or
2. Load configs and override specific values in this script
"""

# ---------------------------------------------------------------------------
# Configuration files
# ---------------------------------------------------------------------------
KITE_CONFIG_PATH = LEI_V3_SYSTEM_CONFIG
CYCLE_CONFIG_PATH = LEI_V3_DOWNLOOP_SPLINE_CONFIG

SAVE_TIMESERIES = True

RESULTS_DIR = (
    Path("results") / KITE_CONFIG_PATH.parent.name / "optimization" / "downloops"
)

# ---------------------------------------------------------------------------
# Initial path guess
# ---------------------------------------------------------------------------
# Regenerate the reel-out B-spline control points from a simple analytic curve
# and override the ones loaded from YAML. Set REGENERATE_INITIAL_GUESS = False
# to fly the control points stored in the config instead. This is the inline
# equivalent of running generate_spline_config.py before this script.
REGENERATE_INITIAL_GUESS = True
INITIAL_GUESS = {
    "curve_type": "lissajous",  # "lissajous" or "lemniscate" (smoother eight)
    "M": 10,
    "n_fit": 400,
    "s_init": 0.0,
    "s_final": 2.0 * np.pi,
    "az_amp0": 0.3,
    "beta0": 0.35,
    "beta_amp0": 0.12,
    "downloops": True,  # downloop direction
}

# CYCLE_CONFIG_PATH = Path(
#     "results/optimized_configs/downloops/depower_downloop_optimized_config_wind_12_z0_0.03_logarithmic_spline.yaml"
# )

# Load configurations from YAML
REELOUT_CONFIG, REELIN_CONFIG = load_cycle_config_from_yaml(CYCLE_CONFIG_PATH)

if REGENERATE_INITIAL_GUESS:
    REELOUT_CONFIG["path_parameters"] = make_bspline_path_parameters_from_named_curve(
        spline_type="periodic",
        r0=REELOUT_CONFIG["path_parameters"]["r0"],
        **INITIAL_GUESS,
    )
    REELOUT_CONFIG["sim_parameters"]["start_angle"] = INITIAL_GUESS["s_init"]
    REELOUT_CONFIG["sim_parameters"]["end_angle"] = INITIAL_GUESS["s_final"]

REELOUT_CONFIG["sim_parameters"]["n_points"] = 100
REELOUT_CONFIG["sim_parameters"]["input_depower"] = 1.6
REELOUT_CONFIG["sim_parameters"]["reg_weight"] = 1.0
REELOUT_CONFIG["sim_parameters"]["detect_simple_bounds"] = True
WIND_CONFIG = {
    "speed_wind_at_100": 8,
    "z0": 0.03,
    "model_type": "logarithmic",
}
START_STATE = {
    "t": 0,
    "s": 0,
    "s_dot": 1,
    "input_steering": 0,
    "tension_tether_ground": 8.4e5,  # Initial guess for tension (N)
    "speed_radial": 1,  # Positive for reel-out
    "distance_radial": REELOUT_CONFIG["path_parameters"][
        "r0"
    ],  # Start at the specified radius
}


def build_wind_model(speed_wind_at_100=8, z0=0.01, model_type="uniform"):
    """Create a wind model using the supplied parameters."""
    wind_model = Wind(
        wind_model=model_type,
        z0=z0,
        direction_wind=0,  # Wind coming from x-direction
    )
    speed_friction = 0.41 * speed_wind_at_100 / np.log(100 / wind_model.z0)
    if model_type == "logarithmic":
        wind_model.speed_friction = speed_friction
    elif model_type == "uniform":
        wind_model.speed_wind_ref = speed_wind_at_100
    return wind_model


def main(run_plots=False):

    system_model = create_system_model_from_yaml(yaml_path=KITE_CONFIG_PATH)

    wind_model = build_wind_model(
        speed_wind_at_100=WIND_CONFIG["speed_wind_at_100"],
        z0=WIND_CONFIG["z0"],
        model_type=WIND_CONFIG["model_type"],
    )
    system_model.wind = wind_model
    reelout = Phase(
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
    filename = f"{depower_prefix}downloop_optimized_config_wind_{WIND_CONFIG['speed_wind_at_100']}_z0_{WIND_CONFIG['z0']}_{WIND_CONFIG['model_type']}_spline.yaml"

    solution.save_config_to_yaml(RESULTS_DIR / filename)
    phase, _ = reelout.run_simulation(run_plots=True, axes=axes, phase_sim=True)
    print(phase.energy_metrics())

    if SAVE_TIMESERIES:
        phase.save_timeseries_csv(
            RESULTS_DIR / filename.replace(".yaml", "_timeseries.csv")
        )

    return reelout


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    main(run_plots=True)
    plt.show()

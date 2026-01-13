"""Run a cycle simulation (reel-out then reel-in) using existing configs.

Usage:
    python run_cycle_simulation.py [--plot]

"""

import matplotlib.pyplot as plt
from pathlib import Path
import importlib.util
import argparse
from awetrim.system.system_model import create_system_model_from_yaml
from awetrim.utils.utils import load_cycle_config_from_yaml
from awetrim.environment.Wind import Wind
import numpy as np

# helper to dynamically import the script modules by path
WIND_CONFIG = {
    "speed_wind_at_100": 12,
    "z0": 0.03,
    "model_type": "logarithmic",
}

n_half_loops = 7

# Configuration file path
CONFIG_PATH = Path("data/LEI-V3-KITE/v3_downloop_config_example.yaml")


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


def load_module_from_path(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore
    return module


def main(run_plots: bool = False):

    # Load both reelout and reelin configurations from the YAML file
    REELOUT_CONFIG, REELIN_CONFIG = load_cycle_config_from_yaml(CONFIG_PATH)
    REELOUT_CONFIG["sim_parameters"]["start_angle"] = np.pi / 2
    REELOUT_CONFIG["sim_parameters"]["end_angle"] = np.pi * (n_half_loops)
    REELOUT_CONFIG["sim_parameters"]["n_points"] = 50 * n_half_loops / 2
    # REELIN_CONFIG["sim_parameters"]["n_points"] = 200
    # Build a single wind and system model here (do it once and reuse)
    # Use the helper functions and config from the reelout script to assemble
    # the components so both phases share the exact same SystemModel and wind.
    system_model = create_system_model_from_yaml(
        yaml_path=Path("data/LEI-V3-KITE/v3_kite_input.yaml"),
    )

    wind_model = build_wind_model(
        speed_wind_at_100=WIND_CONFIG["speed_wind_at_100"],
        z0=WIND_CONFIG["z0"],
        model_type=WIND_CONFIG["model_type"],
    )
    system_model.wind = wind_model

    # Import classes
    from awetrim.timeseries.reelout_phase import Reelout
    from awetrim.timeseries.reelin_phase import ReelinSimple
    from awetrim.timeseries.cycle_phase import CycleSimple

    optimization_params = [
        # "az_amp0",
        # "beta_amp0",
        "beta0",
        "elevation_start_riro",
        "offset_winch_ri",
        "slope_winch_ri",
        # "offset_winch_ro",
        "slope_winch_ro",
        "beta_coeffs",
        "az_coeffs",
        # "kappa",
    ]
    print("Reelout config:", REELOUT_CONFIG)
    print("Reelin config:", REELIN_CONFIG)

    # Instantiate phases using the configs loaded from YAML
    reelout = Reelout(
        system_model=system_model,
        pattern_config=REELOUT_CONFIG,
        depower=0,
    )
    opti_params_ro = [
        "az_amp0",
        "beta_amp0",
        "beta0",
        # "slope_winch_ro",
        "beta_coeffs",
        "az_coeffs",
        # "kappa",
    ]
    # reelout.run_simulation_opti(optimization_params=opti_params_ro, target="power")
    reelin = ReelinSimple(
        system_model=system_model,
        pattern_config=REELIN_CONFIG,
        depower_ri=1,
        depower_riro=1,
    )

    cycle = CycleSimple(reelin=reelin, reelout=reelout)

    print("Running cycle simulation: reel-out -> (opt) reel-in -> transition")
    result = cycle.run_cycle_simulation(optimize_reelin=True, plotting=run_plots)
    plt.show()
    cycle.run_cycle_opti(optimization_params=optimization_params)
    print(cycle.reelin.pattern_config)
    print(cycle.reelout.pattern_config)
    result = cycle.run_cycle_simulation(optimize_reelin=False, plotting=run_plots)
    plt.show()
    if result is None:
        print("Cycle simulation failed")
        return 1

    print("Cycle simulation complete")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true", help="Show combined plots")
    args = parser.parse_args()
    main(run_plots=True)

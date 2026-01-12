import json
from pathlib import Path

from awetrim.utils.utils import load_cycle_config_from_yaml
import numpy as np

from awetrim import SystemModel
from awetrim.environment.Wind import Wind
from awetrim.system.kite import Kite
from awetrim.system.tether import RigidLumpedTether
from awetrim.timeseries.reelout_phase import Reelout
from awetrim.system.system_model import create_system_model_from_yaml

# ---------------------------------------------------------------------------
# Configuration knobs – tweak these values to experiment with the setup.
# ---------------------------------------------------------------------------
PHYSICAL_CONFIG = {
    "mass_wing": 15,
    "mass_kcu": 0,
    "area_wing": 19.75,
    "tether_diameter": 0.006,
}

PATH_PARAMETERS = {
    "r0": 200,
    "az_amp0": 0.25,
    "beta_amp0": 0.1,
    "beta_coeffs": np.array([0, 0, 0, 0, 0]),
    "az_coeffs": [0, 0, 0, 0, 0],
    "kbeta": 0,
    "beta0": 0.55,
    "kappa": 0,
    "phi0": 0,
    # "downloops": True,|
    # "distance_radial_start": 200,
}

RADIAL_PARAMETERS = {
    "reeling_strategy": "force",  # "force" or "constant"
    "force_model": "quadratic",  # "linear" or "quadratic"
    "reeling_speed": 0.0,  # m/s, only for constant reeling
    "max_tether_force": 8400,  # N, only for force reeling
    "min_tether_force": 1000,  # N, only for force reeling
    "softplus": True,
    "softplus_beta": 1e-3,
    "softminus": True,
    "softminus_beta": 1e-3,
    "slope_winch_ro": 800,  # N/(m/s)^2 for quadratic, N/(m/s) for linear
    "offset_winch_ro": 0,  # m/s
}

N = 6  # Number of half eight loops
SIM_PARAMETERS = {
    "start_time": 0,
    "end_time": 35,
    "start_angle": np.pi / 2,
    "end_angle": N * np.pi + np.pi / 2,
    "n_points": 400,
}
CYCLE_CONFIG_PATH = Path("data/LEI-V3-KITE/v3_downloop_config_example.yaml")
# CYCLE_CONFIG_PATH = Path("data/LEI-V3-KITE/v3_uploop_config_example.yaml")
# CYCLE_CONFIG_PATH = Path("data/LEI-V3-KITE/v3_optimized_config.yaml")

# Load configurations from YAML
# REELOUT_CONFIG, _ = load_cycle_config_from_yaml(CYCLE_CONFIG_PATH)
REELOUT_CONFIG = {
    "pattern_type": "cst_lissajous",
    "path_parameters": PATH_PARAMETERS,
    "radial_parameters": RADIAL_PARAMETERS,
    "sim_parameters": SIM_PARAMETERS,
}

AERO_INPUT_FILE = Path("data/LEI-V3-KITE/v3_aero_input.json")

WIND_CONFIG = {
    "speed_wind_at_200": 10,
    "z0": 0.1,
    "model_type": "uniform",
}


def build_wind_model(speed_wind_at_200=8, z0=0.01, model_type="uniform"):
    """Create a wind model using the supplied parameters."""
    wind_model = Wind(
        wind_model=model_type,
        z0=z0,
    )
    speed_friction = 0.41 * speed_wind_at_200 / np.log(200 / wind_model.z0)
    if model_type == "logarithmic":
        wind_model.speed_friction = speed_friction
    elif model_type == "uniform":
        wind_model.speed_wind_ref = speed_wind_at_200
    return wind_model


def main(run_plots=False):

    system_model = create_system_model_from_yaml(
        yaml_path=Path("data/LEI-V3-KITE/v3_kite_input.yaml"),
    )

    wind_model = build_wind_model(
        speed_wind_at_200=WIND_CONFIG["speed_wind_at_200"],
        z0=WIND_CONFIG["z0"],
        model_type=WIND_CONFIG["model_type"],
    )
    system_model.wind = wind_model
    reelout = Reelout(
        system_model=system_model,
        pattern_config=REELOUT_CONFIG,
        depower=0,
    )
    optimization_params = [
        # "az_amp0",
        # "beta_amp0",
        "beta0",
        "slope_winch_ro",
        # "offset",
        "beta_coeffs",
        "az_coeffs",
    ]
    phase, axes = reelout.run_simulation(run_plots=run_plots)
    lift = phase.return_variable("lift_coefficient")
    print("Average lift coefficient:", np.mean(lift))
    drag = phase.return_variable("drag_coefficient")
    print("Average drag coefficient:", np.mean(drag))
    print(phase.energy_metrics())
    solution = reelout.run_simulation_opti(optimization_params=optimization_params)
    phase, _ = reelout.run_simulation(run_plots=run_plots, axes=axes)

    # reelout.pattern_config["path_parameters"]["kappa"] = 1
    # phase, _ = reelout.run_simulation(run_plots=run_plots, axes=axes)
    print(phase.energy_metrics())

    plt.show()
    return reelout


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    main(run_plots=True)
    plt.show()

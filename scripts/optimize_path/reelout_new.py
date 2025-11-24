import json
from pathlib import Path

import numpy as np

from awetrim import SystemModel
from awetrim.environment.Wind import Wind
from awetrim.system.kite import Kite
from awetrim.system.tether import RigidLumpedTether
from awetrim.timeseries.reelout_phase import Reelout

# ---------------------------------------------------------------------------
# Configuration knobs – tweak these values to experiment with the setup.
# ---------------------------------------------------------------------------
PHYSICAL_CONFIG = {
    "mass_wing": 61,
    "mass_kcu": 30,
    "area_wing": 46.85,
    "tether_diameter": 0.014,
}

PATH_PARAMETERS = {
    "r0": 230,
    "az_amp0": 0.4814306739489051 * 1.1,
    "beta_amp0": 0.08726645323472254 * 1.1,
    "beta_coeffs": np.array(
        [0.23282922, -1.0000000, 0.07106071, -0.8524058, 0.46303606]
    ),
    "az_coeffs": [0, 0, 0, 0, 0],
    "kbeta": 0,
    "beta0": 0.45090333335903443 * 1.1,
    "kappa": 0,
    "downloops": True,
    "distance_radial_start": 230,
}

RADIAL_PARAMETERS = {
    "reeling_strategy": "force",  # "force" or "constant"
    "force_model": "quadratic",  # "linear" or "quadratic"
    "reeling_speed": 0.0,  # m/s, only for constant reeling
    "max_tether_force": 25000,  # N, only for force reeling
    "min_tether_force": 4000,  # N, only for force reeling
    "softplus": True,
    "softplus_beta": 1e-4,
    "softminus": True,
    "softminus_beta": 1e-3,
    "slope_winch_ro": 8000,  # N/(m/s)^2 for quadratic, N/(m/s) for linear
    "offset_winch_ro": 0,  # m/s
}

N = 4  # Number of half eight loops
SIM_PARAMETERS = {
    "start_time": 0,
    "end_time": 35,
    "start_angle": 0,
    "end_angle": N * np.pi,
    "n_points": 400,
}

REELOUT_CONFIG = {
    "pattern_type": "cst_lissajous",
    "path_parameters": PATH_PARAMETERS,
    "radial_parameters": RADIAL_PARAMETERS,
    "sim_parameters": SIM_PARAMETERS,
}

AERO_INPUT_FILE = Path("data/LEI-V9-KITE/v9_aero_input.json")

WIND_CONFIG = {
    "speed_wind_at_100": 14,
    "z0": 0.002,
    "model_type": "logarithmic",
}


def load_aero_input(path: Path = AERO_INPUT_FILE):
    """Load aerodynamic input data from disk."""
    with path.open("r") as file:
        return json.load(file)


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


def define_system(
    tether_diameter,
    mass_wing,
    mass_kcu,
    area_wing,
    aero_input,
    wind_model,
):
    """Instantiate a SystemModel with the supplied components."""

    tether = RigidLumpedTether(diameter=tether_diameter)
    kite = Kite(
        mass_wing=mass_wing,
        mass_kcu=mass_kcu,
        area_wing=area_wing,
        aero_input=aero_input,
        steering_control="asymmetric",
    )

    model = SystemModel(
        dof=3,
        kite=kite,
        tether=tether,
        wind_model=wind_model,
    )
    return model


def create_system_model():
    """Assemble the system model using the configuration dictionaries above."""
    aero_input = load_aero_input()
    wind_model = build_wind_model(
        speed_wind_at_100=WIND_CONFIG["speed_wind_at_100"],
        z0=WIND_CONFIG["z0"],
        model_type=WIND_CONFIG["model_type"],
    )
    return define_system(
        tether_diameter=PHYSICAL_CONFIG["tether_diameter"],
        mass_wing=PHYSICAL_CONFIG["mass_wing"],
        mass_kcu=PHYSICAL_CONFIG["mass_kcu"],
        area_wing=PHYSICAL_CONFIG["area_wing"],
        aero_input=aero_input,
        wind_model=wind_model,
    )


def main(run_plots=False):
    system_model = create_system_model()
    reelout = Reelout(
        system_model=system_model,
        pattern_config=REELOUT_CONFIG,
        depower=0,
    )
    optimization_params = [
        "az_amp0",
        "beta_amp0",
        "beta0",
        "slope_winch_ro",
        # "offset",
        "beta_coeffs",
        # "kappa",
    ]
    phase, axes = reelout.run_simulation(run_plots=run_plots)

    solution = reelout.run_simulation_opti(optimization_params=optimization_params)
    phase, _ = reelout.run_simulation(run_plots=run_plots, axes=axes)
    print(phase.energy_metrics())
    reelout.pattern_config["path_parameters"]["kappa"] = 1
    phase, _ = reelout.run_simulation(run_plots=run_plots, axes=axes)
    print(phase.energy_metrics())

    plt.show()
    return reelout


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    main(run_plots=True)
    plt.show()

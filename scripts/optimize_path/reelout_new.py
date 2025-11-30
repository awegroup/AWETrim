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
    "mass_wing": 15,
    "mass_kcu": 0,
    "area_wing": 19.75,
    "tether_diameter": 0.006,
}

PATH_PARAMETERS = {
    "r0": 200,
    "az_amp0": 0.0956087469720186,
    "beta_amp0": 0.04317,
    "beta_coeffs": np.array([0, 0]),
    "az_coeffs": [-0.48624925, 0, 0, 0, 0],
    "kbeta": 0,
    "beta0": 0.31,
    "kappa": 0,
    # "downloops": True,|
    # "distance_radial_start": 200,
}

RADIAL_PARAMETERS = {
    "reeling_strategy": "force",  # "force" or "constant"
    "force_model": "quadratic",  # "linear" or "quadratic"
    "reeling_speed": 0.0,  # m/s, only for constant reeling
    "max_tether_force": 8400,  # N, only for force reeling
    "min_tether_force": 1500,  # N, only for force reeling
    "softplus": True,
    "softplus_beta": 1e-4,
    "softminus": True,
    "softminus_beta": 1e-3,
    "slope_winch_ro": 1500,  # N/(m/s)^2 for quadratic, N/(m/s) for linear
    "offset_winch_ro": 0,  # m/s
}

N = 2  # Number of half eight loops
SIM_PARAMETERS = {
    "start_time": 0,
    "end_time": 35,
    "start_angle": np.pi / 2,
    "end_angle": N * np.pi + np.pi / 2,
    "n_points": 200,
}

REELOUT_CONFIG = {
    "pattern_type": "cst_helix",
    "path_parameters": PATH_PARAMETERS,
    "radial_parameters": RADIAL_PARAMETERS,
    "sim_parameters": SIM_PARAMETERS,
}

AERO_INPUT_FILE = Path("data/LEI-V3-KITE/v3_aero_input.json")

WIND_CONFIG = {
    "speed_wind_at_200": 9,
    "z0": 0.1,
    "model_type": "logarithmic",
}


def load_aero_input(path: Path = AERO_INPUT_FILE):
    """Load aerodynamic input data from disk."""
    with path.open("r") as file:
        return json.load(file)


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
        speed_wind_at_200=WIND_CONFIG["speed_wind_at_200"],
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
        # "beta_coeffs",
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

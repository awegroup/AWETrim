import json
from pathlib import Path

import numpy as np

from awetrim import SystemModel
from awetrim.environment.Wind import Wind
from awetrim.system.kite import Kite
from awetrim.system.tether import RigidLumpedTether
from awetrim.timeseries.reelin_phase import ReelinSimple
from awetrim.system.system_model import create_system_model_from_yaml

# ---------------------------------------------------------------------------
# Configuration knobs – tweak these values to experiment with the setup.
# ---------------------------------------------------------------------------
PHYSICAL_CONFIG = {
    "mass_wing": 15,
    "mass_kcu": 15,
    "area_wing": 19.75,
    "tether_diameter": 0.006,
}

PATH_PARAMETERS = {
    "elevation_start_ri": np.radians(30),
    "elevation_start_riro": np.radians(70),
    "elevation_start_ro": np.radians(30),
    "distance_radial_start": 360,
    "distance_radial_end": 230,
}

RADIAL_PARAMETERS = {
    "reeling_strategy": "force",
    "force_model": "quadratic",
    "reeling_speed": 1.0,
    "max_tether_force": 8400.0,
    "min_tether_force": 2000.0,
    "softplus": True,
    "softplus_beta": 1e-4,
    "softminus": True,
    "softminus_beta": 1e-3,
    "slope_winch_ri": 562,
    "offset_winch_ri": -5,
}

REELIN_CONFIG = {
    "path_parameters": PATH_PARAMETERS,
    "radial_parameters": RADIAL_PARAMETERS,
    "sim_parameters": {
        "start_time": 0,
    },
}

AERO_INPUT_FILE = Path("data/LEI-V3-KITE/v3_aero_input.json")


def load_aero_input(path: Path = AERO_INPUT_FILE):
    """Load aerodynamic input data from disk."""
    with path.open("r") as file:
        return json.load(file)


def build_wind_model(speed_wind_at_100=6, z0=0.0002, model_type="uniform"):
    """Create a wind model using the supplied parameters."""
    wind_model = Wind(
        wind_model=model_type,
        z0=z0,
    )
    speed_friction = 0.41 * speed_wind_at_100 / np.log(100 / wind_model.z0)
    wind_model.speed_friction = speed_friction
    wind_model.speed_wind_ref = speed_wind_at_100
    return wind_model


def main(run_plots=False):
    system_model = create_system_model_from_yaml(
        yaml_path=Path("data/LEI-V3-KITE/v3_kite_input.yaml"),
    )
    wind_model = build_wind_model(
        speed_wind_at_100=8,
        z0=0.01,
        model_type="logarithmic",
    )
    system_model.wind = wind_model
    reelin = ReelinSimple(
        system_model=system_model,
        pattern_config=REELIN_CONFIG,
        depower_ri=1,
        depower_riro=1,
    )
    import matplotlib.pyplot as plt

    reelin.run_simulation(run_plots=run_plots)
    plt.show()
    optimization_params = [
        "elevation_start_riro",
        # "offset_winch_ri",
    ]  # , "slope_winch_ri"]
    solution = reelin.run_simulation_opti(
        optimization_params=optimization_params, target="zero"
    )
    reelin.run_simulation(solution=solution, run_plots=run_plots)
    return reelin


if __name__ == "__main__":
    main(run_plots=True)

import json
from pathlib import Path

import numpy as np

from awetrim import SystemModel
from awetrim.environment.Wind import Wind
from awetrim.system.kite import Kite
from awetrim.system.tether import RigidLumpedTether
from awetrim.timeseries.reelin_phase import ReelinSimple

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
    "elevation_start_ri": np.radians(40),
    "elevation_start_riro": np.radians(90),
    "elevation_start_ro": np.radians(40),
    "distance_radial_start": 360,
    "distance_radial_end": 230,
}

RADIAL_PARAMETERS = {
    "reeling_strategy": "force",
    "force_model": "quadratic",
    "reeling_speed": 1.0,
    "max_tether_force": 25e4,
    "min_tether_force": 4000.0,
    "softplus": False,
    "softplus_beta": 1e-4,
    "softminus": True,
    "softminus_beta": 1e-3,
    "slope_winch_ri": 1000,
    "offset_winch_ri": -5,
}

REELIN_CONFIG = {
    "path_parameters": PATH_PARAMETERS,
    "radial_parameters": RADIAL_PARAMETERS,
    "sim_parameters": {
        "start_time": 0,
    },
}

AERO_INPUT_FILE = Path("data/LEI-V9-KITE/v9_aero_input.json")


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
    wind_model = build_wind_model()
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
    reelin = ReelinSimple(
        system_model=system_model,
        pattern_config=REELIN_CONFIG,
        depower_ri=1,
        depower_riro=1,
    )

    solution = reelin.run_simulation_opti()
    reelin.run_simulation(solution=solution, run_plots=run_plots)
    return reelin


if __name__ == "__main__":
    main(run_plots=True)

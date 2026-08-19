"""Standalone reel-in optimization (the "simple" reel-in).

Builds a ``ReelinSimple`` — which runs a pure reel-in phase followed by the
transition phase — simulates it once, then optimizes a small set of parameters
(here the transition start elevation) with the reel-in end radius constrained to
the target ``distance_radial_end``.

This is the reel-in-only counterpart of the reel-out pattern scripts; for the
full pumping cycle (reel-out + reel-in + transition) use
``scripts/reduced-order-model/optimization/cycle/run_cycle_simulation.py``.

Usage:
    python scripts/reduced-order-model/optimization/reelin/simple_reelin.py [--plot]
"""

import argparse

import numpy as np

from awetrim.environment.wind_factory import create_wind_model
from awetrim.system.factory import create_system_model_from_yaml
from awetrim.timeseries.reelin_phase import ReelinSimple
from awetrim.utils.config_paths import LEI_V3_SYSTEM_CONFIG

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
KITE_CONFIG_PATH = LEI_V3_SYSTEM_CONFIG

WIND_CONFIG = {
    "speed_wind_at_100": 8,
    "z0": 0.01,
    "model_type": "logarithmic",
}

REELIN_CONFIG = {
    "path_parameters": {
        "elevation_start_ri": np.radians(30),
        "elevation_start_riro": np.radians(70),
        "elevation_start_ro": np.radians(30),
        "distance_radial_start": 360,
        "distance_radial_end": 230,
    },
    "radial_parameters": {
        "reeling_strategy": "force",
        "force_model": "linear",
        "reeling_speed": 1.0,
        "max_tether_force": 8400.0,
        "min_tether_force": 2000.0,
        "softplus": True,
        "softplus_beta": 1e-4,
        "softminus": True,
        "softminus_beta": 1e-3,
        "slope_winch_ri": 1000,
        "offset_winch_ri": -2,
    },
    "sim_parameters": {"start_time": 0, "n_points_ri": 150, "n_points_riro": 100},
}

# Parameters optimized for the reel-in. Add e.g. "offset_winch_ri",
# "slope_winch_ri" to also tune the reel-in winch law.
OPTIMIZATION_PARAMS = ["elevation_start_riro", "offset_winch_ri"]


def build_wind_model(speed_wind_at_100=8, z0=0.01, model_type="logarithmic", **profile_kwargs):
    """Wind model with the reference speed given at 100 m.

    ``model_type`` is any analytic law of ``awetrim.environment.profile_laws``
    (uniform, logarithmic, power_law, explog, jet); extra keys such as
    ``alpha``, ``jet_amplitude``/``jet_height``/``jet_width`` or
    ``direction_wind`` are forwarded to ``create_wind_model``.
    """
    return create_wind_model(
        model_type,
        U_ref=speed_wind_at_100,
        z_ref=100.0,
        z0=z0,
        **profile_kwargs,
    )


def main(run_plots: bool = False):
    system_model = create_system_model_from_yaml(yaml_path=KITE_CONFIG_PATH)
    system_model.wind = build_wind_model(**WIND_CONFIG)

    reelin = ReelinSimple(
        system_model=system_model,
        pattern_config=REELIN_CONFIG,
        depower_ri=1.9,
        depower_riro=1.9,
    )

    # Baseline simulation with the initial configuration.
    reelin.run_simulation(run_plots=run_plots)

    # Optimize, then re-simulate with the optimized configuration.
    solution = reelin.run_simulation_opti(
        optimization_params=OPTIMIZATION_PARAMS, target="energy"
    )
    reelin.run_simulation(solution=solution, run_plots=run_plots)
    return reelin


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot", action="store_true", help="Show overview plots")
    args = parser.parse_args()
    main(run_plots=args.plot)
    if args.plot:
        import matplotlib.pyplot as plt

        plt.show()

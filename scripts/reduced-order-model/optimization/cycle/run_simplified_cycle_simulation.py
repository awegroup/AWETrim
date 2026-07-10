"""Run a full pumping cycle as a sequence of parametrized phases.

The cycle is composed of three parametrized phases stitched together by
``CycleSimple``:

    reel-out (``Phase``)  ->  reel-in + transition (``ReelinSimple``)

- ``Phase``        : one reel-out pattern. Here the pattern is ``spline_periodic``
                     (the periodic B-spline production loop), but it can be any
                     pattern_type (downloop / uploop / helix / ...).
- ``ReelinSimple`` : the *parametrized* reel-in, which itself runs two phases —
                     a pure reel-in (``reel_in_simple``) followed by the
                     transition back to reel-out (``transition_simple``).
- ``CycleSimple``  : composes the two, either as a sequential simulation
                     (``run_cycle_simulation``) or as one combined NLP
                     (``run_cycle_opti``).

For the *alternative* "single periodic spline" route — simulating the whole
production loop as one continuous spline rather than stitched phases — use the
reel-out ``Phase`` on its own with a ``spline_periodic`` pattern; see
``scripts/reduced-order-model/optimization/reelout/downloop_pattern.py``.

Usage:
    python scripts/reduced-order-model/optimization/cycle/run_cycle_simulation.py \
        [--plot] [--optimize] [--shape {downloop,uploop,helix}] [--figures N] \
        [--method {alternating,monolithic}]
"""

import argparse

import numpy as np

from awetrim.environment.Wind import Wind
from awetrim.identification.controls import ROM_DEPOWERED_INPUT_DEPOWER
from awetrim.system.factory import create_system_model_from_yaml
from awetrim.timeseries.phase import Phase
from awetrim.timeseries.reelin_phase import ReelinSimple
from awetrim.timeseries.cycle_phase import CycleSimple
from awetrim.utils.config_paths import (
    LEI_V3_DOWNLOOP_SPLINE_CONFIG,
    LEI_V3_HELIX_SPLINE_CONFIG,
    LEI_V3_SYSTEM_CONFIG,
    LEI_V3_UPLOOP_SPLINE_CONFIG,
)
from awetrim.utils.utils import load_cycle_config_from_yaml

# ---------------------------------------------------------------------------
# Configuration files
# ---------------------------------------------------------------------------
KITE_CONFIG_PATH = LEI_V3_SYSTEM_CONFIG

# Reel-out shape -> cycle config. Each config's "reelout" section is a
# spline_periodic production loop; the "reelin" section drives ReelinSimple.
SHAPE_CONFIGS = {
    "downloop": LEI_V3_DOWNLOOP_SPLINE_CONFIG,
    "uploop": LEI_V3_UPLOOP_SPLINE_CONFIG,
    "helix": LEI_V3_HELIX_SPLINE_CONFIG,
}
DEFAULT_SHAPE = "downloop"
DEFAULT_N_FIGURES = 4

WIND_CONFIG = {
    "speed_wind_at_100": 10,
    "z0": 0.03,
    "model_type": "logarithmic",
}

# Parameters optimized when --optimize is passed (combined reel-out + reel-in NLP).
CYCLE_OPTIMIZATION_PARAMS = [
    "elevation_start_riro",
    # "input_depower_ri",
    "offset_winch_ri",
    "slope_winch_ri",
    # "input_depower_ri",
    "slope_winch_ro",
    # "offset_winch_ro",
    "C_phi",
    "C_beta",
    "input_depower_ro",
]


def build_wind_model(speed_wind_at_100=8, z0=0.01, model_type="uniform"):
    """Create a wind model using the supplied parameters."""
    wind_model = Wind(wind_model=model_type, z0=z0, direction_wind=0)
    speed_friction = 0.41 * speed_wind_at_100 / np.log(100 / wind_model.z0)
    if model_type == "logarithmic":
        wind_model.speed_friction = speed_friction
    elif model_type == "uniform":
        wind_model.speed_wind_ref = speed_wind_at_100
    return wind_model


def build_reelout_start_state(reelout_config):
    """Initial state for the reel-out phase, started at the spline radius r0."""
    return {
        "t": 0,
        "s": 0,
        "s_dot": 2,
        "input_steering": 0,
        "tension_tether_ground": 8.4e5,  # initial tension guess (N)
        "speed_radial": 0,  # positive for reel-out
        "distance_radial": reelout_config["path_parameters"]["r0"],
    }


def set_number_of_figures(reelout_config, n_figures):
    """Make the reel-out fly ``n_figures`` repetitions of the periodic figure.

    One figure spans ``(s_final - s_init)`` of the path parameter. The phase grid
    is extended to cover ``n_figures`` periods and the point count is scaled to
    keep the per-figure resolution. The periodic spline wraps the phase into a
    single period, so the figure simply repeats.
    """
    path = reelout_config["path_parameters"]
    sim = reelout_config["sim_parameters"]
    period = float(path.get("s_final", 1.0)) - float(path.get("s_init", 0.0))
    start = float(sim.get("start_angle", 0.0))
    points_per_figure = int(sim.get("n_points", 50))
    sim["start_angle"] = start
    sim["end_angle"] = start + n_figures * period
    sim["n_points"] = points_per_figure * n_figures
    return reelout_config


def main(
    run_plots: bool = False,
    optimize: bool = False,
    shape: str = DEFAULT_SHAPE,
    n_figures: int = DEFAULT_N_FIGURES,
    method: str = "alternating",
) -> int:
    if shape not in SHAPE_CONFIGS:
        raise ValueError(
            f"Unknown shape {shape!r}. Choose from {sorted(SHAPE_CONFIGS)}."
        )
    print(f"Reel-out shape: {shape}, figures per reel-out: {n_figures}")
    reelout_config, reelin_config = load_cycle_config_from_yaml(SHAPE_CONFIGS[shape])
    set_number_of_figures(reelout_config, n_figures)

    system_model = create_system_model_from_yaml(yaml_path=KITE_CONFIG_PATH)
    system_model.wind = build_wind_model(
        speed_wind_at_100=WIND_CONFIG["speed_wind_at_100"],
        z0=WIND_CONFIG["z0"],
        model_type=WIND_CONFIG["model_type"],
    )

    # Reel-out runs powered (l_dp = 1.7 m), set via
    # reelout_config["sim_parameters"]["input_depower"]; reel-in/transition are
    # flown fully depowered (l_dp = ROM_DEPOWERED_INPUT_DEPOWER = 2.1 m).
    # reelout_config["sim_parameters"]["expand_nlp"] = True
    reelout = Phase(
        system_model=system_model,
        pattern_config=reelout_config,
        start_state=build_reelout_start_state(reelout_config),
    )
    reelin = ReelinSimple(
        system_model=system_model,
        pattern_config=reelin_config,
        depower_ri=ROM_DEPOWERED_INPUT_DEPOWER,
        depower_riro=ROM_DEPOWERED_INPUT_DEPOWER,
    )
    cycle = CycleSimple(reelin=reelin, reelout=reelout)

    print("Running cycle simulation: reel-out -> reel-in -> transition")
    result = cycle.run_cycle_simulation(optimize_reelin=False, plotting=run_plots)
    if result is None:
        print("Cycle simulation failed")
        return 1

    if optimize:
        if method == "alternating":
            print("\nOptimizing cycle (alternating reel-out / reel-in)...")
            cycle.run_cycle_alternating()
        else:
            print("\nOptimizing combined cycle (monolithic reel-out + reel-in)...")
            cycle.run_cycle_opti(optimization_params=CYCLE_OPTIMIZATION_PARAMS)
        cycle.run_cycle_simulation(optimize_reelin=False, plotting=run_plots)

    print("Cycle simulation complete")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot", action="store_true", help="Show combined plots")
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Run the combined reel-out + reel-in cycle optimization",
    )
    parser.add_argument(
        "--shape",
        choices=sorted(SHAPE_CONFIGS),
        default=DEFAULT_SHAPE,
        help="Reel-out figure shape",
    )
    parser.add_argument(
        "--figures",
        type=int,
        default=DEFAULT_N_FIGURES,
        help="Number of figures flown during reel-out",
    )
    parser.add_argument(
        "--method",
        choices=["alternating", "monolithic"],
        default="monolithic",
        help="Cycle optimization strategy (with --optimize)",
    )
    args = parser.parse_args()
    exit_code = main(
        run_plots=args.plot,
        optimize=args.optimize,
        shape=args.shape,
        n_figures=args.figures,
        method=args.method,
    )
    if args.plot:
        import matplotlib.pyplot as plt

        plt.show()
    raise SystemExit(exit_code)

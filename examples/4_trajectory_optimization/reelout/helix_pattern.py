import argparse
import json
from pathlib import Path

import numpy as np

from awetrim import SystemModel
from awetrim.environment.wind_factory import create_wind_model
from awetrim.system.kite import Kite
from awetrim.system.tether import RigidLumpedTether
from awetrim.timeseries.phase import Phase
from awetrim.system.factory import create_system_model_from_yaml
from awetrim.kinematics.parametrized_patterns import (
    make_bspline_path_parameters_from_named_curve,
)
from awetrim.timeseries.seed_sizing import SEED_TURN_MARGINS, size_reelout_seed
from awetrim.utils.utils import load_cycle_config_from_yaml
from reelout_kites import (
    DEFAULT_KITE,
    apply_flight_depower,
    kite_dir,
    pattern_config_path,
    seed_overrides,
)

"""
Reelout pattern simulation using YAML configuration files.

Everything about the kite comes from its data folder (``--kite``, default
LEI-V3-KITE; see reelout_kites.py):
- System properties: data/<kite>/system.yaml
- Cycle parameters: data/<kite>/cycle_configs/helix_spline.yaml

To modify parameters:
1. Edit the YAML files directly, or
2. Load configs and override specific values in this script
"""

# ---------------------------------------------------------------------------
# Configuration files
# ---------------------------------------------------------------------------
CYCLE_CONFIG_FILENAME = "helix_spline.yaml"

SAVE_TIMESERIES = True
SEED_MARGIN = SEED_TURN_MARGINS[-1]  # over the steering-limited turn radius

# ---------------------------------------------------------------------------
# Initial path guess
# ---------------------------------------------------------------------------
# Regenerate the reel-out B-spline control points from a simple analytic curve
# and override the ones loaded from YAML. Set REGENERATE_INITIAL_GUESS = False
# to fly the control points stored in the config instead. This is the inline
# equivalent of running generate_spline_config.py before this script.
REGENERATE_INITIAL_GUESS = True
INITIAL_GUESS = {
    "curve_type": "helix",  # helix shape (use a *_pattern script for eights)
    "M": 10,
    "n_fit": 400,
    "s_init": 0.0,
    "s_final": 2.0 * np.pi,
    "az_amp0": 0.1,
    "beta0": 0.35,
    "beta_amp0": 0.1,
    "downloops": True,
}



def load_reelout_config(folder):
    """The kite's helix phase config with the seed regenerated (INITIAL_GUESS at
    the config's r0, overridden by its ``example: seed_curve:``) and the seed
    depower."""
    config_path = pattern_config_path(folder, CYCLE_CONFIG_FILENAME)
    reelout_config, _ = load_cycle_config_from_yaml(config_path)
    if REGENERATE_INITIAL_GUESS:
        guess = {"r0": reelout_config["path_parameters"]["r0"], **INITIAL_GUESS,
                 **seed_overrides(config_path)}
        reelout_config["path_parameters"] = make_bspline_path_parameters_from_named_curve(
            spline_type="periodic", **guess)
        reelout_config["sim_parameters"]["start_angle"] = guess["s_init"]
        reelout_config["sim_parameters"]["end_angle"] = guess["s_final"]
    # the kite's matched reel-out tape (identification.yaml flight_match), if any
    apply_flight_depower(folder, reelout_config["sim_parameters"])
    return reelout_config


WIND_CONFIG = {
    "speed_wind_at_100": 10,
    "z0": 0.03,
    "model_type": "logarithmic",
}
START_STATE = {
    "t": 0,
    "s": 0,
    "s_dot": 3,
    "input_steering": 0,
    "tension_tether_ground": 8.4e5,  # Initial guess for tension (N)
    "speed_radial": 0,  # Positive for reel-out
}


def build_wind_model(
    speed_wind_at_100=8, z0=0.01, model_type="uniform", direction_wind=0, **profile_kwargs
):
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
        direction_wind=direction_wind,
        **profile_kwargs,
    )


def main(run_plots=False, kite=None):
    """``kite``: a folder under data/ or a path to one (default LEI-V3-KITE);
    results go to results/<kite>/optimization/helix."""
    folder = kite_dir(kite)
    results_dir = Path("results") / folder.name / "optimization" / "helix"
    REELOUT_CONFIG = load_reelout_config(folder)
    START_STATE["distance_radial"] = REELOUT_CONFIG["path_parameters"]["r0"]
    system_model = create_system_model_from_yaml(yaml_path=folder / "system.yaml")

    wind_model = build_wind_model(**WIND_CONFIG)
    system_model.wind = wind_model
    # The kite must hold the helix's turns at this wind (one circle: no crossing).
    size_reelout_seed(system_model, REELOUT_CONFIG, margin=SEED_MARGIN, mirror=False,
                      self_crossings=0)
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
    # re-solved while the step box (param_step_bound) binds at the optimum
    solution = reelout.run_simulation_opti_recentred(optimization_params=optimization_params)
    if solution is None:
        raise SystemExit("The optimization did not converge; nothing written.")
    depower_prefix = "depower_" if "input_depower" in optimization_params else ""
    filename = f"{depower_prefix}helix_optimized_config_wind_{WIND_CONFIG['speed_wind_at_100']}_z0_{WIND_CONFIG['z0']}_{WIND_CONFIG['model_type']}_spline.yaml"

    solution.save_config_to_yaml(results_dir / filename)
    phase, _ = reelout.run_simulation(run_plots=True, axes=axes, phase_sim=True)
    print(phase.energy_metrics())

    if SAVE_TIMESERIES:
        phase.save_timeseries_csv(
            results_dir / filename.replace(".yaml", "_timeseries.csv")
        )

    return reelout


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description="Helix reel-out optimisation")
    parser.add_argument("--kite", default=DEFAULT_KITE,
                        help="folder under data/ (or a path to one) holding system.yaml and "
                        f"cycle_configs/{CYCLE_CONFIG_FILENAME} (default {DEFAULT_KITE})")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    if args.no_show:
        plt.switch_backend("Agg")
    main(run_plots=not args.no_show, kite=args.kite)
    if not args.no_show:
        plt.show()

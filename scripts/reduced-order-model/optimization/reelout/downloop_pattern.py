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
from awetrim.identification.controls import ROM_POWERED_INPUT_DEPOWER
from awetrim.utils.config_paths import (
    LEI_V3_DOWNLOOP_SPLINE_CONFIG,
    LEI_V3_ROM_AEROSTRUCTURAL_CONFIG,
    LEI_V3_ROM_AEROSTRUCTURAL_FLIGHT_CONFIG,
    LEI_V3_ROM_SEMI_EMPIRICAL_CONFIG,
    LEI_V3_SYSTEM_CONFIG,
    LEI_V3_SYSTEM_FLOWN_2019_CONFIG,
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
#: --rom choices (default: the one the system file selects).
ROM_CONFIGS = {
    "semi_empirical": LEI_V3_ROM_SEMI_EMPIRICAL_CONFIG,
    "aerostructural": LEI_V3_ROM_AEROSTRUCTURAL_CONFIG,
    "aerostructural_flight": LEI_V3_ROM_AEROSTRUCTURAL_FLIGHT_CONFIG,
}
#: --system choices: the optimisation hardware or the 2019 flown one.
SYSTEM_CONFIGS = {
    "optimisation": LEI_V3_SYSTEM_CONFIG,
    "flown_2019": LEI_V3_SYSTEM_FLOWN_2019_CONFIG,
}

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


def build_wind_model(speed_wind_at_100=8, z0=0.01, model_type="uniform", **profile_kwargs):
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


def main(run_plots=False, rom=None, system="optimisation", results_dir=None,
         up_band=None, stall_margin_deg=None):
    """``rom``/``system``: keys of ROM_CONFIGS / SYSTEM_CONFIGS (None = the
    system file's own ROM); results go to ``results_dir`` (default
    RESULTS_DIR, or RESULTS_DIR/<system>_<rom> when either is given)."""
    if results_dir is None:
        results_dir = (
            RESULTS_DIR / f"{system}_{rom or 'default'}"
            if rom is not None or system != "optimisation" else RESULTS_DIR
        )
    if up_band is not None:
        # Keep the optimised depower inside the band the ROM was identified on
        # (a polynomial ROM extrapolates freely in u_p otherwise).
        REELOUT_CONFIG["sim_parameters"].setdefault("opti_limits_override", {})[
            "input_depower"] = [float(up_band[0]), float(up_band[1])]
    if stall_margin_deg is not None:
        # Keep alpha this far below the ROM's stall angle at every node.
        REELOUT_CONFIG["sim_parameters"]["stall_margin_deg"] = float(stall_margin_deg)
    system_model = create_system_model_from_yaml(
        yaml_path=SYSTEM_CONFIGS[system],
        aero_yaml_path=ROM_CONFIGS[rom] if rom is not None else None,
    )

    wind_model = build_wind_model(**WIND_CONFIG)
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

    solution.save_config_to_yaml(results_dir / filename)
    phase, _ = reelout.run_simulation(run_plots=True, axes=axes, phase_sim=True)
    print(phase.energy_metrics())

    if SAVE_TIMESERIES:
        phase.save_timeseries_csv(
            results_dir / filename.replace(".yaml", "_timeseries.csv")
        )
        metrics = {k: (float(v) if np.isscalar(v) else v)
                   for k, v in dict(phase.energy_metrics()).items()}
        (results_dir / filename.replace(".yaml", "_metrics.json")).write_text(
            json.dumps({"rom": rom, "system": system, **metrics}, indent=2, default=str),
            encoding="utf-8",
        )

    return reelout


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description="LEI-V3 downloop reel-out optimisation")
    parser.add_argument("--rom", choices=sorted(ROM_CONFIGS), default=None)
    parser.add_argument("--system", choices=sorted(SYSTEM_CONFIGS), default="optimisation")
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--up-band", nargs=2, type=float, default=None,
                        metavar=("LO", "HI"),
                        help="bound the optimised power-tape length [m] to the "
                        "ROM's identified band (default: DEFAULT_OPTI_LIMITS)")
    parser.add_argument("--stall-margin", type=float, default=None, metavar="DEG",
                        help="keep alpha this many degrees below the ROM's stall "
                        "angle throughout the loop")
    args = parser.parse_args()
    if args.no_show:
        plt.switch_backend("Agg")
    main(run_plots=not args.no_show, rom=args.rom, system=args.system,
         up_band=args.up_band, stall_margin_deg=args.stall_margin)
    if not args.no_show:
        plt.show()

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from awetrim import SystemModel
from awetrim.environment.wind_factory import create_wind_model
from awetrim.system.kite import Kite
from awetrim.system.tether import RigidLumpedTether
from awetrim.timeseries.phase import Phase
from awetrim.timeseries.seed_sizing import size_reelout_seed
from awetrim.system.factory import create_system_model_from_yaml
from awetrim.kinematics.parametrized_patterns import (
    make_bspline_path_parameters_from_named_curve,
)
from awetrim.utils.utils import load_cycle_config_from_yaml
from reelout_kites import (
    DEFAULT_KITE,
    FILE_ALIASES,
    apply_flight_depower,
    kite_dir,
    kite_file,
    pattern_config_path,
    seed_overrides,
)

"""
Reelout pattern simulation using YAML configuration files.

Everything about the kite comes from its data folder (``--kite``, default
LEI-V3-KITE; see reelout_kites.py):
- System properties: data/<kite>/system.yaml (``--system``)
- Cycle parameters: data/<kite>/cycle_configs/downloop_spline.yaml

To modify parameters:
1. Edit the YAML files directly, or
2. Load configs and override specific values in this script
"""

# ---------------------------------------------------------------------------
# Configuration files
# ---------------------------------------------------------------------------
CYCLE_CONFIG_FILENAME = "downloop_spline.yaml"

SAVE_TIMESERIES = True

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
    # the figure generate_spline_config.py defaults to: inside the ROM's
    # steering band (0.30 / 0.12 rad at 200 m needed more than the kite has)
    "az_amp0": 0.65,
    "beta0": 0.4,
    "beta_amp0": 0.30,
    "downloops": True,  # downloop direction
}

INITIAL_GUESS_R0 = 300.0  # [m] the radius that figure is sized for
SERVER_SEED_R0 = 200.0  # [m] --server-seed: examples/server/client_example.py's length


def load_reelout_config(folder):
    """The kite's downloop phase config with the seed regenerated (INITIAL_GUESS,
    overridden by the config's ``example: seed_curve:``) and the seed depower."""
    config_path = pattern_config_path(folder, CYCLE_CONFIG_FILENAME)
    reelout_config, _ = load_cycle_config_from_yaml(config_path)
    if REGENERATE_INITIAL_GUESS:
        guess = {"r0": INITIAL_GUESS_R0, **INITIAL_GUESS, **seed_overrides(config_path)}
        reelout_config["path_parameters"] = make_bspline_path_parameters_from_named_curve(
            spline_type="periodic", **guess)
        reelout_config["sim_parameters"]["start_angle"] = guess["s_init"]
        reelout_config["sim_parameters"]["end_angle"] = guess["s_final"]
    sim = reelout_config["sim_parameters"]
    sim["n_points"] = 100
    # the kite's matched reel-out tape (identification.yaml flight_match), if any
    apply_flight_depower(folder, sim)
    sim["reg_weight"] = 1.0
    sim["detect_simple_bounds"] = True
    return reelout_config


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
         up_band=None, stall_margin_deg=None, rom_file=None, server_seed=False,
         seed_config=None, kite=None):
    """``kite``: a folder under data/ or a path to one (default LEI-V3-KITE).
    ``rom``/``system``: an alias of reelout_kites.FILE_ALIASES or a file in
    the kite folder (rom None = the system file's own ROM); results go to
    ``results_dir`` (default results/<kite>/optimization/downloops, in a
    <system>_<rom> subfolder when either is given)."""
    folder = kite_dir(kite)
    if results_dir is None:
        results_dir = Path("results") / folder.name / "optimization" / "downloops"
        if rom is not None or system != "optimisation":
            results_dir = results_dir / f"{system}_{rom or 'default'}"
    reelout_config, start_state = load_reelout_config(folder), dict(START_STATE)
    start_state["distance_radial"] = reelout_config["path_parameters"]["r0"]
    if server_seed:
        # The reel-out server's default seed: its InitialGuess eight (smaller and
        # higher than INITIAL_GUESS) at the client example's 200 m tether.
        from awetrim.server.schemas import InitialGuess

        guess = InitialGuess().model_dump()
        reelout_config = copy.deepcopy(reelout_config)
        reelout_config["path_parameters"] = make_bspline_path_parameters_from_named_curve(
            spline_type="periodic", r0=SERVER_SEED_R0, **guess)
        reelout_config["sim_parameters"]["start_angle"] = guess["s_init"]
        reelout_config["sim_parameters"]["end_angle"] = guess["s_final"]
        start_state["distance_radial"] = SERVER_SEED_R0
    if seed_config is not None:
        # A previous run's optimised config (e.g. another ROM's optimum) as the
        # seed, flown as it is: no seed sizing below.
        import yaml

        reelout_config = yaml.safe_load(Path(seed_config).read_text(encoding="utf-8"))["reelout"]
        start_state["distance_radial"] = reelout_config["path_parameters"]["r0"]
    if up_band is not None:
        # Keep the optimised depower inside the band the ROM was identified on
        # (a polynomial ROM extrapolates freely in u_p otherwise).
        reelout_config["sim_parameters"].setdefault("opti_limits_override", {})[
            "input_depower"] = [float(up_band[0]), float(up_band[1])]
    if stall_margin_deg is not None:
        # Keep alpha this far below the ROM's stall angle at every node.
        reelout_config["sim_parameters"]["stall_margin_deg"] = float(stall_margin_deg)
    system_model = create_system_model_from_yaml(
        yaml_path=kite_file(folder, system),
        aero_yaml_path=(rom_file if rom_file is not None
                        else kite_file(folder, rom) if rom is not None else None),
    )

    wind_model = build_wind_model(**WIND_CONFIG)
    system_model.wind = wind_model
    # The kite must hold the eight's turns at this wind.
    if seed_config is None:
        size_reelout_seed(system_model, reelout_config)
    reelout = Phase(
        system_model=system_model,
        pattern_config=reelout_config,
        start_state=start_state,
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
    # re-solved while the step box (param_step_bound) binds at the optimum
    solution = reelout.run_simulation_opti_recentred(optimization_params=optimization_params)
    if solution is None:
        raise SystemExit("The optimization did not converge; nothing written.")
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
            json.dumps({"kite": folder.name, "rom": rom, "system": system, **metrics},
                       indent=2, default=str),
            encoding="utf-8",
        )

    return reelout


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description="Downloop reel-out optimisation")
    parser.add_argument("--kite", default=DEFAULT_KITE,
                        help="folder under data/ (or a path to one) holding system.yaml and "
                        f"cycle_configs/{CYCLE_CONFIG_FILENAME} (default {DEFAULT_KITE})")
    parser.add_argument("--rom", default=None,
                        help="ROM config: a file in the kite folder, or aerostructural / "
                        "aerostructural_flight (default: the one system.yaml selects)")
    parser.add_argument("--system", default="optimisation",
                        help="system file: a file in the kite folder, or optimisation "
                        "(system.yaml, default) / flown_2019")
    parser.add_argument("--rom-file", type=Path, default=None,
                        help="any ROM file (wins over --rom)")
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--seed-config", type=Path, default=None,
                        help="seed with a previous run's optimised config (its reelout "
                        "block), flown as it is")
    parser.add_argument("--server-seed", action="store_true",
                        help="seed with the reel-out server's default InitialGuess eight "
                        "at SERVER_SEED_R0 instead of INITIAL_GUESS")
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
         up_band=args.up_band, stall_margin_deg=args.stall_margin,
         rom_file=args.rom_file, results_dir=args.results_dir,
         server_seed=args.server_seed,
         seed_config=args.seed_config, kite=args.kite)
    if not args.no_show:
        plt.show()

"""Run a cycle simulation (reel-out then reel-in) using existing configs.

This script loads the configuration and system construction helpers from
`reelout_new.py` and `simple_reelin.py` (located in the same folder) and runs
an end-to-end simulation where:

1. The reel-out phase is simulated to determine the final radial distance.
2. That final distance becomes the start distance for the reel-in phase.
3. The reel-in phase is optionally optimized (single parameter) and simulated.
4. Optionally plot both phases together.

Usage:
    python run_cycle_simulation.py [--plot]

"""

import matplotlib.pyplot as plt
from pathlib import Path
import importlib.util
import argparse

# helper to dynamically import the script modules by path
WIND_CONFIG = {
    "speed_wind_at_100": 12,
    "z0": 0.05,
    "model_type": "uniform",
}


def load_module_from_path(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore
    return module


def main(run_plots: bool = False):
    repo_root = Path(__file__).resolve().parents[2]
    scripts_dir = repo_root / "scripts" / "optimize_path"

    reelout_path = scripts_dir / "reelout_new.py"
    reelin_path = scripts_dir / "simple_reelin.py"

    reelout_mod = load_module_from_path(reelout_path)
    reelin_mod = load_module_from_path(reelin_path)

    # Build a single wind and system model here (do it once and reuse)
    # Use the helper functions and config from the reelout script to assemble
    # the components so both phases share the exact same SystemModel and wind.
    aero_input = reelout_mod.load_aero_input()
    # build_wind_model accepts optional args; use defaults from reelout script
    wind_model = reelout_mod.build_wind_model(
        speed_wind_at_100=WIND_CONFIG["speed_wind_at_100"],
        z0=WIND_CONFIG["z0"],
        model_type=WIND_CONFIG["model_type"],
    )
    system_model = reelout_mod.define_system(
        tether_diameter=reelout_mod.PHYSICAL_CONFIG["tether_diameter"],
        mass_wing=reelout_mod.PHYSICAL_CONFIG["mass_wing"],
        mass_kcu=reelout_mod.PHYSICAL_CONFIG["mass_kcu"],
        area_wing=reelout_mod.PHYSICAL_CONFIG["area_wing"],
        aero_input=aero_input,
        wind_model=wind_model,
    )

    # Import classes
    from awetrim.timeseries.reelout_phase import Reelout
    from awetrim.timeseries.reelin_phase import ReelinSimple
    from awetrim.timeseries.cycle_phase import CycleSimple

    optimization_params = [
        "az_amp0",
        "beta_amp0",
        "beta0",
        "elevation_start_riro",
        "offset_winch_ri",
        # "slope_winch_ri",
        # "offset_winch_ro",
        "slope_winch_ro",
        "beta_coeffs",
        # "kappa",
    ]
    # Instantiate phases using the configs defined in the scripts
    reelout = Reelout(
        system_model=system_model,
        pattern_config=reelout_mod.REELOUT_CONFIG,
        depower=0,
    )

    # reelout.run_simulation_opti(
    #     optimization_params=optimization_params, target="energy"
    # )
    reelin = ReelinSimple(
        system_model=system_model,
        pattern_config=reelin_mod.REELIN_CONFIG,
        depower_ri=getattr(reelin_mod, "REELIN_CONFIG", {}).get("depower_ri", 1),
        depower_riro=getattr(reelin_mod, "REELIN_CONFIG", {}).get("depower_riro", 1),
    )

    cycle = CycleSimple(reelin=reelin, reelout=reelout)

    print("Running cycle simulation: reel-out -> (opt) reel-in -> transition")
    # result = cycle.run_cycle_simulation(optimize_reelin=True, plotting=run_plots)
    plt.show()
    cycle.run_cycle_opti(optimization_params=optimization_params)
    print(cycle.reelin.pattern_config)
    result = cycle.run_cycle_simulation(optimize_reelin=False, plotting=run_plots)
    plt.show()
    if result is None:
        print("Cycle simulation failed")
        return 1

    print("Cycle simulation complete")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true", help="Show combined plots")
    args = parser.parse_args()
    main(run_plots=True)

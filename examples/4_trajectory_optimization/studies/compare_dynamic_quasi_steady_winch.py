# Copyright (c) 2023-2026 Oriol Cayon, Delft University of Technology
# SPDX-License-Identifier: Apache-2.0
"""Quasi-steady vs dynamic forward simulation, with tether inertia and winch friction.

Marches one path with its winch force law (no optimisation) under these models:

==========  ==========  ==============  =============================
name        kite        tether inertia  winch
==========  ==========  ==============  =============================
qs          QS          no              ideal
qs_ti       QS          yes             ideal
dyn_ti      dynamic     yes             drum inertia
qs_ti_f     QS          yes             friction
dyn_ti_f    dynamic     yes             drum inertia + friction
qs_f        QS          no              friction
qs_drum     QS          no              drum inertia + friction
dyn         dynamic     no              drum inertia + friction
==========  ==========  ==============  =============================

QS = quasi-steady: no kite, drum or tether accelerations (the tether's
centrifugal, Coriolis and turn terms remain). Dynamic = implicit Euler over the
march's node steps; the drum and the kite share dv_r/dt. The drivetrain and the
tether are the system file's (system.yaml drums[0], tethers[0]). Cases:
``eight``, one reel-out figure eight (cycle_configs/downloop_spline.yaml, sized
to the kite at the wind), and ``cycle``, a whole pumping cycle (an optimized
full-cycle YAML, ``--cycle-config``).

Usage (from the repository root):
    python examples/4_trajectory_optimization/studies/compare_dynamic_quasi_steady_winch.py \\
        --cycle-config results/LEI-V3-KITE/optimization/full_cycle/<optimized>.yaml \\
        --wind-speed 9 --wind-height 6
"""

import argparse
import copy
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "examples/4_trajectory_optimization/cycle"))
import run_full_cycle_opti as rfc  # noqa: E402

from awetrim.system.factory import create_system_model_from_yaml  # noqa: E402
from awetrim.timeseries.phase import Phase  # noqa: E402
from awetrim.timeseries.seed_sizing import size_reelout_seed  # noqa: E402

FRICTION = {"winch_friction": True}
DRUM = {"winch_inertia_mass": True}
# name: (quasi_steady, sim_parameters overrides, tether inertia)
MODELS = {
    "qs": (True, {}, False),
    "qs_ti": (True, {}, True),
    "dyn_ti": (False, DRUM, True),
    "qs_ti_f": (True, FRICTION, True),
    "dyn_ti_f": (False, {**DRUM, **FRICTION}, True),
    "qs_f": (True, FRICTION, False),
    "qs_drum": (True, {**DRUM, **FRICTION}, False),
    "dyn": (False, {**DRUM, **FRICTION}, False),
}
DEFAULT_MODELS = ["qs", "qs_ti", "dyn_ti", "qs_ti_f", "dyn_ti_f"]


def new_system_model(tether_inertia=False):
    system_model = create_system_model_from_yaml(
        yaml_path=rfc.KITE_CONFIG_PATH,
        tether_config={"model": "rigid_lumped", "inertia": tether_inertia},
    )
    system_model.wind = rfc.build_wind_model(**rfc.WIND_CONFIG)
    return system_model


def simulate(reelout_config, quasi_steady, sim_overrides, tether_inertia):
    """Forward march of one config; returns the phase's DataFrame."""
    cfg = copy.deepcopy(reelout_config)
    cfg["sim_parameters"].update(sim_overrides)
    phase = Phase(
        system_model=new_system_model(tether_inertia),
        pattern_config=cfg,
        start_state=rfc.build_start_state(cfg),
        quasi_steady=quasi_steady,
    )
    result, _ = phase.run_simulation(phase_sim=True)
    return result.to_dataframe()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--kite", default=rfc.DEFAULT_KITE)
    parser.add_argument("--cycle-config", required=True, help="optimized full-cycle YAML")
    parser.add_argument("--eight-config", default=None,
                        help="figure-eight YAML (default: the kite's cycle_configs/downloop_spline.yaml)")
    parser.add_argument("--wind-speed", type=float, default=None)
    parser.add_argument("--wind-height", type=float, default=None)
    parser.add_argument("--cases", nargs="+", default=["eight", "cycle"])
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS, choices=list(MODELS))
    parser.add_argument(
        "--eight-margin", type=float, default=1.6,
        help="turn-radius margin the eight is sized with (default 1.6): at the "
        "seed sizing's 1.2 the full dynamic kite with the drum's inertia cannot "
        "follow it (its steering runs out mid-turn, a fold of the node equations)",
    )
    parser.add_argument("--out", default=None, help="output folder (default results/<kite>/winch_dynamics)")
    args = parser.parse_args()

    kite = rfc.configure_kite(args.kite)
    rfc.WIND_CONFIG = rfc.override_wind(rfc.WIND_CONFIG, args.wind_speed, args.wind_height)
    out = Path(args.out) if args.out else rfc.RESULTS_DIR.parents[1] / "winch_dynamics"
    out.mkdir(parents=True, exist_ok=True)
    eight = args.eight_config or Path(kite["cycle_config_dir"]) / "downloop_spline.yaml"
    configs = {"eight": eight, "cycle": args.cycle_config}
    print(f"Wind: {rfc.describe_wind()}; output: {out}")

    for case in args.cases:
        reelout = rfc.load_reelout_config(configs[case])
        reelout["sim_parameters"].setdefault("slow_path_rate", True)
        # Keep a march that stops early: where the dynamic kite can no longer
        # follow the path is a result here, not an error.
        reelout["sim_parameters"]["require_full_trajectory"] = False
        if case == "eight":
            # The stored eight is sized for another wind: fit it to the turns
            # the kite holds here (the server's kite-sized seed).
            resized = size_reelout_seed(
                new_system_model(), reelout, margin=args.eight_margin
            )
            print(f"eight refitted to the kite at this wind: {resized}")
        for name in args.models:
            quasi_steady, overrides, tether_inertia = MODELS[name]
            print(f"\n=== {case} / {name} ===")
            try:
                df = simulate(reelout, quasi_steady, overrides, tether_inertia)
            except Exception as exc:  # report and go on with the other models
                print(f"FAILED: {exc}")
                continue
            path = out / f"{case}_{name}.csv"
            df.to_csv(path, index=False)
            t = df["t"].to_numpy()
            p = (df["tension_tether_ground"] * df["speed_radial"]).to_numpy()
            print(f"saved {path}: {len(df)} nodes, {t[-1] - t[0]:.1f} s, "
                  f"mean tether power {np.trapezoid(p, t) / (t[-1] - t[0]) / 1e3:.2f} kW")


if __name__ == "__main__":
    main()

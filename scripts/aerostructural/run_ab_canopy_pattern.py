"""A/B the canopy triangulation on the unsteered kite: does it stay symmetric?

The unactuated kite at the centre of the wind window, with gravity off, is
mirror-symmetric in every input. Its solved shape must therefore be mirror-
symmetric too, and any left-right mismatch it develops is model error.

A constant-strain triangle mesh built by splitting every quad along the *same*
diagonal is not mirror-symmetric: under ``y -> -y`` that diagonal maps to the
other one. On a finely resolved mesh the resulting bias is small; here the quads
are a sizeable fraction of the wing, so it is not. This script measures it.

Three patterns, same everything else --- see ``structural_billow.canopy_mesh``:

    diagonal   two triangles per quad on one diagonal (historical, biased)
    union      both diagonals at half stress resultant, no extra nodes
    cross      a node per quad centre, four triangles, and the only one that
               lets a quad dome rather than merely fold

Usage (from project root):
    python scripts/aerostructural/run_ab_canopy_pattern.py
"""

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np

from awetrim.aerostructural.logging_config import *  # noqa: F401,F403
from awetrim.aerostructural.results import aerostructural_results_root, save_sim_output
from awetrim.aerostructural.billow import structural_billow as sb
from awetrim.aerostructural.fem import read_struc_geometry_yaml
from common import DEFAULT_KITE_NAME

from run_chain_depower_BILLOW import DEPOWER_QUADRATIC, build_once, run_chain, span_metrics

PATTERNS = ("diagonal", "union", "cross")


def built_tape_length(shared):
    """Depower tape rest length [m] as the geometry stores it."""
    # The reader mutates its input, so hand it a copy.
    reader = read_struc_geometry_yaml.main(
        copy.deepcopy(shared["struc_geometry"]),
        config=shared["config"],
        system_config=shared["system_config"],
    )
    power_tape_index, l0_arr = reader[4], reader[13]
    return float(l0_arr[power_tape_index])


def tape_length(u_dp):
    """The kite's depower calibration, u_dp -> tape length [m]."""
    a, b, c = DEPOWER_QUADRATIC
    return a * u_dp**2 + b * u_dp + c


def mirror_mismatch(positions, grid):
    """Left-right mismatch of the wing grid [m], mirroring about ``y = 0``.

    Section ``i`` is compared with section ``n-1-i`` reflected in the centre
    plane, station by station. Zero is the correct answer for this load case.
    """
    mirror = np.array([1.0, -1.0, 1.0])
    rows, columns = grid.shape
    values = [
        float(np.linalg.norm(positions[grid[i, k]] - positions[grid[rows - 1 - i, k]] * mirror))
        for i in range(rows // 2)
        for k in range(columns)
    ]
    return np.asarray(values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wind", type=float, default=4.2)
    parser.add_argument("--panels-per-section", type=int, default=2)
    parser.add_argument("--tol", type=float, default=0.5,
                        help="coupled residual gate [N]. The comparison is only "
                             "fair if every pattern converges to the SAME level: "
                             "at the shipped 5 N gate they stopped at 0.85, 3.23 "
                             "and 3.31 N, and a looser state shows more residual "
                             "asymmetry regardless of the mesh.")
    parser.add_argument("--max-iter", type=int, default=60)
    parser.add_argument("--stagnation-tol", type=float, default=None,
                        help="coupled stagnation tolerance [N]: the run stops "
                             "as 'stagnated' when the residual moves less than "
                             "this over n_max_constant_residual_force "
                             "iterations. Default: a tenth of --tol. as_config's "
                             "absolute 2 N is larger than the whole remaining "
                             "gap to a 0.5 N gate, so it declared a steadily "
                             "converging x2 run stagnated at 1.12 N."),
    parser.add_argument("--gamma-loop", default="base", choices=("base", "anderson"),
                        help="VSM inner circulation solver. Default 'base', and "
                             "only for THIS script: the load case is unsteered "
                             "and mirror-symmetric, which is the measurement, "
                             "and 'base' is exactly mirror-symmetric (3e-7 N per "
                             "panel) where 'anderson' is not (0.18 N) and "
                             "tightening its tolerance does not help. With AV on, "
                             "'anderson' also converges to a DIFFERENT trim "
                             "branch than 'base' from the same cold start -- "
                             "stalled-tip on a wing that is 7 deg clear of its "
                             "onset -- and that is what an unguarded A/B here "
                             "fails on: 0.26 N at iteration 31 and 310.8 mm of "
                             "left-right mismatch, against 0.013 N in 7 and "
                             "0.025 mm on 'base', at the same cost. This is NOT "
                             "a recommendation for post-stall work: AV is what "
                             "regularises a stalled circulation and the "
                             "accelerator is what makes AV-active solves "
                             "tractable, so as_config's anderson + 1e-8 stands "
                             "for the depower and steering drivers.")
    parser.add_argument("--refine", type=int, default=1,
                        help="canopy mesh refinement k: every quad becomes k x k "
                             "before triangulating. The coarse lattice is kept, "
                             "so tubes, cables and bridle attachments are "
                             "untouched; only the fabric is refined.")
    parser.add_argument("--patterns", nargs="+", default=list(PATTERNS),
                        choices=list(PATTERNS))
    parser.add_argument("--attached-polars", action="store_true",
                        help="solve on polars continued past the stall onset "
                             "(aerodynamic.attached_polars). With artificial "
                             "viscosity on, a tip past the onset can lock onto "
                             "a stalled branch, often on ONE side only, which "
                             "breaks symmetry however symmetric the structure "
                             "is; the attached polars have no such fixed point.")
    parser.add_argument("--no-av", action="store_true",
                        help="artificial viscosity off (aerodynamic."
                             "is_with_artificial_viscosity). Default: inherit "
                             "as_config, which has it on.")
    parser.add_argument("--inner-tol", type=float, default=None,
                        help="structural_billow.force_tolerance [N]: the inner "
                             "solve's absolute floor. Default: as_config.")
    parser.add_argument("--inner-tol-rel", type=float, default=None,
                        help="structural_billow.relative_force_tolerance, times "
                             "the total load. Default: as_config.")
    parser.add_argument("--udp", type=float, default=None,
                        help="walk the depower tape from its built length to "
                             "this u_dp before the state is recorded (the "
                             "kite's quadratic tape calibration). The tape is "
                             "one central element, so the walk is symmetric. "
                             "Default: stay at the built length.")
    parser.add_argument("--tape-step", type=float, default=0.04,
                        help="tape step of the --udp walk [m]")
    parser.add_argument("--no-stagnation-stop", action="store_true",
                        help="disable the driver's stagnation stop, leaving "
                             "--max-iter as the only guard. That check compares "
                             "TWO residuals n_max_constant_residual_force apart, "
                             "so an oscillating residual trips it by "
                             "coincidence: a u_dp 0.25 walk was stopped at "
                             "21.5 N, one iteration after a tape step, because "
                             "iteration 5 had happened to sit at 23.2 N.")
    parser.add_argument("--tag", default="",
                        help="suffix for the result folders and summary, so a "
                             "rerun under a changed model (a new load transfer, "
                             "say) sits beside the earlier results instead of "
                             "overwriting them")
    parser.add_argument("--strut-padding", default=None,
                        choices=("legacy", "bisect_longest"),
                        help="where the strut padding nodes go (read_struc_geometry_yaml). "
                             "Default: leave it to the config. Both give the same 97 "
                             "tube elements and the same reference curvature; they "
                             "differ in spacing, so this is the knob for a mesh A/B.")
    parser.add_argument("--kite", default=DEFAULT_KITE_NAME)
    args = parser.parse_args()
    suffix = f"_{args.tag}" if args.tag else ""

    project = Path(__file__).resolve().parents[2]
    aerodynamic = {}
    if args.gamma_loop:
        aerodynamic["gamma_loop_type"] = args.gamma_loop
    if args.attached_polars:
        aerodynamic["attached_polars"] = True
    if args.no_av:
        aerodynamic["is_with_artificial_viscosity"] = False
    overrides = {"aerodynamic": aerodynamic} if aerodynamic else None
    shared = build_once(project, args.kite, args.panels_per_section, overrides)
    if args.strut_padding:
        # Safe to set after build_once: the structural reader runs inside
        # run_chain, which deep-copies shared["config"] at call time.
        shared["config"]["strut_padding"] = args.strut_padding
    shared["config"]["wind_speed_wind_ref"] = float(args.wind)
    shared["config"]["aero_structural_solver"]["tol"] = float(args.tol)
    shared["config"]["aero_structural_solver"]["max_iter"] = int(args.max_iter)
    stagnation = (args.stagnation_tol if args.stagnation_tol is not None
                  else 0.1 * float(args.tol))
    shared["config"]["aero_structural_solver"]["stagnation_tol"] = float(stagnation)
    if args.no_stagnation_stop:
        shared["config"]["aero_structural_solver"]["n_max_constant_residual_force"] = (
            int(args.max_iter) + 1
        )
    root = aerostructural_results_root(project, args.kite) / "billow_canopy_ab"
    root.mkdir(parents=True, exist_ok=True)

    reach, step = 0.0, 0.0
    if args.udp is not None:
        built_length = built_tape_length(shared)
        reach = tape_length(args.udp) - built_length
        step = abs(float(args.tape_step))
        print(f"depower tape {built_length:.4f} -> {built_length + reach:.4f} m "
              f"(u_dp {args.udp}), walked in {step} m steps")

    print(f"unsteered, {'unactuated' if args.udp is None else 'depower-actuated'}, "
          f"v_w = {args.wind} m/s, {args.panels_per_section} panels per section\n")
    print(f"{'pattern':10s} {'nodes':>6} {'DOF':>6} {'tri':>5} | "
          f"{'built':>8} {'trimmed':>9} {'generated':>10} | "
          f"{'res':>6} {'it':>3} | {'span':>6} | {'s':>5}")
    print("-" * 96)

    summary = {}
    for pattern in args.patterns:
        started = time.perf_counter()
        shared["config"].setdefault("structural_billow", {})
        settings = dict(shared["config"].get("structural_billow") or {})
        settings["canopy_pattern"] = pattern
        if args.inner_tol is not None:
            settings["force_tolerance"] = float(args.inner_tol)
        if args.inner_tol_rel is not None:
            settings["relative_force_tolerance"] = float(args.inner_tol_rel)
        settings["canopy_refinement"] = args.refine
        config_backup = copy.deepcopy(shared["config"])
        shared["config"]["structural_billow"] = settings

        folder = root / ((pattern if args.refine == 1
                          else f"{pattern}_x{args.refine}") + suffix)
        # reach = 0 -> no actuation, a single converged unsteered state;
        # otherwise the recorded state is the end of the depower walk.
        tracking, meta, geometry, structure = run_chain(
            shared, args.wind, reach, step, folder
        )
        shared["config"] = config_backup

        n_iter = int(meta["n_iter"])
        final = np.asarray(tracking["positions"][n_iter - 1])
        grid = structure.grid
        built = mirror_mismatch(structure.model.nodes, grid)
        trimmed = mirror_mismatch(final, grid)
        span, arc, rise = span_metrics(final, geometry)
        residual = float(np.asarray(tracking["residual_norm"])[n_iter - 1])
        elapsed = time.perf_counter() - started

        summary[pattern] = dict(
            nodes=int(structure.model.n_nodes),
            dof=int(structure.model.layout.n_dof),
            triangles=int(len(structure.model.element_set(sb.CANOPY).connectivity)),
            built_max_mm=float(built.max() * 1e3),
            trimmed_max_mm=float(trimmed.max() * 1e3),
            trimmed_mean_mm=float(trimmed.mean() * 1e3),
            residual=residual, iterations=n_iter - 1,
            span=span, le_arc=arc, arch_rise=rise, seconds=elapsed,
        )
        s = summary[pattern]
        print(f"{pattern:10s} {s['nodes']:6d} {s['dof']:6d} {s['triangles']:5d} | "
              f"{s['built_max_mm']:7.3f}m {s['trimmed_max_mm']:8.3f}m "
              f"{s['trimmed_max_mm'] - s['built_max_mm']:9.3f}m | "
              f"{residual:6.2f} {s['iterations']:3d} | {span:6.3f} | {elapsed:5.0f}",
              flush=True)

    summary_path = root / f"summary{suffix}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("-" * 96)
    print("mismatch columns are millimetres, max over the wing grid.\n")

    if "diagonal" not in summary:
        return
    base = summary["diagonal"]["trimmed_max_mm"]
    for pattern in [p for p in args.patterns if p != "diagonal"]:
        got = summary[pattern]["trimmed_max_mm"]
        print(f"  {pattern:9s}: {got:7.3f} mm vs {base:7.3f} mm "
              f"({100 * (got - base) / base:+.1f}%)")
    print(f"\nwritten to {summary_path}")


if __name__ == "__main__":
    main()

"""A lightly steered Billow state at the centre of the wind window.

The kite is first converged unsteered at the built depower tape, then the
steering tapes are stepped to a half-difference ``--steer`` (the first tape
shortened, the second lengthened by the same amount) and re-converged. Both
happen inside ONE coupled call -- the driver's steering walk -- so the steered
state starts from the unsteered equilibrium, and
``steering_settle_iterations_after_update`` holds the loop open while the
rolled, turning state grows out of the millimetre-level asymmetry the tape step
first produces. Without that hold a light steering input can "converge" on the
still-symmetric state (measured on the PSS path at u_s 0.025 and 0.05).

Reported, for the unsteered state the run passes through and the steered one it
ends on: the trim's course rate, the kite attitude (a best-fit rigid rotation of
the wing from its built shape), apparent speed, tether force, and the turn gain
``chi_dot / (v_a * delta)`` the steering campaigns and the flight identification
quote (2019 reel-out flight: 0.230 rad/s per m^2/s). And the left-right
asymmetry the steering produced, split into a rigid rotation and a genuine
change of shape.

Usage (from project root):
    python scripts/aerostructural/run_steering_BILLOW.py
    python scripts/aerostructural/run_steering_BILLOW.py --steer 0.025
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from awetrim.aerostructural.logging_config import *  # noqa: F401,F403
from awetrim.aerostructural.results import aerostructural_results_root
from common import DEFAULT_KITE_NAME

from check_mirror_asymmetry import decompose, kabsch, mirror_partners
from run_chain_depower_BILLOW import build_once, run_chain


def attitude_deg(positions, built, wing):
    """Roll, pitch, yaw [deg] of the wing's best-fit rigid rotation from built.

    Extracted in the x -> y -> z sequence ``rotate_geometry`` applies, so the
    angles read the same way as the trim's increments.
    """
    rotation, _ = kabsch(built[wing], positions[wing])
    roll = np.arctan2(rotation[2, 1], rotation[2, 2])
    pitch = -np.arcsin(np.clip(rotation[2, 0], -1.0, 1.0))
    yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
    return np.degrees([roll, pitch, yaw])


def state_row(tracking, index, built, wing):
    """Flight state of one tracked coupled iteration."""
    trim = np.asarray(tracking["trim_state"])[index]
    speed = float(np.asarray(tracking["speed_apparent"])[index])
    delta = float(np.asarray(tracking["steering_half_difference"])[index])
    roll, pitch, yaw = attitude_deg(np.asarray(tracking["positions"][index]), built, wing)
    return dict(
        iteration=int(index),
        steering_half_difference=delta,
        residual=float(np.asarray(tracking["residual_norm"])[index]),
        speed_apparent=speed,
        speed_tangential=float(trim[0]),
        course_rate=float(trim[4]),
        roll_deg=float(roll), pitch_deg=float(pitch), yaw_deg=float(yaw),
        aero_force=float(np.linalg.norm(np.asarray(tracking["f_ext"][index]).sum(axis=0))),
        turn_gain=(float(trim[4]) / (speed * delta)
                   if np.isfinite(delta) and abs(delta) > 1e-9 else float("nan")),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steer", type=float, default=0.05,
                        help="steering tape half-difference [m]: the first tape "
                             "shortened and the second lengthened by this much. "
                             "The 2019 steering envelope is 0.25 m.")
    parser.add_argument("--steer-step", type=float, default=0.0,
                        help="walk the steering in steps of this size [m], "
                             "re-converging at each; 0 = one step")
    parser.add_argument("--wind", type=float, default=4.2,
                        help="wind at the reference height [m/s]; 4.2 puts the "
                             "built-tape kite near v_a = 20 m/s")
    parser.add_argument("--panels-per-section", type=int, default=2)
    parser.add_argument("--pattern", default="cross",
                        choices=("diagonal", "union", "cross"))
    parser.add_argument("--refine", type=int, default=1,
                        help="canopy mesh refinement; 1 is the coarse mesh")
    parser.add_argument("--tol", type=float, default=0.5,
                        help="coupled residual gate [N]")
    parser.add_argument("--stagnation-tol", type=float, default=None,
                        help="coupled stagnation tolerance [N]; default a tenth "
                             "of --tol (as_config's 2 N exceeds the whole gap "
                             "to a 0.5 N gate)")
    parser.add_argument("--settle", type=int, default=None,
                        help="MINIMUM iterations after a steering step before "
                             "an exit (covers the dead band before the course "
                             "rate moves); default as_config's "
                             "steering_settle_iterations_after_update")
    parser.add_argument("--course-rate-tol", type=float, default=None,
                        help="exit only once the trim course rate's estimated "
                             "remaining drift is below this [rad/s]; default "
                             "the driver's 1e-3 "
                             "(steering_settle_course_rate_tol)")
    parser.add_argument("--course-rate-rtol", type=float, default=None,
                        help="relative part of the same test, a fraction of "
                             "|course rate|; the drift is accepted within "
                             "max(tol, rtol |chi_dot|). A loose value (0.1) "
                             "takes a snapshot of a state the loop does not "
                             "yet settle (see the trim/structure attitude "
                             "note in aerostructural/AGENTS.md); the report "
                             "then gives the extrapolated end point too.")
    parser.add_argument("--gamma-loop", default="base", choices=("base", "anderson"),
                        help="VSM circulation loop. Default base, which is what "
                             "the unsteered coarse baselines converged on: with "
                             "artificial viscosity on, the LEI-V3's stalled tips "
                             "trap Anderson in its post-stall limit cycle at "
                             "every evaluation (1000-iteration cap, no fallback), "
                             "and the unconverged tip loads kept the coupled "
                             "residual wandering at 3-27 N for 15 iterations "
                             "where base closes 0.04 N in 5. 'anderson' = "
                             "as_config.")
    parser.add_argument("--relaxation", type=float, default=None,
                        help="fixed relaxation factor with Aitken OFF (1.0 = "
                             "plain fixed point). Default: as_config's Aitken. "
                             "Measured on this case: after the steering step "
                             "Aitken pins at its 0.05 floor, because every "
                             "update carries a 0.78 deg rigid swing about the "
                             "KCU that the trim undoes the next iteration, so "
                             "the shape crawls at ~3%% per iteration.")
    parser.add_argument("--max-iter", type=int, default=60)
    parser.add_argument("--tag", default="")
    parser.add_argument("--strut-padding", default=None,
                        choices=("legacy", "bisect_longest"),
                        help="where the strut padding nodes go; default leaves it "
                             "to the config. Set legacy to reproduce a stored run.")
    parser.add_argument("--kite", default=DEFAULT_KITE_NAME)
    args = parser.parse_args()

    project = Path(__file__).resolve().parents[2]
    # Before build_once: it constructs the VSM solver from the config it is
    # handed, so a loop type set afterwards would be silently ignored.
    shared = build_once(project, args.kite, args.panels_per_section,
                        {"aerodynamic": {"gamma_loop_type": args.gamma_loop}})
    solver = shared["config"]["aero_structural_solver"]
    solver["tol"] = float(args.tol)
    solver["max_iter"] = int(args.max_iter)
    solver["stagnation_tol"] = float(
        args.stagnation_tol if args.stagnation_tol is not None else 0.1 * args.tol
    )
    if args.settle is not None:
        solver["steering_settle_iterations_after_update"] = int(args.settle)
    if args.course_rate_tol is not None:
        solver["steering_settle_course_rate_tol"] = float(args.course_rate_tol)
    if args.course_rate_rtol is not None:
        solver["steering_settle_course_rate_rtol"] = float(args.course_rate_rtol)
    if args.relaxation is not None:
        solver["is_with_aitken_relaxation"] = False
        solver["relaxation_factor"] = float(args.relaxation)
    if args.strut_padding:
        shared["config"]["strut_padding"] = args.strut_padding
    settings = dict(shared["config"].get("structural_billow") or {})
    settings.update(canopy_pattern=args.pattern, canopy_refinement=args.refine)
    shared["config"]["structural_billow"] = settings

    mesh = args.pattern if args.refine == 1 else f"{args.pattern}_x{args.refine}"
    name = f"{mesh}_steer_{'p' if args.steer >= 0 else 'm'}{abs(args.steer) * 1e3:04.0f}mm"
    folder = (aerostructural_results_root(project, args.kite) / "billow_steering"
              / (f"{name}_{args.tag}" if args.tag else name))

    aero = shared["config"]["aerodynamic"]
    print(f"steering half-difference {args.steer:+.4f} m "
          f"({'one step' if not args.steer_step else f'{args.steer_step} m steps'}), "
          f"v_w = {args.wind} m/s, {mesh} canopy, "
          f"{args.panels_per_section} panels per section, gate {args.tol} N, "
          f"{args.gamma_loop} loop, AV "
          f"{'on' if aero.get('is_with_artificial_viscosity') else 'OFF'}\n", flush=True)
    started = time.perf_counter()
    tracking, meta, _, structure = run_chain(
        shared, args.wind, 0.0, 0.0, folder, steer=args.steer, steer_step=args.steer_step
    )
    elapsed = time.perf_counter() - started

    n_iter = int(meta["n_iter"])
    built = structure.model.nodes
    grid = structure.grid
    wing = np.unique(grid)
    rows = [state_row(tracking, k, built, wing) for k in range(1, n_iter)]

    print(f"\n{'it':>3} {'delta':>7} {'res':>8} {'v_a':>6} {'chi_dot':>8} "
          f"{'roll':>6} {'pitch':>6} {'yaw':>6} {'F':>6}")
    print("-" * 66)
    for row in rows:
        print(f"{row['iteration']:3d} {row['steering_half_difference']:7.4f} "
              f"{row['residual']:8.3f} {row['speed_apparent']:6.2f} "
              f"{row['course_rate']:+8.4f} {row['roll_deg']:+6.2f} "
              f"{row['pitch_deg']:+6.2f} {row['yaw_deg']:+6.2f} {row['aero_force']:6.0f}")

    # The unsteered state is the last iterate before the steering step.
    deltas = np.array([row["steering_half_difference"] for row in rows])
    unsteered = [row for row in rows if abs(row["steering_half_difference"]) < 1e-9]
    unsteered = unsteered[-1] if unsteered else None
    steered = rows[-1]

    per_pair, rigid_deg, _ = decompose(
        np.asarray(tracking["positions"][n_iter - 1]), built, grid, mirror_partners(built)
    )
    global_mm = 1e3 * max(p[2] for p in per_pair)
    intrinsic_mm = 1e3 * max(p[3] for p in per_pair)
    tip = per_pair[0]

    print("-" * 66)
    converged = bool(meta["converged"]) and bool(
        np.isclose(deltas[-1], args.steer, atol=1e-9)
    )
    print(f"{'CONVERGED' if converged else 'NOT CONVERGED'} in {n_iter - 1} coupled "
          f"iterations, {elapsed / 60:.1f} min -> {folder}\n")
    for label, row in (("unsteered", unsteered), ("steered", steered)):
        if row is None:
            continue
        print(f"{label:>10}: residual {row['residual']:.3f} N (it {row['iteration']}), "
              f"v_a {row['speed_apparent']:.2f} m/s, chi_dot {row['course_rate']:+.4f} rad/s, "
              f"roll {row['roll_deg']:+.2f} deg, yaw {row['yaw_deg']:+.2f} deg, "
              f"F {row['aero_force']:.0f} N")
    remaining = float(meta.get("course_rate_remaining", float("nan")))
    extrapolated = steered["course_rate"] + np.sign(steered["course_rate"]) * remaining
    print(f"\nturn gain chi_dot / (v_a delta) = {steered['turn_gain']:+.3f} rad/s per m^2/s "
          f"(2019 reel-out flight 0.230, PSS coupled campaigns 0.20-0.23)")
    print(f"course rate still to come at exit (settle rule's estimate): {remaining:.2e} rad/s "
          f"-> extrapolated chi_dot {extrapolated:+.4f} rad/s, gain "
          f"{extrapolated / (steered['speed_apparent'] * args.steer):+.3f}")
    print(f"left-right asymmetry: {global_mm:.1f} mm global, {intrinsic_mm:.1f} mm of it "
          f"shape (rigid part {rigid_deg:.2f} deg); tip pair {1e3 * tip[3]:.1f} mm shape "
          f"on {1e3 * tip[4]:.1f} mm of elastic deformation")

    summary = dict(
        steer=args.steer, steer_step=args.steer_step, wind=args.wind,
        gamma_loop=args.gamma_loop,
        relaxation=("aitken" if args.relaxation is None else args.relaxation),
        artificial_viscosity=bool(aero.get("is_with_artificial_viscosity")),
        pattern=args.pattern, refine=args.refine,
        panels_per_section=args.panels_per_section, tol=args.tol,
        converged=converged, iterations=n_iter - 1, seconds=elapsed,
        unsteered=unsteered, steered=steered,
        tether_force=float(meta["tether_force"]),
        course_rate_remaining=remaining,
        course_rate_extrapolated=float(extrapolated),
        course_rate_tol=solver.get("steering_settle_course_rate_tol"),
        course_rate_rtol=solver.get("steering_settle_course_rate_rtol"),
        asymmetry_global_mm=global_mm, asymmetry_intrinsic_mm=intrinsic_mm,
        asymmetry_rigid_deg=float(rigid_deg),
        history=rows,
    )
    (folder / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

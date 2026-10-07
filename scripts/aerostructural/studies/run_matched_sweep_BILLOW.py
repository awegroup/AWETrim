"""Depower and steering chains at the matched depower, on Billow's full model.

The wes-quasi-steady paper attributes the 2019 flight's depower on the Billow
WIREFRAME (cables and pulleys only): u_dp 0.242 on reel-out, 0.323 on reel-in,
at the centre of the wind window. This re-runs the paper's two centre-of-window
campaigns on the FULL model -- inflatable tube beams and a wrinkling membrane
canopy on the same line system -- so the attribution can be checked against a
structure that deforms like the kite does:

* **depower**: u_dp from 0.180 to the reel-in value, on the paper's own tape
  grid (0.05 m steps from u_dp 0.180) with both matched values added as exact
  rows, so every row lines up with a wireframe row or a matched point;
* **steering**: the tape half-difference 0 -> 0.25 m (the 2019 envelope) in
  0.025 m steps at the reel-out value.

The load case is the paper's REF state: window centre (elevation 0, azimuth 0,
course 90 deg), no gravity, no radial speed, tetherless trim (the full-model
driver never carries the tether), the as-flown 2019 system (KCU 22 kg).

**One chain per TARGET apparent speed**, as in the paper: the trim is only
approximately v^2-self-similar here, because the tube law saturates and the
canopy wrinkles, so where a row sits in load has to be controlled rather than
left to slide with L/D. The wind of every row is predicted from the paper's
ratio table (v_a / v_w as a cubic in u_dp, wireframe) scaled by the ratio the
chain actually achieved one row back, and the row is re-converged at a
corrected wind when it still lands more than ``--va-tol`` off target.

Each chain is ONE process holding ONE Billow structure: every row starts from
the previous row's equilibrium, tapes in place, and the driver walks the next
tape step inside its own loop. That is a continuation, not a restart -- a cold
solve per row would re-select the solution branch every time. A chain resumes
after an interruption: rows already on disk are skipped and the first missing
one is reached from a cold start by walking the tapes to it.

Usage (from project root):
    python scripts/aerostructural/studies/run_matched_sweep_BILLOW.py --mode steering --target-va 19
    python scripts/aerostructural/studies/run_matched_sweep_BILLOW.py --mode depower --direction up --target-va 19
"""

import argparse
import copy
import csv
import json
import time
from pathlib import Path

import numpy as np

from awetrim.aerostructural.logging_config import *  # noqa: F401,F403
from awetrim.aerostructural.results import aerostructural_results_root, save_sim_output
from awetrim.aerostructural.case import DEFAULT_KITE_NAME

from run_chain_depower_BILLOW import (
    DEPOWER_QUADRATIC,
    build_once,
    coupled_call,
    depower_input,
    prepare_structure,
    span_metrics,
    structure_billow_rest_length,
)
from run_steering_BILLOW import attitude_deg

#: 2019 centre-of-window attribution on the Billow wireframe (the paper's
#: attributed_udp_billow9.json, 2026-09-13): reel-out and reel-in.
UDP_REELOUT = 0.242
UDP_REELIN = 0.323

#: The paper's centre-of-window REF state.
REF_STATE = dict(
    angle_elevation_deg=0.0,
    angle_azimuth_deg=0.0,
    angle_course_deg=90.0,
    speed_radial=0.0,
    distance_radial=286.6,
    is_with_gravity=False,
)

#: Scalars copied from the final trim into each row record.
TRIM_KEYS = (
    "cl", "cd", "cfx", "cfy", "aoa_deg", "aoa_course_deg", "side_slip_deg",
    "aero_roll_deg", "stall_margin_min_deg", "n_stalled_panels", "n_panels",
    "stalled_fraction", "stalled_area_fraction", "av_stage", "Umag",
    "kcu_drag_coefficient",
    # The deformed body's projected area the VSM normalised cl/cd by; the
    # flight EKF uses the yaml's 19.75 m^2, so compare cl * area / 19.75.
    "kcu_area_reference_m2",
)


def tape_length(udp):
    """Depower map u_dp -> tape length [m], l = a u^2 + b u + c."""
    a, b, c = DEPOWER_QUADRATIC
    return a * udp**2 + b * udp + c


def depower_grid(udp_min, udp_max, ldp_step, anchors):
    """Tape lengths [m]: the paper's grid from ``udp_min`` plus exact anchors."""
    start, stop = tape_length(udp_min), tape_length(udp_max)
    grid = list(start + ldp_step * np.arange(int(np.floor((stop - start) / ldp_step + 1e-9)) + 1))
    grid += [tape_length(u) for u in anchors if udp_min - 1e-9 <= u <= udp_max + 1e-9]
    grid = np.unique(np.round(grid, 4))
    return [float(g) for g in grid]


class RatioModel:
    """v_a / v_w predicted along a chain.

    The table's cubic in u_dp carries the SHAPE (how L/D moves with depower);
    the chain's own last achieved ratio carries the LEVEL, which the full
    model need not share with the wireframe the table was fitted on.
    """

    def __init__(self, table_path):
        self.coeffs = None
        if table_path:
            fit = json.loads(Path(table_path).read_text(encoding="utf-8"))["fit"]
            self.coeffs = np.asarray(fit["coeffs"], float)
            self.range = tuple(fit["udp_range"])

    def shape(self, udp):
        if self.coeffs is None:
            return 5.0
        return float(np.polyval(self.coeffs, np.clip(udp, *self.range)))

    def predict(self, udp, last=None):
        """``last`` = (udp, achieved ratio) of the previous row, or None."""
        if last is None:
            return self.shape(udp)
        return last[1] * self.shape(udp) / self.shape(last[0])


def row_name(mode, ldp, us):
    return (f"udp_{float(depower_input(ldp)):.4f}" if mode == "depower"
            else f"us_{us:.3f}")


def load_rows(chain_dir):
    rows = []
    for path in sorted(chain_dir.glob("*/row.json")):
        if path.parent.name.startswith("_"):   # set aside by hand
            continue
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    return rows


def write_csv(chain_dir, rows):
    flat = [{k: v for k, v in row.items() if not isinstance(v, (list, dict))} for row in rows]
    if not flat:
        return
    fields = list(dict.fromkeys(k for row in flat for k in row))
    with (chain_dir / "rows.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flat)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("depower", "steering"), required=True)
    parser.add_argument("--direction", choices=("up", "down"), default="up",
                        help="depower only: rows above (up) or below (down) the "
                             "built tape length, walked away from it")
    parser.add_argument("--target-va", type=float, required=True,
                        help="apparent speed every row of the chain is held at [m/s]")
    parser.add_argument("--udp-min", type=float, default=0.180)
    parser.add_argument("--udp-max", type=float, default=UDP_REELIN)
    parser.add_argument("--udp-steering", type=float, default=UDP_REELOUT,
                        help="depower the steering chain is solved at")
    parser.add_argument("--ldp-step", type=float, default=0.05,
                        help="depower tape grid step [m] (the paper's)")
    parser.add_argument("--us-max", type=float, default=0.25)
    parser.add_argument("--us-step", type=float, default=0.025)
    parser.add_argument("--walk-step", type=float, default=0.05,
                        help="largest tape step inside one coupled call [m]")
    parser.add_argument("--ratio-table", default=None,
                        help="va_window_<flight>.json with the v_a/v_w cubic")
    parser.add_argument("--va-tol", type=float, default=0.3,
                        help="accept a row within this of the target [m/s]")
    parser.add_argument("--max-corrections", type=int, default=2,
                        help="warm re-solves at a corrected wind per row")
    parser.add_argument("--tol-min", type=float, default=0.5,
                        help="coupled residual gate floor [N]")
    parser.add_argument("--tol-rel", type=float, default=2e-4,
                        help="coupled residual gate as a fraction of the previous "
                             "row's tether force (the larger of the two applies)")
    parser.add_argument("--fallback-rel", type=float, default=2e-3,
                        help="accept a row that stagnated above the gate as a "
                             "'plateau' when its residual is within this fraction "
                             "of the tether force and its course rate has settled "
                             "(the steered rows' limit cycle; the paper's fallback)")
    parser.add_argument("--max-iter", type=int, default=60)
    parser.add_argument("--max-walk-retries", type=int, default=2,
                        help="extra calls allowed for a tape walk that ran out of "
                             "iterations before arriving")
    parser.add_argument("--collapse-fraction", type=float, default=0.2,
                        help="stop the chain when a row ends this far off its target "
                             "apparent speed (fraction) with a stalled wing")
    parser.add_argument("--max-bad-rows", type=int, default=2,
                        help="stop after this many consecutive unconverged rows")
    parser.add_argument("--gamma-loop", default="base", choices=("base", "anderson"),
                        help="VSM circulation loop; base, see run_steering_BILLOW")
    parser.add_argument("--panels-per-section", type=int, default=2)
    parser.add_argument("--pattern", default="cross")
    parser.add_argument("--system-config", default="data/LEI-V3-KITE/system_flown_2019.yaml")
    parser.add_argument("--root", default="billow_matched_sweep",
                        help="results folder under results/<kite>/aerostructural/")
    parser.add_argument("--kite", default=DEFAULT_KITE_NAME)
    args = parser.parse_args()

    project = Path(__file__).resolve().parents[3]
    shared = build_once(project, args.kite, args.panels_per_section,
                        {"aerodynamic": {"gamma_loop_type": args.gamma_loop}},
                        system_config_path=project / args.system_config)
    config = copy.deepcopy(shared["config"])
    config.update(REF_STATE)
    config["steering_tape_extension_step"] = float(args.us_step)
    settings = dict(config.get("structural_billow") or {})
    settings.update(canopy_pattern=args.pattern, canopy_refinement=1)
    config["structural_billow"] = settings
    solver = config["aero_structural_solver"]
    solver["max_iter"] = int(args.max_iter)

    tag = f"va{args.target_va:g}"
    # A steering chain at any depower other than the paper's reel-out value
    # gets that depower in its folder name, so chains never share rows.
    steering_name = ("steering" if abs(args.udp_steering - UDP_REELOUT) < 1e-6
                     else f"steering_udp{args.udp_steering:.3f}")
    chain = (f"depower_{tag}_{args.direction}" if args.mode == "depower"
             else f"{steering_name}_{tag}")
    chain_dir = aerostructural_results_root(project, args.kite) / args.root / chain
    chain_dir.mkdir(parents=True, exist_ok=True)
    ratio = RatioModel(args.ratio_table)

    prepared = prepare_structure(shared, config)
    structure = prepared["structure"]
    tape_index = prepared["power_tape_index"]
    built_tape = structure_billow_rest_length(structure, tape_index)
    wing = np.unique(structure.grid)
    steer_index = [int(i) for i in prepared["steering_tape_indices"][:2]]
    built_steer = [structure_billow_rest_length(structure, i) for i in steer_index]

    def live_tape():
        return structure_billow_rest_length(structure, tape_index)

    def live_steer():
        """Half-difference on the structure now, the driver's convention."""
        left, right = (structure_billow_rest_length(structure, i) for i in steer_index)
        return 0.5 * ((built_steer[0] - left) + (right - built_steer[1]))

    # Rows as (tape length, steering half-difference), in walking order.
    if args.mode == "depower":
        grid = depower_grid(args.udp_min, args.udp_max, args.ldp_step,
                            (UDP_REELOUT, UDP_REELIN))
        rows = ([(l, 0.0) for l in grid if l > built_tape + 1e-6] if args.direction == "up"
                else [(l, 0.0) for l in reversed(grid) if l < built_tape - 1e-6])
    else:
        n_us = int(round(args.us_max / args.us_step))
        rows = [(tape_length(args.udp_steering), round(args.us_step * k, 6))
                for k in range(n_us + 1)]

    done = {row["name"]: row for row in load_rows(chain_dir)
            if row.get("status") in ("ok", "off_target", "plateau")}
    print(f"{chain}: {len(rows)} rows, {len(done)} already on disk, built tape "
          f"{built_tape:.4f} m (u_dp {float(depower_input(built_tape)):.3f}), "
          f"{args.gamma_loop} loop\n", flush=True)

    last = None               # (udp, achieved ratio) of the last solved row
    ratios_us = []            # achieved ratios along the steering chain
    tether_force = None
    bad_rows = 0
    chain_started = time.perf_counter()

    for ldp, us in rows:
        name = row_name(args.mode, ldp, us)
        if name in done:
            record = done[name]
            last = (record["udp"], record["va_over_vw"])
            if args.mode == "steering":
                ratios_us.append(record["va_over_vw"])
            tether_force = record["tether_force"]
            continue

        udp = float(depower_input(ldp))
        if args.mode == "steering" and len(ratios_us) >= 2:
            predicted = 2 * ratios_us[-1] - ratios_us[-2]
        elif last is None and abs(ldp - live_tape()) > args.walk_step:
            # A long walk from a cold start passes through every tape between
            # here and there, at ONE wind. Size it for the least-depowered
            # (fastest-flying) state on the path, not the destination: sized
            # for u_dp 0.315, the walk from the built tape flew at 27 m/s and
            # fell into a limit cycle. The first correction brings v_a back up.
            predicted = ratio.shape(min(udp, float(depower_input(live_tape()))))
        else:
            predicted = ratio.predict(udp, last)
        wind = args.target_va / predicted
        gate = max(args.tol_min, args.tol_rel * (tether_force or 0.0))
        solver["tol"] = float(gate)
        solver["stagnation_tol"] = 0.1 * float(gate)

        started = time.perf_counter()
        attempts, iterations = [], 0
        arrived = False
        # Every attempt walks from where the tapes ARE: a call that runs out of
        # iterations mid-walk leaves them short, and a correction that only
        # changed the wind would then save the row at the wrong tape length.
        for attempt in range(args.max_corrections + 1 + args.max_walk_retries):
            reach = ldp - live_tape()
            steer = us - live_steer()
            reach = 0.0 if abs(reach) < 1e-7 else reach
            steer = 0.0 if abs(steer) < 1e-7 else steer
            config["wind_speed_wind_ref"] = float(wind)
            tracking, meta = coupled_call(
                shared, prepared, config,
                reach=reach, step=min(abs(reach), args.walk_step) if reach else 0.0,
                steer=steer,
            )
            arrived = abs(ldp - live_tape()) < 1e-6 and abs(us - live_steer()) < 1e-6
            n_iter = int(meta["n_iter"])
            iterations += n_iter - 1
            va = float(meta["speed_apparent"])
            residual = float(np.asarray(tracking["residual_norm"])[n_iter - 1])
            attempts.append(dict(wind=wind, va=va, residual=residual,
                                 iterations=n_iter - 1, converged=bool(meta["converged"])))
            print(f"  {name} attempt {attempt}: v_w {wind:.3f} -> v_a {va:.2f} "
                  f"(target {args.target_va:g}), residual {residual:.3f} N "
                  f"(gate {gate:.2f}), {n_iter - 1} it, "
                  f"{'converged' if meta['converged'] else 'NOT converged'}"
                  f"{'' if arrived else f', tapes short (l_dp {live_tape():.4f}, u_s {live_steer():.4f})'}",
                  flush=True)
            if not np.isfinite(va) or not np.isfinite(residual):
                break
            corrections = sum(1 for a in attempts if a.get("arrived"))
            attempts[-1]["arrived"] = arrived
            if arrived and abs(va - args.target_va) <= args.va_tol:
                break
            if arrived and corrections >= args.max_corrections:
                break
            wind *= args.target_va / va

        finite = bool(np.isfinite(va) and np.isfinite(residual))
        converged = bool(meta["converged"]) and finite
        # A stagnated row whose loads balance to the fallback fraction of the
        # tether force, with a settled turn, is the steered limit cycle rather
        # than a failure: accepted, and labelled so it can be told apart.
        # Settled = the course rate's spread over the last 8 iterations, not
        # remaining_drift: on a limit cycle the change ratios sit near 1 and
        # the drift estimate is infinite by construction.
        rates = np.asarray(tracking["trim_state"])[max(1, n_iter - 8):n_iter, 4]
        spread = float(np.ptp(rates)) if rates.size else float("inf")
        settled = us < 1e-9 or spread <= max(2e-3, 0.02 * abs(float(meta["course_rate"])))
        plateau = (finite and not converged and settled
                   and residual <= args.fallback_rel * abs(float(meta["tether_force"])))
        on_target = abs(va - args.target_va) <= args.va_tol
        status = ("failed" if not finite
                  else "not_converged" if not arrived
                  else "plateau" if plateau and on_target
                  else "not_converged" if not converged
                  else "ok" if on_target else "off_target")

        row_dir = chain_dir / name
        row_dir.mkdir(parents=True, exist_ok=True)
        save_sim_output(tracking, meta, row_dir)

        positions = np.asarray(tracking["positions"][n_iter - 1])
        span, arc, rise = span_metrics(positions, prepared["geometry"])
        roll, pitch, yaw = attitude_deg(positions, prepared["built_nodes"], wing)
        trim = meta.get("trim_results") or {}
        course_rate = float(meta["course_rate"])
        record = dict(
            name=name, chain=chain, mode=args.mode, target_va=args.target_va,
            udp=udp, ldp=ldp, us=us, ldp_live=live_tape(), us_live=live_steer(),
            wind_speed=float(wind), va=va,
            va_over_vw=va / wind if finite else float("nan"),
            v_tau=float(meta["speed_tangential"]), course_rate=course_rate,
            course_rate_remaining=float(meta["course_rate_remaining"]),
            turn_gain=(course_rate / (va * us) if us > 0 and finite else float("nan")),
            tether_force=float(meta["tether_force"]),
            force_over_va2=float(meta["tether_force"]) / va**2 if finite else float("nan"),
            aero_force=float(np.linalg.norm(np.asarray(tracking["f_ext"][n_iter - 1]).sum(axis=0))),
            residual=residual, gate=gate, iterations=iterations,
            attempts=attempts, seconds=time.perf_counter() - started,
            converged=converged, status=status,
            roll_deg=float(roll), pitch_deg=float(pitch), yaw_deg=float(yaw),
            span=span, le_arc=arc, arch_rise=rise,
            **{key: trim.get(key) for key in TRIM_KEYS},
            alpha_at_ac_deg=trim.get("alpha_at_ac_deg"),
        )
        (row_dir / "row.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        write_csv(chain_dir, load_rows(chain_dir))
        print(f"{name}: {status}, u_dp {udp:.4f}, u_s {us:.3f}, v_a {va:.2f}, "
              f"F_t {record['tether_force']:.0f} N, CL {record['cl']}, CD {record['cd']}, "
              f"chi_dot {course_rate:+.4f}, margin {record['stall_margin_min_deg']}, "
              f"{iterations} it, {record['seconds'] / 60:.1f} min", flush=True)

        if status == "failed":
            print("  state is not finite -- chain stopped; rerun to resume cold", flush=True)
            break
        # Collapse: the row could not be held near its apparent speed and the
        # wing has stalled. On the full model more wind means more deformation
        # and a higher angle of attack, so near the stall margin each wind
        # correction pushes the kite further over (measured: v_a 18.3 -> 16.8
        # -> 9.7 m/s at u_dp 0.231, 40 of 54 panels stalled). Every later row
        # would start from that state, so the chain ends here.
        margin = record.get("stall_margin_min_deg")
        if (status in ("off_target", "not_converged") and finite
                and abs(va - args.target_va) > args.collapse_fraction * args.target_va
                and margin is not None and margin < 0):
            print(f"  collapsed onto the stalled branch (v_a {va:.2f} for target "
                  f"{args.target_va:g}, margin {margin:+.1f} deg) -- chain stopped", flush=True)
            break
        if status == "not_converged":
            bad_rows += 1
            if bad_rows >= args.max_bad_rows:
                print(f"  {bad_rows} consecutive unconverged rows -- chain stopped", flush=True)
                break
        else:
            bad_rows = 0
        last = (udp, record["va_over_vw"])
        if args.mode == "steering":
            ratios_us.append(record["va_over_vw"])
        tether_force = record["tether_force"]

    print(f"\n{chain} done in {(time.perf_counter() - chain_started) / 60:.1f} min "
          f"-> {chain_dir}", flush=True)


if __name__ == "__main__":
    main()

"""Solve the aerostructural dataset the ROM is identified on (step 3, part 1).

Coupled VSM + Billow states at the centre of the wind window, gravity off,
along continuation chains, at either structural fidelity (`fidelity` in
identification.yaml, or --fidelity): `billow`, the full model (tube beams,
membrane canopy, bridle), or `wireframe`, the bridle line system. The
wireframe is reduced from the same full structure file (`fem_geometry`,
default the kite's awesIO structure.yml) and flies its full-chord aerodynamic mesh
(wireframe.driver.build_case_reduced_fem: one structural file for both
fidelities; billowed with aerodynamic.billowing.contraction, loads moved with
aerodynamic.load_transfer: strut_lever):

* depower chains: from the tape length the geometry is built at, the power
  tape walks to each target length, one converged state per step;
* steering chains: the tape is first walked to a set length, then the
  steering half-difference grows, one converged state per step.

Each step warm-starts from the previous one on the SAME structure (the
drivers' solve_step, awetrim.aerostructural.billow.driver and
awetrim.aerostructural.wireframe.driver), so a chain costs about one tape
walk. On every converged state (an anchor) a frozen-shape angle-of-attack
first moved to the realistic stall branch (settle_on_down_branch, on by default),
and a frozen-shape angle-of-attack polar is solved, marched past the stall and
back down (aerodynamic_vsm.run_frozen_alpha_polar_about_trim): the
trims alone tie alpha to the depower almost one to one, the polars separate
them. Everything is read from data/<kite>/identification.yaml (`sweep:`).

Output, results/<kite>/identification/<fidelity>/sweep/ (format in
awetrim.identification.aerostructural_dataset):
    trims.csv     one row per anchor (the coupled trim)
    polars.csv    one row per polar sample
    <chain>/step_<n>/sim_output.h5   the coupled solve of each anchor

Parallel: runs on disjoint --chains may share the output folder (the sweep
files are written under a lock). Resumable: a finished chain is skipped. An unfinished wireframe chain goes
on after its last complete anchor, from the structure that anchor's
sim_output.h5 saved; an unfinished Billow chain is solved again from its
start (its beam frames are not saved). A state that does not converge ends its
chain, and so does a trim that fails (recorded in the step's trim_failed.txt and
listed at the end of the run); the sweep goes on with the
next chain. A finished chain is skipped on a rerun: delete its chain_done to
try it again, or to extend it after raising a steering `max`. About a minute per
anchor.

Optional `start_from_converged` (wireframe): before the chains, the built
tape is solved at every wind, walking the wind up and down on one structure
until each wind's state is converged with |course rate| below a tolerance
(results/.../sweep/base/); every chain then starts from its wind's base state
instead of the unloaded build, whose first iterations can settle on a
one-tip-stalled, turning state with zero steering.

Turn-law states (wireframe, `turn_law:` in the sweep block): the ROM's
turn-rate law (steering_law turn_rate) needs b(u_p), c5 and c6, which the
centre-of-window chains cannot separate from its gain
(awetrim.identification.turn_law). A few saved anchors are re-solved
tetherless: at the centre at mass scale 0 and 1, and with gravity at
other elevations and courses. Written to sweep/turn_law/trims.csv;
solved after the chains by a full run (no --chains), or alone with
--turn-law [ANCHOR ...]. Resumable.

--repolar (wireframe) solves the polars of the saved anchors again over the
current polar_d_alpha_deg, e.g. after widening it to reach the stall, and those that
fail the polar check: each
anchor is restored and re-solved where it stands for its trimmed body, and its
trim and polar rows are replaced together.

Usage (project root):
    python examples/3_rom_identification/run_identification_sweep.py
    python examples/3_rom_identification/run_identification_sweep.py --fidelity wireframe
    python examples/3_rom_identification/run_identification_sweep.py --chains w4.0_depower_2.60
    python examples/3_rom_identification/run_identification_sweep.py --chains w5.0_depower_1.35 w5.0_depower_2.60
    python examples/3_rom_identification/run_identification_sweep.py --list
    python examples/3_rom_identification/run_identification_sweep.py --base-only
    python examples/3_rom_identification/run_identification_sweep.py --repolar
    python examples/3_rom_identification/run_identification_sweep.py --turn-law
    python examples/3_rom_identification/run_identification_sweep.py --turn-law w5.0_steering_1.95/07
"""

from __future__ import annotations

import argparse
import copy
import os
import time
from contextlib import contextmanager
import dataclasses
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from awetrim.aerostructural import aerodynamic_vsm
from awetrim.aerostructural.case import (
    DEFAULT_STRUC_GEOMETRY_FILENAME,
    Case,
    air_density_at_case,
    resolve_kite_paths,
    runtime_config,
)
from awetrim.aerostructural.results import save_sim_output
from awetrim.identification import aerostructural_dataset as ds
from awetrim.identification import turn_law

PROJECT_DIR = Path(__file__).resolve().parents[2]
DONE_MARKER = "chain_done"
TRIM_FAILED_FILE = "trim_failed.txt"
BASE_DIR = "base"
LOCK_FILE = ".sweep_files.lock"
LOCK_STALE_S = 120.0  # a lock this old was left by a killed run


@contextmanager
def sweep_files_lock(out: Path):
    """Write the shared sweep files one run at a time: runs on disjoint
    --chains may share ``out``, and a resume rewrites whole files."""
    out.mkdir(parents=True, exist_ok=True)
    lock = out / LOCK_FILE
    while True:
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except (FileExistsError, PermissionError):  # Windows: the same while the holder deletes it
            try:
                if time.time() - lock.stat().st_mtime > LOCK_STALE_S:
                    lock.unlink()
                    continue
            except (FileNotFoundError, PermissionError):
                time.sleep(0.1)
                continue
            time.sleep(0.1)
    try:
        yield
    finally:
        os.close(fd)
        lock.unlink()


@dataclass
class Step:
    """One converged state of a chain: where the tapes go before it is solved."""

    power_tape_length: float | None  # None: leave the tape where it is
    steering_half_difference: float


@dataclass
class Chain:
    name: str
    wind_speed: float
    steps: list[Step]
    # Stalled span fraction past which the chain ends; None: only a state that
    # does not converge ends it.
    max_stalled_fraction: float | None = None


def _walk(start: float, target: float, step: float) -> list[float]:
    """Tape lengths from ``start`` (excluded) to ``target`` (included)."""
    n = int(np.ceil(abs(target - start) / step - 1e-9))
    return [float(v) for v in np.linspace(start, target, n + 1)[1:]] if n else []


def build_chains(spec: dict, built_tape_length: float) -> list[Chain]:
    """The chains of ``identification.yaml``'s ``sweep`` block."""
    depower, steering = spec["depower"], spec.get("steering") or {}
    chains = []
    for wind in spec["wind_speeds"]:
        # The built state itself opens the first depower chain only.
        for i, target in enumerate(depower["targets"]):
            lengths = _walk(built_tape_length, target, depower["step"])
            steps = ([Step(None, 0.0)] if i == 0 else []) + [Step(l, 0.0) for l in lengths]
            chains.append(Chain(f"w{wind:.1f}_depower_{target:.2f}", wind, steps,
                                spec["max_stalled_fraction"]))
        for length in steering.get("power_tape_lengths", []):
            halves = _walk(0.0, steering["max"], steering["step"])
            steps = [Step(length, h) for h in halves]
            chains.append(Chain(f"w{wind:.1f}_steering_{length:.2f}", wind, steps,
                                steering.get("max_stalled_fraction", spec["max_stalled_fraction"])))
    return chains


def sweep_case(spec: dict) -> Case:
    """The sweep's flight state (``sweep.case``) at the first wind speed and
    the built tape lengths; each solve changes the wind and the tapes (:func:`_at`)."""
    state = dict(spec["case"])
    wind = {**state.pop("wind"), "speed_wind_ref": float(min(spec["wind_speeds"]))}
    return Case(**state, wind=wind, power_tape_extension=0.0, steering_tape_extension=0.0)


def _at(config: dict, *, wind_speed=None, power_tape=None, steering=None) -> dict:
    """``config`` at another wind speed [m/s] and tape command [m]: the case
    keys change together, through the Case."""
    case = Case.from_config(config)
    changes = {}
    if wind_speed is not None:
        changes["wind"] = {**case.wind, "speed_wind_ref": float(wind_speed)}
    if power_tape is not None:
        changes["power_tape_extension"] = float(power_tape)
    if steering is not None:
        changes["steering_tape_extension"] = float(steering)
    return dataclasses.replace(case, **changes).apply(config)


def open_case(fidelity: str, spec: dict, kite: str, work_dir: Path | None = None):
    """``(driver, case)``: the fidelity's driver module and its built case
    (the wireframe's reduced geometry in ``work_dir``, default the
    fidelity's ``geometry`` folder)."""
    kite_dir = PROJECT_DIR / "data" / kite
    system_path = kite_dir / spec.get("system_file", "system.yaml")
    if fidelity == "billow":
        from awetrim.aerostructural.billow import driver

        case = driver.build_case(
            sweep_case(spec), spec.get("overrides"), kite_name=kite, project_dir=PROJECT_DIR,
            system_config_path=system_path,
        )
    else:
        from awetrim.aerostructural.wireframe import driver

        config_path, _, _ = resolve_kite_paths(PROJECT_DIR, kite)
        config = runtime_config(sweep_case(spec), config_path, spec.get("overrides"))
        # the wireframe reduced from the full structure (one structural file
        # for both fidelities), its aero mesh the full geometry's (billowed
        # when aerodynamic.billowing.contraction > 0)
        case = driver.build_case_reduced_fem(
            config=config, kite_name=kite, project_dir=PROJECT_DIR,
            work_dir=work_dir or PROJECT_DIR / "results" / kite / "identification" / fidelity / "geometry",
            system_config_path=system_path,
            fem_geometry_filename=spec.get("fem_geometry", DEFAULT_STRUC_GEOMETRY_FILENAME),
        )
    return driver, case


def prepare(driver, case, spec: dict, config: dict):
    """A fresh structure, at the stiffnesses of its file."""
    return driver.prepare_structure(case, config)


def base_state_path(out: Path, wind_speed: float) -> Path:
    return out / BASE_DIR / f"w{wind_speed:.1f}" / "sim_output.h5"


def solve_base_states(driver, case, spec: dict, out: Path) -> None:
    """The converged unsteered state at the built tape, one per wind, that
    every chain starts from (``start_from_converged`` in identification.yaml;
    wireframe only, it needs ``restore``).

    A cold solve from the unloaded build is a poor start: at the trimmed
    attitude the built tips are past stall, the stalled circulation is
    multi-valued and the first iterations can settle on a one-tip-stalled,
    turning state with zero steering (V3Kite emulation 2026-10-28: 0.05-0.08
    rad/s at 3 and 6 m/s). One structure is therefore walked through the
    winds, up then down, each solve starting from the previous wind's shape,
    until every wind's latest state converged with |course rate| below
    ``course_rate_tol``: from a converged neighbour the sideways mode dies
    out (1e-7 rad/s).
    """
    settings = spec["start_from_converged"] or {}
    tol = float(settings.get("course_rate_tol", 1e-4))
    max_passes = int(settings.get("max_passes", 4))
    done = out / BASE_DIR / DONE_MARKER
    if done.exists():
        print(f"base states: done, skipped ({done})", flush=True)
        return
    winds = sorted(float(w) for w in spec["wind_speeds"])
    config = _at(copy.deepcopy(case.config), wind_speed=winds[0], power_tape=0.0, steering=0.0)
    config["power_tape_extension_step"] = float(spec["depower"]["step"])
    config["steering_tape_extension_step"] = float(spec["steering"]["step"])
    prepared = prepare(driver, case, spec, config)
    latest = {}
    last_saved = None
    for n_pass in range(max_passes):
        order = winds if n_pass % 2 == 0 else winds[::-1]
        for wind in order[1:] if n_pass else order:  # a turn re-solves no wind twice
            started = time.perf_counter()
            config = _at(config, wind_speed=wind)
            try:
                tracking, meta = driver.solve_step(case, prepared, config, final_state={})
            except aerodynamic_vsm.CasadiTrimFailed as failure:
                latest[wind] = (False, float("nan"))
                print(f"  base pass {n_pass + 1}, wind {wind:.1f} m/s: TRIM FAILED, {str(failure)[:200]}", flush=True)
                # The failure leaves the structure mid-solve: back to the last saved shape.
                prepared = prepare(driver, case, spec, config)
                if last_saved is not None:
                    prepared.restore(last_saved)
                continue
            path = base_state_path(out, wind)
            path.parent.mkdir(parents=True, exist_ok=True)
            save_sim_output(tracking, meta, path.parent)
            last_saved = path
            course_rate = float(np.ravel(meta["opt_x"])[4])
            latest[wind] = (bool(meta["converged"]), course_rate)
            print(f"  base pass {n_pass + 1}, wind {wind:.1f} m/s: course rate {course_rate:+.2e} rad/s, "
                  f"{meta['n_iter']} it, {'converged' if meta['converged'] else 'NOT CONVERGED'} "
                  f"| {(time.perf_counter() - started) / 60:.1f} min", flush=True)
        if all(ok and abs(rate) < tol for ok, rate in latest.values()):
            done.write_text("".join(f"{w:.1f} {r:+.3e}\n" for w, (_, r) in sorted(latest.items())),
                            encoding="utf-8")
            return
    raise SystemExit(
        f"base states not clean after {max_passes} passes (|course rate| < {tol}): "
        + ", ".join(f"{w:.1f} m/s {r:+.2e}{'' if ok else ' unconverged'}" for w, (ok, r) in sorted(latest.items()))
    )


def polar_increments(spec: dict) -> np.ndarray:
    """The frozen-shape polar's angle-of-attack increments [deg]."""
    lo, hi, inc = spec["polar_d_alpha_deg"]
    return np.round(np.arange(lo, hi + 0.5 * inc, inc), 9)


def anchor_rows(chain: Chain, index: int, prepared, config: dict, meta: dict,
                final_state: dict, d_alpha: np.ndarray):
    """``(trim_row, polar_rows, trim)`` of a converged state, its frozen-shape
    polar solved; None when the state did not converge (or is not finite)."""
    trim, body = final_state["results_aero"], final_state["body_aero"]
    if not meta["converged"] or not np.all(
        np.isfinite(np.asarray(trim["total_aero_force_vec"], dtype=float))
    ):
        return None
    polar = aerodynamic_vsm.run_frozen_alpha_polar_about_trim(
        body, prepared.vsm_solver, d_alpha, gamma_trim=trim.get("gamma_distribution")
    )
    f_trim = np.asarray(trim["total_aero_force_vec"], dtype=float)
    f_zero = next(s["force_world"] for s in polar if s["d_alpha_deg"] == 0.0)
    anchor = f"{chain.name}/{index:02d}"
    m_arr = np.asarray(prepared.m_arr, dtype=float)
    trim_row = ds.trim_row(
        anchor=anchor, chain=chain.name, step=index,
        wind_speed=chain.wind_speed,
        power_tape_length=prepared.power_tape_length,
        steering_half_difference=prepared.steering_half_difference,
        rho=air_density_at_case(config), mass_kcu=float(m_arr[0]),
        mass_total=float(m_arr.sum()), trim=trim, meta=meta,
        body_z=aerodynamic_vsm.body_centre_panel_axes(body)[2],
        polar_check=float(np.linalg.norm(f_zero - f_trim) / np.linalg.norm(f_trim)),
    )
    return trim_row, ds.polar_rows(anchor, polar), trim


def repolar_chain(driver, case, chain: Chain, spec: dict, out: Path) -> int:
    """Solve the polars of a chain's saved anchors again over the CURRENT
    ``polar_d_alpha_deg`` (e.g. after widening it): each anchor is restored
    from its sim_output.h5 and re-solved where it stands (no actuation, a few
    iterations from a converged shape) for its trimmed aero body, and its trim
    and polar rows are replaced together. Anchors whose polar already spans
    the range and pass the polar check are skipped (resumable). Returns the anchors redone."""
    d_alpha = polar_increments(spec)
    trims, polars = _read_rows(out / ds.TRIMS_FILE), _read_rows(out / ds.POLARS_FILE)
    if trims is None:
        return 0
    steps = sorted(_step_of(trims.anchor[trims.anchor.str.startswith(chain.name + "/")]))
    n_done = 0
    for index in steps:
        anchor = f"{chain.name}/{index:02d}"
        have = polars.d_alpha_deg[polars.anchor == anchor] if polars is not None else pd.Series(dtype=float)
        checked = (trims.polar_check[trims.anchor == anchor] <= ds.POLAR_CHECK_TOLERANCE).all()
        if checked and len(have) and have.min() <= d_alpha.min() + 1e-9 and have.max() >= d_alpha.max() - 1e-9:
            continue
        started = time.perf_counter()
        config = _at(copy.deepcopy(case.config), wind_speed=chain.wind_speed,
                     power_tape=0.0, steering=0.0)
        config["power_tape_extension_step"] = float(spec["depower"]["step"])
        config["steering_tape_extension_step"] = float(spec["steering"]["step"])
        prepared = prepare(driver, case, spec, config)
        prepared.restore(out / chain.name / f"step_{index:02d}" / "sim_output.h5")
        final_state = {}
        try:
            _, meta = driver.solve_step(case, prepared, config, final_state=final_state)
        except aerodynamic_vsm.CasadiTrimFailed as failure:
            print(f"  {anchor}: TRIM FAILED, rows kept as they were: {str(failure)[:200]}", flush=True)
            continue
        rows = anchor_rows(chain, index, prepared, config, meta, final_state, d_alpha)
        if rows is None:
            print(f"  {anchor}: re-solve NOT CONVERGED, rows kept as they were", flush=True)
            continue
        trim_row, polar_rows, trim = rows
        with sweep_files_lock(out):
            for name, new in ((ds.TRIMS_FILE, [trim_row]), (ds.POLARS_FILE, polar_rows)):
                table = _read_rows(out / name)
                if table is not None:
                    table[table.anchor != anchor].to_csv(out / name, index=False)
                ds.append_rows(out / name, new)
        n_done += 1
        print(f"  {anchor}: polar {d_alpha.min():+.0f}..{d_alpha.max():+.0f} deg about alpha "
              f"{trim['aoa_deg']:5.2f} deg, polar check {trim_row['polar_check']:.1e}, "
              f"{meta['n_iter']} it | {(time.perf_counter() - started) / 60:.1f} min", flush=True)
    return n_done


def settle_on_down_branch(driver, case, spec: dict, prepared, tracking, meta, final_state,
                          config: dict, d_alpha, saved: Path, anchor: str):
    """The state on the realistic stall branch: the frozen shape is marched past
    the stall and back down in angle of attack, and the coupled problem solved
    again from that circulation. A chain reaches its states walking the tapes, so
    near stall it stays on the attached branch, past the stall of the same wing
    marched down (LEI-V3, 5 m/s, no steering: 1.35 m had 12 % too much tension,
    1.15 m twice the tension and 4.5 deg too little angle of attack). From a state
    that has only one branch this re-solve changes nothing.

    Returns ``(prepared, tracking, meta, final_state, config)``, the first solve's
    when the re-solve fails, from the state saved at ``saved``. Needs the driver's
    continuation (wireframe); other fidelities keep the first solve.
    """
    if not hasattr(driver, "_trim_seed_of"):
        return prepared, tracking, meta, final_state, config
    trim = final_state["results_aero"]
    gamma_down = aerodynamic_vsm.down_branch_circulation(
        final_state["body_aero"], prepared.vsm_solver, d_alpha, gamma_trim=trim.get("gamma_distribution")
    )
    prepared.trim_seed = driver._trim_seed_of(trim.get("opt_x"), gamma_down)
    settled = _at(config, power_tape=0.0, steering=0.0)
    settled["power_tape_extension_step"] = float(spec["depower"]["step"])
    settled["steering_tape_extension_step"] = float(spec["steering"]["step"])
    final = {}
    try:
        new_tracking, new_meta = driver.solve_step(case, prepared, settled, final_state=final)
        converged = bool(new_meta["converged"])
    except aerodynamic_vsm.CasadiTrimFailed:
        converged = False
    if not converged:
        print(f"  {anchor}: re-solve on the down branch failed, first solve kept", flush=True)
        prepared = prepare(driver, case, spec, settled)
        prepared.restore(saved)
        return prepared, tracking, meta, final_state, config
    shift = final["results_aero"]["tether_force"] / trim["tether_force"] - 1.0
    print(f"  {anchor}: down branch moved the tether force {100.0 * shift:+.2f} %, alpha "
          f"{trim['aoa_deg']:.2f} -> {final['results_aero']['aoa_deg']:.2f} deg", flush=True)
    return prepared, new_tracking, new_meta, final, settled


def solve_chain(driver, case, chain: Chain, spec: dict, out: Path, start: int = 0) -> int:
    """Solve one chain from step ``start``, appending its anchors to the
    sweep files. A chain resumed past its first step starts from the
    structure saved with step ``start - 1``; a fresh one from its wind's
    base state when the sweep has them (``start_from_converged``), else
    from the build. Returns the chain's number of anchors."""
    config = _at(copy.deepcopy(case.config), wind_speed=chain.wind_speed)
    prepared = prepare(driver, case, spec, config)
    if start:
        prepared.restore(out / chain.name / f"step_{start - 1:02d}" / "sim_output.h5")
    elif spec.get("start_from_converged"):
        prepared.restore(base_state_path(out, chain.wind_speed))
    d_alpha = polar_increments(spec)
    n_written = start
    for index, step in enumerate(chain.steps[start:], start=start):
        started = time.perf_counter()
        step_dir = out / chain.name / f"step_{index:02d}"
        step_dir.mkdir(parents=True, exist_ok=True)
        (step_dir / TRIM_FAILED_FILE).unlink(missing_ok=True)
        anchor = f"{chain.name}/{index:02d}"
        # The depower walk comes first and the steering after it, both from
        # the structure's current state (coupled_solver.main's order).
        tape_reach = (
            0.0 if step.power_tape_length is None
            else step.power_tape_length - prepared.power_tape_length
        )
        config = _at(
            config,
            power_tape=tape_reach,
            steering=step.steering_half_difference - prepared.steering_half_difference,
        )
        config["power_tape_extension_step"] = float(spec["depower"]["step"])
        config["steering_tape_extension_step"] = float(spec["steering"]["step"])
        final_state = {}
        try:
            tracking, meta = driver.solve_step(case, prepared, config, final_state=final_state)
        except aerodynamic_vsm.CasadiTrimFailed as failure:
            # A continuation has no state to go on from, as with an unconverged one.
            (step_dir / TRIM_FAILED_FILE).write_text(f"{failure}\n", encoding="utf-8")
            print(
                f"  {anchor}: TRIM FAILED, chain stopped "
                f"| {(time.perf_counter() - started) / 60:.1f} min",
                flush=True,
            )
            break
        save_sim_output(tracking, meta, step_dir)
        if spec.get("settle_on_down_branch", True) and meta["converged"]:
            prepared, tracking, meta, final_state, config = settle_on_down_branch(
                driver, case, spec, prepared, tracking, meta, final_state, config,
                d_alpha, step_dir / "sim_output.h5", anchor,
            )
            save_sim_output(tracking, meta, step_dir)

        rows = anchor_rows(chain, index, prepared, config, meta, final_state, d_alpha)
        # An unconverged state ends the chain before its polar: the fit drops
        # it anyway, and a diverged (NaN) shape has no polar to take.
        if rows is None:
            print(
                f"  {anchor}: NOT CONVERGED, chain stopped "
                f"| {(time.perf_counter() - started) / 60:.1f} min",
                flush=True,
            )
            break
        trim_row, polar_rows, trim = rows
        with sweep_files_lock(out):
            ds.append_rows(out / ds.TRIMS_FILE, [trim_row])
            ds.append_rows(out / ds.POLARS_FILE, polar_rows)
        n_written += 1

        stalled = float(trim.get("stalled_fraction", 0.0) or 0.0)
        print(
            f"  {anchor}: l_dp {prepared.power_tape_length:.3f} m, steer "
            f"{prepared.steering_half_difference:+.3f} m | v_a {np.linalg.norm(trim['va_vel_world']):5.2f} "
            f"m/s, alpha {trim['aoa_deg']:5.2f} deg, F_t {trim['tether_force']:6.0f} N, "
            f"chi_dot {np.ravel(trim['opt_x'])[4]:+.3f} rad/s, stalled {stalled:.2f} | "
            f"{meta['n_iter']} it, {'converged' if meta['converged'] else 'NOT CONVERGED'}"
            f" | {(time.perf_counter() - started) / 60:.1f} min",
            flush=True,
        )
        if chain.max_stalled_fraction is not None and stalled > chain.max_stalled_fraction:
            print(f"  {chain.name}: stopped (stalled)", flush=True)
            break
    (out / chain.name).mkdir(parents=True, exist_ok=True)
    (out / chain.name / DONE_MARKER).write_text(f"{n_written} anchors\n", encoding="utf-8")
    return n_written


# ───────────────────────────────────────────── states off the sweep's centre
def open_window_case(spec: dict, kite: str, work_dir: Path, *, gravity: bool):
    """The wireframe case of the window states: tetherless (tether, gravity and
    shear are the ROM's point-mass physics, the aerodynamics are what is
    measured), gravity as asked."""
    spec = copy.deepcopy(spec)
    overrides = spec.setdefault("overrides", {})
    overrides["tether"] = {**(overrides.get("tether") or {}), "include_in_trim": False}
    overrides.setdefault("coupling", {})["is_with_gravity"] = bool(gravity)
    return open_case("wireframe", spec, kite, work_dir=work_dir)


def window_wind(wind_centre: float, elevation_deg: float, azimuth_deg: float) -> float:
    """The wind off-centre: scaled by 1 / (cos beta cos phi) so that its
    component along the tether is the centre's."""
    return wind_centre / (np.cos(np.radians(elevation_deg)) * np.cos(np.radians(azimuth_deg)))


def anchor_wind(sweep_out: Path, anchor: str) -> float:
    """The wind speed [m/s] a sweep anchor was solved at."""
    trims = _read_rows(sweep_out / ds.TRIMS_FILE)
    rows = None if trims is None else trims[trims.anchor == anchor]
    if rows is None or rows.empty:
        raise SystemExit(f"anchor {anchor} is not in {sweep_out / ds.TRIMS_FILE}")
    return float(rows.wind_speed.iloc[0])


def solve_window_state(driver, case, spec: dict, sweep_out: Path, base: str, state, *,
                       mass_scale: float, gravity: bool, wind_centre: float) -> dict | None:
    """Re-solve sweep anchor ``base`` (``<chain>/<step>``, restored from its
    sim_output.h5) at ``state`` = (elevation, azimuth, course) [deg] with the
    kite's mass scaled by ``mass_scale``. The trim row (with ``base``,
    ``state``, ``mass_scale``, ``gravity`` and the angles), or None when the
    trim fails."""
    elevation, azimuth, course = (float(v) for v in state)
    label = turn_law.state_label(elevation, azimuth, course)
    anchor = turn_law.state_anchor(base, label, mass_scale, gravity)
    chain, step = base.split("/")
    wind = window_wind(wind_centre, elevation, azimuth)
    config = _at(copy.deepcopy(case.config), wind_speed=wind, power_tape=0.0, steering=0.0)
    config = dataclasses.replace(
        Case.from_config(config), elevation_deg=elevation, azimuth_deg=azimuth, course_deg=course
    ).apply(config)
    config["power_tape_extension_step"] = float(spec["depower"]["step"])
    config["steering_tape_extension_step"] = float(spec["steering"]["step"])
    prepared = prepare(driver, case, spec, config)
    prepared.restore(sweep_out / chain / f"step_{int(step):02d}" / "sim_output.h5")
    prepared.m_arr = np.asarray(prepared.m_arr, dtype=float) * max(mass_scale, turn_law.MASS_FLOOR)
    final_state: dict = {}
    started = time.perf_counter()
    try:
        _, meta = driver.solve_step(case, prepared, config, final_state=final_state)
    except aerodynamic_vsm.CasadiTrimFailed as failure:
        print(f"{anchor}: TRIM FAILED {str(failure)[:160]}", flush=True)
        return None
    trim, body = final_state["results_aero"], final_state["body_aero"]
    m_arr = np.asarray(prepared.m_arr, dtype=float)
    row = ds.trim_row(
        anchor=anchor, chain=chain, step=int(step), wind_speed=wind,
        power_tape_length=prepared.power_tape_length,
        steering_half_difference=prepared.steering_half_difference,
        rho=air_density_at_case(config), mass_kcu=float(m_arr[0]), mass_total=float(m_arr.sum()),
        trim=trim, meta=meta, body_z=aerodynamic_vsm.body_centre_panel_axes(body)[2],
        polar_check=0.0,
    )
    row.update(base=base, state=label, mass_scale=float(mass_scale), gravity=bool(gravity),
               elevation_deg=elevation, azimuth_deg=azimuth, course_deg=course)
    print(f"{anchor}: converged {meta['converged']} ({meta['n_iter']} it), "
          f"alpha {trim['aoa_deg']:.3f} deg, F_t {trim['tether_force']:.0f} N, "
          f"omega_r {-row['omega_z']:+.4f} rad/s | {(time.perf_counter() - started) / 60:.1f} min",
          flush=True)
    return row


def turn_law_jobs(settings: dict) -> list[tuple[str, tuple, float, bool]]:
    """``(base, state, mass_scale, gravity)`` of identification.yaml's
    ``sweep.turn_law``: every ``inertia_anchors`` and ``gravity_anchors``
    anchor at the centre at mass scales 0 and 1 (gravity off; a and b), and
    every ``gravity_anchors`` anchor at each ``gravity_states`` (gravity on,
    full mass; c5, c6)."""
    gravity_anchors = list(settings.get("gravity_anchors") or [])
    bases = list(dict.fromkeys(list(settings.get("inertia_anchors") or []) + gravity_anchors))
    jobs = [(base, turn_law.CENTRE, scale, False) for base in bases for scale in (1.0, 0.0)]
    jobs += [(base, tuple(float(v) for v in state), 1.0, True)
             for base in gravity_anchors for state in settings.get("gravity_states") or []]
    return jobs


def solve_turn_law_states(spec: dict, kite: str, out: Path, bases: list[str] | None = None) -> None:
    """The states the turn-rate law's b(u_p), c5, c6 are measured on
    (awetrim.identification.turn_law), into ``out/turn_law/trims.csv``:
    resumable (a state already written is skipped), and runs on disjoint
    ``bases`` may share the folder."""
    settings = spec.get("turn_law")
    if not settings:
        raise SystemExit("--turn-law: identification.yaml sets no sweep.turn_law")
    folder = out / turn_law.TURN_LAW_DIR
    path = folder / ds.TRIMS_FILE
    jobs = [j for j in turn_law_jobs(settings) if not bases or j[0] in bases]
    for gravity in (False, True):
        with sweep_files_lock(folder):
            done = set(_read_rows(path).anchor) if path.exists() else set()
        todo = [j for j in jobs if j[3] == gravity and turn_law.state_anchor(
            j[0], turn_law.state_label(*j[1]), j[2], gravity) not in done]
        print(f"turn-law states, gravity {'on' if gravity else 'off'}: {len(todo)} to solve", flush=True)
        if not todo:
            continue
        driver, case = open_window_case(spec, kite, folder / "geometry", gravity=gravity)
        for base, state, scale, _ in todo:
            row = solve_window_state(driver, case, spec, out, base, state, mass_scale=scale,
                                     gravity=gravity, wind_centre=anchor_wind(out, base))
            if row is not None:
                with sweep_files_lock(folder):
                    ds.append_rows(path, [row])


def _read_rows(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path, keep_default_na=False, na_values=["nan", "NaN"])


def _step_of(anchors: pd.Series) -> pd.Series:
    return anchors.str.rsplit("/", n=1).str[-1].astype(int)


def resume_point(out: Path, chain_name: str) -> int:
    """The steps of an unfinished chain already complete -- trim row and
    polar rows written -- in an unbroken run from step 0."""
    trims, polars = _read_rows(out / ds.TRIMS_FILE), _read_rows(out / ds.POLARS_FILE)
    if trims is None or polars is None:
        return 0
    prefix = chain_name + "/"
    with_trim = set(_step_of(trims.anchor[trims.anchor.str.startswith(prefix)]))
    with_polar = set(_step_of(polars.anchor[polars.anchor.str.startswith(prefix)]))
    n = 0
    while n in with_trim and n in with_polar:
        n += 1
    return n


def _drop_chain_rows(out: Path, chain_name: str, keep_steps: int = 0) -> None:
    """Remove an unfinished chain's rows from step ``keep_steps`` on, so its
    rerun replaces them."""
    for name in (ds.TRIMS_FILE, ds.POLARS_FILE):
        path = out / name
        table = _read_rows(path)
        if table is not None:
            ours = table["anchor"].str.startswith(chain_name + "/")
            drop = ours & (_step_of(table["anchor"].where(ours, "x/0")) >= keep_steps)
            if drop.any():
                table[~drop].to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--kite", default="LEI-V3-KITE", help="folder under data/")
    parser.add_argument("--fidelity", choices=ds.FIDELITIES, default=None,
                        help="structural model (default: identification.yaml's)")
    parser.add_argument("--chains", nargs="+", help="solve only these chains")
    parser.add_argument("--wind-speeds", type=float, nargs="+", default=None,
                        help="replace identification.yaml's wind_speeds (e.g. extra chains "
                        "at higher wind; combine with --chains)")
    parser.add_argument("--list", action="store_true", help="list the chains and exit")
    parser.add_argument("--base-only", action="store_true",
                        help="solve the base states (start_from_converged) and exit; "
                        "run this first before launching parallel --chains runs")
    parser.add_argument("--repolar", action="store_true",
                        help="solve the polars of the saved anchors again over the current "
                        "polar_d_alpha_deg (wireframe; with --chains, those chains only)")
    parser.add_argument("--turn-law", nargs="*", default=None, metavar="ANCHOR",
                        help="solve only the turn-law states of sweep.turn_law (wireframe; "
                        "with anchors, those anchors' only, so runs can go in parallel). "
                        "A full run (no --chains) solves them after the chains")
    args = parser.parse_args()

    fidelity, spec = ds.identification_settings(
        PROJECT_DIR / "data" / args.kite / "identification.yaml", "sweep", args.fidelity
    )
    if args.wind_speeds:
        spec["wind_speeds"] = list(args.wind_speeds)
    out = PROJECT_DIR / "results" / args.kite / "identification" / fidelity / "sweep"
    driver, case = open_case(fidelity, spec, args.kite)
    built = prepare(driver, case, spec, case.config).power_tape_length
    chains = build_chains(spec, built)
    if args.list:
        for chain in chains:
            print(f"{chain.name}: {len(chain.steps)} states, wind {chain.wind_speed} m/s")
        return
    if args.chains:
        unknown = set(args.chains) - {c.name for c in chains}
        if unknown:
            raise SystemExit(f"unknown chains {sorted(unknown)}; see --list")
        chains = [c for c in chains if c.name in args.chains]

    print(f"{args.kite} ({fidelity}): built power-tape length {built:.3f} m, {len(chains)} chains "
          f"-> {out}", flush=True)
    if args.turn_law is not None or (spec.get("turn_law") and not args.chains):
        if fidelity != "wireframe":
            raise SystemExit("the turn-law states need the wireframe (Billow cannot restore a state)")
    if args.turn_law is not None:
        solve_turn_law_states(spec, args.kite, out, args.turn_law or None)
        return
    if args.repolar:
        if fidelity != "wireframe":
            raise SystemExit("--repolar needs the wireframe (Billow cannot restore a state)")
        for chain in chains:
            print(f"{chain.name}: repolar, {repolar_chain(driver, case, chain, spec, out)} anchors redone",
                  flush=True)
        return
    if spec.get("start_from_converged"):
        if fidelity != "wireframe":
            raise SystemExit("start_from_converged needs the wireframe (Billow cannot restore a state)")
        solve_base_states(driver, case, spec, out)
    elif args.base_only:
        raise SystemExit("--base-only: identification.yaml sets no start_from_converged")
    if args.base_only:
        return
    for chain in chains:
        if (out / chain.name / DONE_MARKER).exists():
            print(f"{chain.name}: done, skipped", flush=True)
            continue
        # The wireframe resumes after its last complete anchor; Billow's
        # beam frames are not in sim_output.h5, so its chain starts over.
        with sweep_files_lock(out):
            start = resume_point(out, chain.name) if fidelity == "wireframe" else 0
            _drop_chain_rows(out, chain.name, keep_steps=start)
        resumed = f", resumed at step {start}" if start else ""
        print(f"{chain.name}: {len(chain.steps)} states{resumed}", flush=True)
        solve_chain(driver, case, chain, spec, out, start=start)
    failed = [
        path.parent.relative_to(out).as_posix()
        for chain in chains
        for path in sorted((out / chain.name).glob(f"step_*/{TRIM_FAILED_FILE}"))
    ]
    if failed:
        print(f"{len(failed)} chain(s) stopped by a failed trim: {', '.join(failed)}", flush=True)
    if spec.get("turn_law") and not args.chains:
        solve_turn_law_states(spec, args.kite, out)


if __name__ == "__main__":
    main()

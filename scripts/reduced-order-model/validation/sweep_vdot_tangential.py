"""Tangential acceleration against tangential speed at flown states (paper Fig. 6).

Every flown 2019 reel-in state reconstructed by the quasi-steady ROM reads
d(v_dot_tau)/d(v_tau) > 0 at its root (speed-UNSTABLE in v_tau at fixed
controls), while reel-out reads < 0. Two readings: (i) the reel-in root is the
slow, unstable root of the paper's Fig. 6 and a fast stable root exists at the
same condition; (ii) there is no fast root at that depower, so the kite sits on
the only equilibrium there is. This sweep tells them apart.

At a few flown states (``validate_quasi_steady_state_v3.py`` CSV rows), with
the controls held at the reconstructed values (u_s, u_p; chi_dot fixed -- it
only enters the normal force row, which this criterion drops), it sweeps v_tau,
solves the RADIAL balance for the tether tension at each speed, and evaluates
v_dot_tau = g_chi / m from the tangential row of SystemModel.force_residual.
Zero crossings are the quasi-steady roots: a falling crossing is stable, a
rising one unstable. Alpha along the sweep shows where the ROM's stall lies.

Same model as the validator: system_flown_2019.yaml, the chosen ROM, a
logarithmic wind (z0 0.1) set by the sample's friction velocity, and the
validator's tether (``--tether``): a RigidLumpedTether, or the discretised
WilliamsTether (10 elements, inelastic, cf 0.01). With Williams the "radial
balance" is the radial force row PLUS the ground closure, solved at each speed
for the kite-end tension, the tether length and the last element's angles
(Newton, continued from the previous speed); the tangential row is then read
off as before, so the tether sags and drags at every point of the sweep.

Usage (project root; run the validator with --rom <rom> first)
    python scripts/reduced-order-model/validation/sweep_vdot_tangential.py --rom semi_empirical
    python scripts/reduced-order-model/validation/sweep_vdot_tangential.py --tether williams
"""

from __future__ import annotations

import argparse
import contextlib
import io
from pathlib import Path

import casadi as ca
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import brentq

from awetrim import SystemModel
from awetrim.environment.Wind import Wind
from awetrim.plotting.plotting import PALETTE, set_plot_style
from awetrim.system.factory import create_system_model_from_yaml
from awetrim.system.tether import RigidLumpedTether
from awetrim.system.williams_tether import WilliamsTether
from awetrim.utils.config_paths import LEI_V3_SYSTEM_FLOWN_CONFIG

ROOT = Path("results/LEI-V3-KITE/rom_validation")
ROM_FILES = {"semi_empirical": "rom_config_semi_empirical.yaml",
             "aerostructural": "rom_config_aerostructural.yaml",
             "aerostructural_flight": "rom_config_aerostructural_flight_corrected.yaml"}
SPEEDS = np.linspace(0.5, 45.0, 90)
TENSION_BRACKET = (1.0, 3.0e5)


def build_model(rom: str, tether_kind: str = "rigid") -> SystemModel:
    """The validator's model (validate_quasi_steady_state_v3.py --tether)."""
    with contextlib.redirect_stdout(io.StringIO()):
        factory = create_system_model_from_yaml(
            LEI_V3_SYSTEM_FLOWN_CONFIG,
            aero_yaml_path=LEI_V3_SYSTEM_FLOWN_CONFIG.parent / ROM_FILES[rom],
            steering_control="asymmetric",
        )
    diameter = factory.tether.diameter_tether
    tether = (WilliamsTether(diameter=diameter, n_elements=10, elastic=False, cf=0.01)
              if tether_kind == "williams" else RigidLumpedTether(diameter=diameter))
    wind = Wind(wind_model="logarithmic", z0=0.1, direction_wind=0)
    return SystemModel(dof=3, quasi_steady=True, kite=factory.kite,
                       tether=tether, wind_model=wind)


def residual_functions(model):
    """Tangential and radial force rows and alpha over all their symbols."""
    g = model.force_residual
    outputs = ca.vertcat(g[0], g[2], model.kite.angle_of_attack_for(model))
    symbols = ca.symvar(outputs)
    return ca.Function("rows", symbols, [outputs]), [s.name() for s in symbols]


TENSION_GRID = np.geomspace(*TENSION_BRACKET, 400)


def sweep_state(function, names, state, mass, v_root, tension_root):
    """v_dot_tau(v_tau) and alpha(v_tau), tension from the radial balance.

    The radial row is NOT monotonic in tension (tension also turns the bridle
    resultant, hence alpha), so it can have two tension roots at one speed.
    Every sign change on a log grid is refined, and the sweep is CONTINUED
    from the reconstructed root (v_root, tension_root) outward in both
    directions, taking at each speed the tension root nearest the previous
    one: it traces the branch through the flown state.
    """
    def rows(v_tau, tension):
        values = dict(state, speed_tangential=v_tau, tension_tether_ground=tension)
        return np.asarray(function(*[values[n] for n in names]), dtype=float).ravel()

    def tension_roots(v):
        g = np.array([rows(v, t)[1] for t in TENSION_GRID])
        out = []
        for k in np.flatnonzero(np.sign(g[:-1]) * np.sign(g[1:]) < 0):
            out.append(brentq(lambda t: rows(v, t)[1], TENSION_GRID[k],
                              TENSION_GRID[k + 1], xtol=1e-6))
        return out

    vdot, alpha = np.full(SPEEDS.size, np.nan), np.full(SPEEDS.size, np.nan)
    start = int(np.argmin(np.abs(SPEEDS - v_root)))
    for direction in (range(start, SPEEDS.size), range(start - 1, -1, -1)):
        previous = tension_root
        for k in direction:
            roots = tension_roots(SPEEDS[k])
            if not roots:
                break  # the branch ends: no tension balances the radial row
            tension = min(roots, key=lambda t: abs(np.log(t / previous)))
            g_chi, _, a = rows(SPEEDS[k], tension)
            vdot[k], alpha[k], previous = g_chi / mass, a, tension
    return vdot, alpha


def williams_functions(model):
    """Newton solve of radial row + ground closure for the tether unknowns.

    Returns (solve, rows, names): solve(x0, p) -> x; rows(x, p) -> ([g_chi,
    alpha], residual norm); x = (tension_kite, length, azimuth_last,
    elevation_last); p = every other symbol, in ``names`` order.
    """
    tether = model.tether
    x = ca.vertcat(tether.tension_tether_kite, tether.tether_length,
                   tether.azimuth_last_element, tether.elevation_last_element)
    shape = tether.tether_shape_symbolic_for(model)
    g = model.force_residual
    residual = ca.vertcat(g[2], shape["ground_position"])
    out = ca.vertcat(g[0], model.kite.angle_of_attack_for(model))
    unknowns = {x[i].name() for i in range(4)}
    others = [s for s in ca.symvar(ca.vertcat(residual, out)) if s.name() not in unknowns]
    p = ca.vertcat(*others)
    solve = ca.rootfinder("radial_ground", "newton",
                          ca.Function("res", [x, p], [residual]),
                          {"error_on_fail": False, "max_iter": 50, "abstol": 1e-8})
    rows = ca.Function("rows", [x, p], [out, ca.norm_2(residual)])
    return solve, rows, [s.name() for s in others]


def sweep_state_williams(functions, state, mass, v_root, tension_root):
    """Williams counterpart of sweep_state: continuation in the tether unknowns.

    Converged first at the reconstructed speed from a straight-tether guess,
    then continued outward; a branch ends where Newton no longer closes the
    radial balance (the fold) or the tether goes slack.
    """
    solve, rows, names = functions

    def at(v_tau, x0):
        values = dict(state, speed_tangential=v_tau)
        p = [values[n] for n in names]
        x = np.asarray(solve(x0, p), dtype=float).ravel()
        out, res = rows(x, p)
        out = np.asarray(out, dtype=float).ravel()
        ok = float(res) < 1e-5 and x[0] > 0.0 and np.all(np.isfinite(out))
        return (x if ok else None), out

    vdot, alpha = np.full(SPEEDS.size, np.nan), np.full(SPEEDS.size, np.nan)
    start = int(np.argmin(np.abs(SPEEDS - v_root)))
    # The CSV stores the GROUND tension; the kite end differs by the tether
    # weight and drag, close enough to start Newton.
    guess = np.array([tension_root, state["distance_radial"],
                      state["angle_azimuth"], state["angle_elevation"]])
    x_root, _ = at(v_root, guess)
    if x_root is None:
        return vdot, alpha
    for direction in (range(start, SPEEDS.size), range(start - 1, -1, -1)):
        previous = x_root
        for k in direction:
            x, out = at(SPEEDS[k], previous)
            if x is None:
                break
            vdot[k], alpha[k], previous = out[0] / mass, out[1], x
    return vdot, alpha


def crossings(vdot):
    """Indices just before each sign change, with the slope sign there."""
    out = []
    for k in range(len(vdot) - 1):
        if np.isfinite(vdot[k]) and np.isfinite(vdot[k + 1]) and vdot[k] * vdot[k + 1] < 0:
            v0 = SPEEDS[k] - vdot[k] * (SPEEDS[k + 1] - SPEEDS[k]) / (vdot[k + 1] - vdot[k])
            out.append((v0, "stable" if vdot[k + 1] < vdot[k] else "unstable"))
    return out


def pick_states(data, phase, n):
    reel_out = (data.speed_radial > 0.5) & (data.input_depower < 1.9) & (data.kite_elevation < 0.75)
    reel_in = (data.speed_radial < -0.5) & (data.input_depower >= 1.9)
    rows = data[(reel_in if phase == "reel-in" else reel_out) & data.predicted_tension.notna()]
    return rows.iloc[np.linspace(0, len(rows) - 1, n).astype(int)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--rom", choices=sorted(ROM_FILES), default="aerostructural_flight")
    parser.add_argument("--n", type=int, default=4, help="states per phase")
    parser.add_argument("--tether", choices=("rigid", "williams"), default="rigid")
    parser.add_argument("--csv", type=Path, default=None,
                        help="validator CSV (default: <rom>/qs_validation[_williams].csv)")
    args = parser.parse_args()
    set_plot_style()
    suffix = "_williams" if args.tether == "williams" else ""

    data = pd.read_csv(args.csv or ROOT / args.rom / f"qs_validation{suffix}.csv")
    if "distance_radial" not in data:
        raise SystemExit("re-run the validator: the CSV has no per-sample condition")
    model = build_model(args.rom, args.tether)
    if args.tether == "williams":
        functions = williams_functions(model)
    else:
        function, names = residual_functions(model)
    mass = float(model.kite.mass_wing + model.kite.mass_kcu)
    alpha_stall = float(model.kite.aero_params["angle_of_attack_stall"])

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 7.5), sharex=True)
    colours = [PALETTE[c] for c in ("Blue", "Vermillion", "Bluish Green", "Reddish Purple",
                                    "Orange", "Sky Blue")]
    for col, phase in enumerate(("reel-out", "reel-in")):
        ax_v, ax_a = axes[0, col], axes[1, col]
        for colour, (_, row) in zip(colours, pick_states(data, phase, args.n).iterrows()):
            state = {
                "distance_radial": row.distance_radial, "angle_course": row.angle_course,
                "speed_radial": row.speed_radial, "angle_azimuth": row.angle_azimuth,
                "angle_elevation": row.kite_elevation, "speed_friction": row.speed_friction,
                "timeder_angle_course": row.measured_course_rate,
                "input_depower": row.input_depower,
                "input_steering": row.predicted_input_steering,
                "timeder_speed_tangential": 0.0, "timeder_speed_radial": 0.0,
            }
            if args.tether == "williams":
                vdot, alpha = sweep_state_williams(
                    functions, state, mass, row.predicted_speed_tangential,
                    row.predicted_tension)
            else:
                vdot, alpha = sweep_state(function, names, state, mass,
                                          row.predicted_speed_tangential,
                                          row.predicted_tension)
            roots = crossings(vdot)
            label = (f"t {row.time:.0f} s, $v_r$ {row.speed_radial:+.1f}, "
                     f"$u_p$ {row.input_depower:.2f}: "
                     + ", ".join(f"{v:.1f} {s}" for v, s in roots))
            ax_v.plot(SPEEDS, vdot, color=colour, lw=1.6, label=label)
            ax_v.axvline(row.measured_speed_tangential, color=colour, lw=0.8, ls=":")
            ax_a.plot(SPEEDS, np.degrees(alpha), color=colour, lw=1.6)
            print(f"{phase} t={row.time:.1f}: measured v_tau {row.measured_speed_tangential:.1f}, "
                  f"reconstructed {row.predicted_speed_tangential:.1f}; roots {roots}")
        ax_v.axhline(0.0, color=PALETTE["Black"], lw=0.8)
        ax_a.axhline(np.degrees(alpha_stall), color=PALETTE["Black"], lw=0.8, ls="--")
        ax_v.set_title(f"{phase} (dotted: measured $v_\\tau$)", fontsize=10)
        ax_v.set_ylim(-60, 60)
        ax_v.legend(frameon=False, fontsize=7, loc="lower left")
        ax_a.set_xlabel(r"$v_\tau$ (m s$^{-1}$)")
        ax_a.set_ylim(-20, 60)
    axes[0, 0].set_ylabel(r"$\dot v_\tau$ (m s$^{-2}$)")
    axes[1, 0].set_ylabel(r"$\alpha_w$ ($^\circ$) (dashed: stall)")
    fig.suptitle(f"Tangential acceleration at flown 2019 states, {args.rom.replace('_', '-')} "
                 f"ROM, {args.tether} tether, controls fixed (paper Fig. 6)", fontsize=10)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(ROOT / args.rom / f"vdot_sweep{suffix}.{ext}", dpi=200)
    print(f"Wrote {ROOT / args.rom / ('vdot_sweep' + suffix)}.png/.pdf")


if __name__ == "__main__":
    main()

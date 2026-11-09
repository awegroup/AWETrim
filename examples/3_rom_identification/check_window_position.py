"""Does the identification depend on where in the wind window it is flown?

The identification sweep (run_identification_sweep.py) solves every state at
the centre of the wind window. This check re-solves saved sweep anchors (the
coupled solve, restored from their sim_output.h5) at other elevation, azimuth
and course, tetherless, the wind scaled by 1 / (cos beta cos phi) so that the
wind component along the tether is the centre's (the sweep's own window-state
solve, run_identification_sweep.solve_window_state). Tether, gravity and
shear are the ROM's point-mass physics; what is tested here is the
aerodynamics the ROM is identified on.

Each state can also be solved with the kite's mass scaled (``--mass-scales``;
0 is taken as 1e-6 so the inertial load still distributes over the nodes) and
with gravity (``--gravity``, states labelled @g).

Analysis (``--analyse-only`` or after solving), per anchor:

1. Residuals of the identified ROM (system.yaml's ROM file, or ``--rom``) at
   each state's (alpha, u_p, u_s, v_a): CL, CD, theta_b(u_p) and the roll law
   phi_a,w = k(u_p, u_s) u_s. Off-centre residuals that stay inside the
   centre's say the centre sampling is sufficient for that law.
2. The turn-rate law (Cayon & Schmehl, in preparation, Eq. 33;
   awetrim.identification.turn_law) with the anchor's a and b from its centre
   at mass scale 0 and 1, and c5, c6 from every gravity-on state, PREDICTING
   every other state; next to it the roll law's implied turn rate at the
   centre's roll, ``w_c (L / L_c) (v_tau_c / v_tau)`` (the lateral force
   balance at a fixed roll).

The turn law's b(u_p), c5, c6 for the ROM are measured by the sweep itself
(run_identification_sweep.py --turn-law); this script checks the law where the
sweep does not go.

Usage (project root):
    python examples/3_rom_identification/check_window_position.py
    python examples/3_rom_identification/check_window_position.py --mass-scales 1 0.5 0
    python examples/3_rom_identification/check_window_position.py --gravity --states 30,0,0 30,0,180
    python examples/3_rom_identification/check_window_position.py --analyse-only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_identification_sweep as sweep  # noqa: E402
from awetrim.identification import aerostructural_dataset as ds  # noqa: E402
from awetrim.identification import turn_law as tl  # noqa: E402

PROJECT_DIR = sweep.PROJECT_DIR
#: (elevation, azimuth, course) [deg]
DEFAULT_STATES = [
    tl.CENTRE,
    (30.0, 0.0, 90.0),
    (30.0, 0.0, 0.0),
    (30.0, 0.0, 180.0),
    (30.0, 30.0, 45.0),
    (30.0, -30.0, 135.0),
    (50.0, 20.0, 270.0),
]
DEFAULT_ANCHORS = ["w5.0_steering_1.95/00", "w5.0_steering_1.95/07"]


def parse_state(text: str) -> tuple[float, float, float]:
    beta, phi, chi = (float(v) for v in text.split(","))
    return beta, phi, chi


def solve(args) -> None:
    _, spec = ds.identification_settings(
        PROJECT_DIR / "data" / args.kite / "identification.yaml", "sweep", "wireframe"
    )
    args.out.mkdir(parents=True, exist_ok=True)
    driver, case = sweep.open_window_case(spec, args.kite, args.out / "geometry", gravity=args.gravity)
    path = args.out / ds.TRIMS_FILE
    done = set(pd.read_csv(path).anchor) if path.is_file() else set()
    for base in args.anchors:
        wind = args.wind_centre or sweep.anchor_wind(args.sweep, base)
        for mass_scale in args.mass_scales:
            for state in args.states:
                if tl.state_anchor(base, tl.state_label(*state), mass_scale, args.gravity) in done:
                    continue
                row = sweep.solve_window_state(driver, case, spec, args.sweep, base, state,
                                               mass_scale=mass_scale, gravity=args.gravity,
                                               wind_centre=wind)
                if row is not None:
                    ds.append_rows(path, [row])


def rom_residuals(trims: pd.DataFrame, rom_path: Path, steering_tape_per_unit_us: float):
    """The identified ROM's laws minus the coupled solve, per state."""
    import yaml

    from awetrim.identification.aero_polynomial import fits_from_rom_aerodynamics
    from awetrim.system.kite import roll_gain, stall_blend

    aero = yaml.safe_load(rom_path.read_text(encoding="utf-8"))["aerodynamics"]
    p = aero["params"]
    fits = fits_from_rom_aerodynamics(aero)
    empty_polars = pd.DataFrame(columns=["anchor", "d_alpha_deg", "gamma_converged"]
                                + [f"{v}_{c}" for v in ("va", "force_aero") for c in "xyz"])
    samples, s = ds.coefficient_tables(trims, empty_polars, reference_area=aero["reference_area"],
                                       steering_tape_per_unit_us=steering_tape_per_unit_us)
    s = s.merge(samples[["anchor", "CL", "CD"]], on="anchor")
    x = {"alpha": s.alpha_w.to_numpy(), "u_p": s.u_p.to_numpy(), "u_s": s.u_s.to_numpy(),
         "v_a": s.v_a.to_numpy()}
    x["stall"] = stall_blend(x["alpha"], p["angle_of_attack_stall"], p["width_stall"], xp=np)
    k = roll_gain(p["gain_roll_steering"], p.get("gain_roll_steering_depower") or [],
                  p.get("gain_roll_steering_squared") or [], x["u_p"], x["u_s"])
    return pd.DataFrame({
        "anchor": s.anchor,
        "alpha_deg": np.degrees(s.alpha_w),
        "dCL": s.CL - fits["CL"].predict(x),
        "dCD": s.CD - fits["CD"].predict(x),
        "dtheta_b_deg": np.degrees(s.theta_b - (p["angle_pitch_tether_0"]
                                                + p["slope_angle_pitch_tether_depower"] * s.u_p)),
        "phi_a_w_deg": np.degrees(s.phi_a_w),
        "dphi_a_w_deg": np.degrees(s.phi_a_w - k * x["u_s"]),
    })


def lift(t: pd.DataFrame) -> np.ndarray:
    f = ds.vector(t, "force_aero")
    va = ds.vector(t, "va")
    e = va / np.linalg.norm(va, axis=1)[:, None]
    return np.linalg.norm(f - np.sum(f * e, axis=1)[:, None] * e, axis=1)


def predict(g: pd.DataFrame, terms: pd.Series, c5: float, c6: float) -> pd.DataFrame:
    """Eq. 33 with the anchor's a, b and the measured c5, c6 at every state,
    against the coupled turn rate; and the roll law at the centre's roll."""
    a, b = terms.a, terms.b
    beta, chi = np.radians(g.elevation_deg), np.radians(g.course_deg)
    weight = np.where(g.gravity, g.mass * tl.GRAVITY, 0.0)
    # F_g = -m g [cos chi cos beta, sin chi cos beta, sin beta]
    f_chi, f_n, f_r = (-weight * np.cos(chi) * np.cos(beta), -weight * np.sin(chi) * np.cos(beta),
                       -weight * np.sin(beta))
    law = ((a * g.v_a**2 * g.u_s - b * f_n - c5 * f_chi - c6 * f_r)
           / (g.v_a + b * g.mass * g.v_tau))
    centre = g[(g.state == tl.state_label(*tl.CENTRE)) & (g.mass_scale == 1.0) & ~g.gravity].iloc[0]
    roll = (centre.w * (g.L / centre.L) * (centre.v_tau / g.v_tau) * (centre.mass / g.mass))
    return pd.DataFrame({
        "state": g.state, "mass_scale": g.mass_scale, "gravity": g.gravity, "v_tau": g.v_tau,
        "v_a": g.v_a, "w_coupled": g.w, "w_turn_law": law,
        "err_turn_law_pct": 100 * (law / g.w - 1),
        "err_roll_law_pct": np.where((g.mass_scale > 0) & ~g.gravity, 100 * (roll / g.w - 1), np.nan),
    })


def analyse(args) -> None:
    from awetrim.system.factory import resolve_rom_config_path

    rom = args.rom or resolve_rom_config_path(PROJECT_DIR / "data" / args.kite / "system.yaml")
    _, fit = ds.identification_settings(
        PROJECT_DIR / "data" / args.kite / "identification.yaml", "fit", "wireframe")
    per_us = float(fit["steering_tape_per_unit_us"])
    raw = pd.read_csv(args.out / ds.TRIMS_FILE)
    t = tl.turn_law_table(raw, steering_tape_per_unit_us=per_us)
    t = t.merge(rom_residuals(raw, Path(rom), per_us), on="anchor")
    t["L"] = lift(t)
    inertia = tl.inertia_terms(t)
    c5, c6, gravity = tl.gravity_terms(t, inertia)
    pd.set_option("display.width", 220)
    fmt = lambda x: f"{x:8.4f}"  # noqa: E731
    print(f"ROM: {rom}")
    for base, g in t.groupby("base"):
        print(f"\n=== {base}: identified laws, coupled minus ROM ===")
        cols = ["state", "mass_scale", "gravity", "alpha_deg", "dCL", "dCD", "dtheta_b_deg",
                "phi_a_w_deg", "dphi_a_w_deg"]
        print(g[cols].sort_values(["gravity", "mass_scale", "state"]).to_string(index=False, float_format=fmt))
        if base not in set(inertia.base):
            print("(turn law: needs the centre at mass scales 0 and 1 and nonzero steering)")
            continue
        terms = inertia.set_index("base").loc[base]
        print(f"\n--- turn law: a = {terms.a:.4f} rad/m, b = {terms.b:.5f} 1/kg (inertia share "
              f"{terms.inertia_share:.2f}), c5 {c5:.5f}, c6 {c6:.5f} 1/kg; a, b from the centre at "
              "mass 0 and 1, the rest predicted ---")
        print(predict(g, terms, c5, c6).sort_values(["gravity", "mass_scale", "state"])
              .to_string(index=False, float_format=fmt))
    if not inertia.empty:
        print("\n=== b = k3/k2 per anchor (centre, gravity off, mass 0 and 1) ===")
        print(inertia.sort_values("u_p").to_string(index=False, float_format=lambda x: f"{x:9.5f}"))
    if not gravity.empty:
        print(f"\n=== gravity on: c5 = {c5:.5f}, c6 = {c6:.5f} 1/kg (least squares over "
              f"{len(gravity)} states) ===")
        print(gravity.to_string(index=False, float_format=lambda x: f"{x:9.5f}"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--kite", default="LEI-V3-KITE")
    parser.add_argument("--sweep", type=Path, default=None,
                        help="the wireframe sweep folder (default results/<kite>/identification/"
                        "wireframe/sweep)")
    parser.add_argument("--out", type=Path, default=None,
                        help="default results/<kite>/identification/wireframe/window_check")
    parser.add_argument("--anchors", nargs="+", default=DEFAULT_ANCHORS,
                        help="sweep anchors <chain>/<step>")
    parser.add_argument("--states", nargs="+", type=parse_state, default=DEFAULT_STATES,
                        help="elevation,azimuth,course [deg], e.g. 30,0,90")
    parser.add_argument("--mass-scales", nargs="+", type=float, default=[1.0, 0.0])
    parser.add_argument("--gravity", action="store_true",
                        help="solve with gravity on (states labelled @g)")
    parser.add_argument("--wind-centre", type=float, default=None,
                        help="wind at the centre [m/s] (default each anchor's sweep wind); "
                        "off-centre it is scaled by 1 / (cos beta cos phi)")
    parser.add_argument("--rom", type=Path, default=None, help="ROM file (default system.yaml's)")
    parser.add_argument("--analyse-only", action="store_true")
    args = parser.parse_args()
    root = PROJECT_DIR / "results" / args.kite / "identification" / "wireframe"
    args.sweep = args.sweep or root / "sweep"
    args.out = args.out or root / "window_check"
    if not args.analyse_only:
        solve(args)
    analyse(args)


if __name__ == "__main__":
    main()

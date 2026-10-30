"""Identify the reduced-order model from the aerostructural sweep (step 3, part 2).

Every parameter of the ROM (Cayon, van Deursen, Schmehl 2026, WES 11, 1097)
from the coupled VSM + Billow states run_identification_sweep.py solved at the
CENTRE OF THE WIND WINDOW (gravity off) -- no flight data. Reads
results/<kite>/identification/<fidelity>/sweep/ (format in
awetrim.identification.aerostructural_dataset) and the kite constants of
data/<kite>/identification.yaml (`fit:`, with its entry for the fidelity).

What is identified, against the ROM's own definitions (awetrim.system.kite)
    theta_b(u_p)  paper Eqs. 1-2, alpha_w = alpha_b - theta_b, from the trims:
                  alpha_b is the paper Eq. D11 angle of the bridle resultant
                  F_b the ROM builds (tether + KCU drag and the KCU's share of
                  the inertia: the KCU hangs at the bridle point), alpha_w the
                  centre-panel angle. Linear in u_p over the fitted band (the
                  ROM's form).
    C_L, C_D      polynomials in (alpha, u_p, u_s, v_a) on the trims and their
                  frozen-shape polars: the trims alone tie alpha to u_p almost
                  one to one, the polars separate them. C_D is the WING's: the
                  bridle drag is in, the KCU drag is NOT
                  (kcu_drag_in_coefficients: false -- the ROM adds it from the
                  system file's KCU hardware, so a KCU swap needs no
                  re-identification), nor the tether (the ROM models it).
                  Terms chosen by forward selection (then backward
                  elimination) on CROSS-VALIDATION RMSE with folds by ANCHOR
                  (one frozen shape never straddles train and test).
    gain_roll_steering  paper Eq. 43, phi_a,w = k u_s: the roll of the lift
                  about v_a relative to F_b, on the steering chains; with
                  gain_roll_steering_depower, k = k(u_p) a polynomial in the
                  tape (degree 0-2 by leave-one-chain-out CV).

Conventions: u_p = power-tape length l_dp [m]; u_s = steering-tape
half-difference / fit.steering_tape_per_unit_us; alpha in rad; coefficients on
fit.reference_area, stated in the file as reference_area.

Output, results/<kite>/identification/<fidelity>/rom_aerostructural/:
    rom_config_aerostructural.yaml   the ROM (or --out)
    polar_samples.csv, trim_samples.csv, identification_report.json

The ROM file is in the format awetrim.system.kite reads: name it in the
kite's system.yaml (models.reduced_order.aerodynamics) to fly it in step 4.
The LEI-V3 ROM shipped in data/LEI-V3-KITE/rom_config_aerostructural.yaml was
identified this way on a larger campaign with the wireframe model.

Usage (project root)
    python examples/3_rom_identification/identify_rom_aerostructural.py
    python examples/3_rom_identification/identify_rom_aerostructural.py --fidelity wireframe
    python examples/3_rom_identification/identify_rom_aerostructural.py --no-write
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import yaml

from awetrim.identification import aero_polynomial as ap
from awetrim.identification import aerostructural_dataset as ds
from awetrim.system.kite import stall_blend

PROJECT_DIR = Path(__file__).resolve().parents[2]

#: Forward selection stops when the best term improves CV RMSE by < this.
MIN_RELATIVE_IMPROVEMENT = 0.01
CV_FOLDS = 5

# Candidate library. u_s enters through even powers only (left/right
# symmetry). The ROM's solvers do not bound alpha, so the model must behave
# outside the operating range. No alpha^3: a cubic C_D went negative beyond
# the data (a spurious reel-in root at alpha -31 deg) and a cubic C_L cannot
# follow the coupled model's stall. The STALL is the ``stall`` variable
# instead (awetrim.system.kite.stall_blend, a smooth step at
# angle_of_attack_stall): terms multiplied by it act only past the stall,
# C = C_attached + stall * (C_separated - C_attached). Every term is a plain,
# smooth monomial.
CANDIDATES_CL = [
    {"alpha": 1}, {"alpha": 2},
    {"u_p": 1}, {"u_p": 2}, {"alpha": 1, "u_p": 1}, {"alpha": 2, "u_p": 1},
    {"u_s": 2}, {"alpha": 1, "u_s": 2}, {"u_p": 1, "u_s": 2},
    {"v_a": 1}, {"alpha": 1, "v_a": 1}, {"u_p": 1, "v_a": 1},
]
STALL_TERMS = [
    {"stall": 1}, {"stall": 1, "alpha": 1}, {"stall": 1, "alpha": 2},
    {"stall": 1, "u_p": 1}, {"stall": 1, "u_s": 2},
]
CANDIDATES_CL = CANDIDATES_CL + STALL_TERMS
CANDIDATES_CD = list(CANDIDATES_CL)
#: Stall-location grid [deg], searched on grouped CV of the full C_L library.
STALL_ANGLE_GRID_DEG = tuple(np.arange(12.0, 22.01, 0.5))
STALL_WIDTH_GRID_DEG = (0.5, 1.0, 1.5)
#: A polar is kept only if its 0 step re-solves the coupled trim's force to
#: this relative accuracy (it re-solves the trim's own body, so ~1e-15).
POLAR_CHECK_TOLERANCE = 1e-3


# ───────────────────────────────────────────────────────────────── fits
def fit_theta_b(trims, band) -> dict:
    """Linear theta_b(u_p) on the unsteered trims inside the band."""
    rows = trims[(trims.u_s == 0.0) & trims.u_p.between(*band)]
    slope, intercept = np.polyfit(rows.u_p, rows.theta_b, 1)
    resid = rows.theta_b - (intercept + slope * rows.u_p)
    return {
        "angle_pitch_tether_0": float(intercept),
        "slope_angle_pitch_tether_depower": float(slope),
        "rms_deg": float(np.rad2deg(resid.std())),
        "max_deg": float(np.rad2deg(resid.abs().max())),
        "n": int(len(rows)),
    }


def fit_roll_gain(trims, max_degree: int = 2) -> dict:
    """phi_a,w = k(u_p) u_s through the origin on the steered trims, with
    k(u_p) = k0 + k1 u_p + ... a polynomial whose degree (0..max_degree) is
    chosen by leave-one-chain-out cross-validation RMSE (a chain's states
    share one tape, so a chain never straddles train and test)."""
    rows = trims[trims.u_s != 0.0]
    if rows.empty:
        raise SystemExit("no steered anchors in the sweep: cannot fit the roll gain")
    u_s, u_p, phi = (rows[c].to_numpy(float) for c in ("u_s", "u_p", "phi_a_w"))
    chains = rows.chain.to_numpy()
    # at most one fewer coefficient than distinct tapes
    max_degree = min(max_degree, len(np.unique(u_p.round(3))) - 1)

    def design(deg, mask=slice(None)):
        return np.column_stack([u_s[mask] * u_p[mask] ** p for p in range(deg + 1)])

    cv = {}
    for deg in range(max_degree + 1):
        sq = []
        for c in np.unique(chains):
            test = chains == c
            coef, *_ = np.linalg.lstsq(design(deg, ~test), phi[~test], rcond=None)
            sq.append((phi[test] - design(deg, test) @ coef) ** 2)
        cv[deg] = float(np.sqrt(np.concatenate(sq).mean()))
    deg = min(cv, key=cv.get)
    coef, *_ = np.linalg.lstsq(design(deg), phi, rcond=None)
    resid = phi - design(deg) @ coef
    out = {"gain_roll_steering": float(coef[0]),
           "rms_deg": float(np.rad2deg(resid.std())),
           "cv_rms_deg": {d: float(np.rad2deg(v)) for d, v in cv.items()},
           "degree": int(deg), "n": int(len(rows))}
    if deg:
        out["gain_roll_steering_depower"] = [float(c) for c in coef[1:]]
    return out


def regressors(polar, stall) -> dict[str, np.ndarray]:
    """ROM regressors of the samples, with the stall switch at ``stall``."""
    data = {c: polar[c].to_numpy() for c in ("alpha", "u_p", "u_s", "v_a")}
    data["stall"] = stall_blend(data["alpha"], stall["angle_of_attack_stall"],
                                stall["width_stall"], xp=np)
    return data


def locate_stall(polar) -> dict:
    """Stall angle and width minimising the grouped CV RMSE of the FULL C_L
    library (fixed terms: fast, and separates locating the stall from
    choosing the terms)."""
    groups = polar["anchor"].to_numpy()
    best = None
    for angle in STALL_ANGLE_GRID_DEG:
        for width in STALL_WIDTH_GRID_DEG:
            stall = {"angle_of_attack_stall": np.deg2rad(angle),
                     "width_stall": np.deg2rad(width)}
            cv = ap._cv_rmse(regressors(polar, stall), polar["CL"].to_numpy(),
                             CANDIDATES_CL, abs_basis=False, folds=CV_FOLDS,
                             groups=groups)
            if best is None or cv < best[0]:
                best = (cv, stall)
    print(f"stall: {np.rad2deg(best[1]['angle_of_attack_stall']):.1f} deg, width "
          f"{np.rad2deg(best[1]['width_stall']):.1f} deg (full-library C_L CV {best[0]:.5f})")
    return best[1]


def fit_coefficients(polar, stall) -> dict[str, ap.PolynomialFit]:
    data = regressors(polar, stall)
    groups = polar["anchor"].to_numpy()
    fits = {}
    for target, library in (("CL", CANDIDATES_CL), ("CD", CANDIDATES_CD)):
        print(f"\n--- {target}: forward selection on grouped {CV_FOLDS}-fold CV RMSE")
        fits[target] = ap.select_model(
            data, polar[target].to_numpy(), target=target,
            candidate_terms=library, regressors=tuple(data),
            criterion="cv", cv_groups=groups, cv_folds=CV_FOLDS,
            min_relative_improvement=MIN_RELATIVE_IMPROVEMENT, backward=True,
            verbose=True,
        )
    return fits


def _grid(alpha, up, us, va, stall):
    data = {"alpha": alpha, "u_p": np.full_like(alpha, up),
            "u_s": np.full_like(alpha, us), "v_a": np.full_like(alpha, va)}
    data["stall"] = stall_blend(alpha, stall["angle_of_attack_stall"],
                                stall["width_stall"], xp=np)
    return data


def check_extrapolation(fits, polar, stall, band) -> None:
    """C_D must stay positive over +-30 deg alpha across the fitted u_p, u_s
    and v_a ranges: a ROM solver that wanders outside the data must meet more
    drag, never thrust."""
    alpha = np.deg2rad(np.linspace(-30.0, 30.0, 121))
    worst = np.inf
    for up in np.linspace(*band, 7):
        for us in (0.0, float(polar.u_s.abs().max())):
            for va in (float(polar.v_a.min()), float(polar.v_a.max())):
                worst = min(worst, float(fits["CD"].predict(_grid(alpha, up, us, va, stall)).min()))
    print(f"extrapolation check: min C_D over +-30 deg = {worst:.4f}")
    positive = alpha >= 0.0
    va_mid = float(polar.v_a.median())
    for up in band:
        cl = fits["CL"].predict(_grid(alpha, up, 0.0, va_mid, stall))
        i = int(np.argmax(np.where(positive, cl, -np.inf)))
        print(f"  C_L max at u_p {up} m: {cl[i]:.3f} at alpha {np.rad2deg(alpha[i]):.1f} deg; "
              f"C_L(30 deg) {cl[-1]:.3f}, C_L(-30 deg) {cl[0]:.3f}")
    if worst <= 0.0:
        print("WARNING: the C_D polynomial goes non-positive outside the data; "
              "restrict the candidate library or widen the sweep")


# ───────────────────────────────────────────────────────────────── output
def rom_document(fits, theta, roll, polar, fit_spec, stall, kite, n_trims, fidelity) -> dict:
    aero = ap.build_rom_aerodynamics(
        fits.values(),
        extra_params={
            "angle_pitch_tether_0": theta["angle_pitch_tether_0"],
            "slope_angle_pitch_tether_depower": theta["slope_angle_pitch_tether_depower"],
            "gain_roll_steering": roll["gain_roll_steering"],
            **({"gain_roll_steering_depower": roll["gain_roll_steering_depower"]}
               if "gain_roll_steering_depower" in roll else {}),
            **{k: float(v) for k, v in stall.items()},
        },
    )
    aero = {"model": aero["model"], "reference_area": float(fit_spec["reference_area"]),
            "kcu_drag_in_coefficients": False,
            "params": aero["params"], "coefficients": aero["coefficients"]}
    alpha_deg = np.rad2deg(polar.alpha)
    return {
        "metadata": {
            "description": (
                f"AEROSTRUCTURAL ROM of {kite}: every parameter identified from "
                f"coupled VSM + Billow ({fidelity}) solutions at the centre of "
                "the wind window (gravity off), no flight data."
            ),
            "identified_by": "examples/3_rom_identification/identify_rom_aerostructural.py",
            "identified_on": date.today().isoformat(),
            "source": (
                f"run_identification_sweep.py: {n_trims} anchors (coupled trims) "
                f"plus {int((polar.source == 'polar').sum())} frozen-shape polar "
                "samples"
            ),
            "control_convention": (
                "u_p is the power-tape length l_dp [m]; u_s is the steering-tape "
                f"half-difference / {fit_spec['steering_tape_per_unit_us']} m."
            ),
            "drag_scope": (
                "C_D is the wing's, bridle-line drag included. The KCU drag is "
                "NOT in it (kcu_drag_in_coefficients: false): the ROM adds it "
                "from the system file's KCU length/diameter/turbine through "
                "awetrim.aerodynamics.kcu_drag, at the bridle point. The tether "
                "drag is the ROM tether model's."
            ),
            "fit_quality": {
                "CL_cv_rmse": float(fits["CL"].cv_rmse),
                "CD_cv_rmse": float(fits["CD"].cv_rmse),
                "theta_b_rms_deg": theta["rms_deg"],
                "roll_rms_deg": roll["rms_deg"],
            },
        },
        "controls": fit_spec["controls"],
        "validity": {
            "angle_of_attack_deg": [float(np.floor(alpha_deg.min())),
                                    float(np.ceil(alpha_deg.max()))],
            "input_depower": [round(float(polar.u_p.min()), 4), round(float(polar.u_p.max()), 4)],
            "input_steering_abs_max": round(float(polar.u_s.abs().max()), 4),
            # the fit block's own validity limits (e.g. min_tether_force_n,
            # the force margin that depends on the wind and so is no tape limit)
            **(fit_spec.get("validity") or {}),
        },
        "aerodynamics": aero,
        "tether": fit_spec["rom_tether"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--kite", default="LEI-V3-KITE", help="folder under data/")
    parser.add_argument("--fidelity", choices=ds.FIDELITIES, default=None,
                        help="structural model of the sweep (default: identification.yaml's)")
    parser.add_argument("--sweep", type=Path, default=None,
                        help="sweep folder (default results/<kite>/identification/<fidelity>/sweep)")
    parser.add_argument("--out", type=Path, default=None,
                        help="ROM file to write (default in the output folder)")
    parser.add_argument("--no-write", action="store_true",
                        help="fit and report, do not write the ROM file")
    args = parser.parse_args()

    fidelity, fit_spec = ds.identification_settings(
        PROJECT_DIR / "data" / args.kite / "identification.yaml", "fit", args.fidelity
    )
    band = tuple(fit_spec["depower_band"])
    results = PROJECT_DIR / "results" / args.kite / "identification" / fidelity
    out_dir = results / "rom_aerostructural"
    sweep = args.sweep or results / "sweep"

    trims_raw, polars_raw = ds.load_sweep(sweep)
    ok = trims_raw.converged & trims_raw.success_physical
    trims_raw = trims_raw[ok]
    polar, trims = ds.coefficient_tables(
        trims_raw, polars_raw[polars_raw.anchor.isin(
            trims_raw.anchor[trims_raw.polar_check <= POLAR_CHECK_TOLERANCE])],
        reference_area=fit_spec["reference_area"],
        steering_tape_per_unit_us=fit_spec["steering_tape_per_unit_us"],
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    polar.to_csv(out_dir / "polar_samples.csv", index=False)
    trims.to_csv(out_dir / "trim_samples.csv", index=False)
    lens = polar[polar.source == "polar"]
    print(f"{len(trims)} anchors ({int((~ok).sum())} unconverged dropped), "
          f"{len(lens)} polar samples ({int((~lens.converged).sum())} unconverged); "
          f"S_ref {fit_spec['reference_area']} m2")

    polar = polar[polar.converged & polar.u_p.between(*band)]
    theta = fit_theta_b(trims, band)
    # The roll law is the attached wing's: drop the states past the sweep's
    # own stall stop (the last state of a chain that stopped on stall).
    _, sweep_spec = ds.identification_settings(
        PROJECT_DIR / "data" / args.kite / "identification.yaml", "sweep", fidelity
    )
    stalled = trims.anchor.map(trims_raw.set_index("anchor").stalled_fraction)
    roll = fit_roll_gain(trims[stalled <= sweep_spec["max_stalled_fraction"]])
    stall = locate_stall(polar)
    fits = fit_coefficients(polar, stall)
    check_extrapolation(fits, polar, stall, band)

    print("\ntheta_b = {:.4f} + {:.4f} u_p [rad]  (rms {:.2f}, max {:.2f} deg, n {})".format(
        theta["angle_pitch_tether_0"], theta["slope_angle_pitch_tether_depower"],
        theta["rms_deg"], theta["max_deg"], theta["n"]))
    print("gain_roll_steering = {:.4f} {} rad per u_s  (degree {}, rms {:.2f} deg, n {}; "
          "CV rms by degree {})".format(
              roll["gain_roll_steering"],
              "".join(f"{c:+.4f} u_p^{p}" for p, c in
                      enumerate(roll.get("gain_roll_steering_depower", []), start=1)),
              roll["degree"], roll["rms_deg"], roll["n"],
              {d: round(v, 3) for d, v in roll["cv_rms_deg"].items()}))
    for target, fit in fits.items():
        print(f"{target}: CV RMSE {fit.cv_rmse:.5f}, in-sample R2 {fit.metrics['r2']:.5f}")
        print(f"    {fit.intercept:+.5f}" + "".join(
            f"\n    {coef:+.5f} {ap.term_label(pm)}" for pm, coef in fit.terms))

    doc = rom_document(fits, theta, roll, polar, fit_spec, stall, args.kite, len(trims),
                       fidelity)
    (out_dir / "identification_report.json").write_text(
        json.dumps({
            "theta_b": theta, "roll": roll,
            "stall_deg": {k: float(np.rad2deg(v)) for k, v in stall.items()},
            "selection": {
                t: {"criterion": f"grouped {CV_FOLDS}-fold CV RMSE, stop below "
                                 f"{MIN_RELATIVE_IMPROVEMENT:.0%} improvement",
                    "path": fit.selection_path, "cv_rmse": fit.cv_rmse,
                    "terms": [ap.term_label(pm) for pm, _ in fit.terms]}
                for t, fit in fits.items()
            },
            "metadata": doc["metadata"],
        }, indent=2), encoding="utf-8")
    if not args.no_write:
        out = args.out or out_dir / "rom_config_aerostructural.yaml"
        header = (
            f"# AEROSTRUCTURAL reduced-order model of {args.kite}.\n"
            "# GENERATED by examples/3_rom_identification/identify_rom_aerostructural.py;\n"
            "# re-run it rather than editing coefficients by hand.\n"
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(header + yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
        print(f"\nWrote {out}")


if __name__ == "__main__":
    main()

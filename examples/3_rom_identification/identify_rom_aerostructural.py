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
                  tape, and gain_roll_steering_squared, k + s(u_p) u_s^2 (degrees
                  of both by leave-one-chain-out CV).
    turn-rate law with --steering-law turn_rate (Cayon & Schmehl, in preparation,
                  Eq. 33): b(u_p), c5, c6 measured on the sweep's turn-law
                  states (sweep/turn_law/, awetrim.identification.turn_law),
                  then a(u_p, u_s) on the steering chains as for the roll gain.

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
from itertools import product
import json
from datetime import date
from pathlib import Path

import numpy as np
import yaml

from awetrim.identification import aero_polynomial as ap
from awetrim.identification import aerostructural_dataset as ds
from awetrim.identification import turn_law as tl
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
#: Width of the stall switch [deg]; the onset gives where it is, not how sharp.
STALL_WIDTH_DEG = 1.0
#: The coefficients are fitted up to this far past the stall [deg]. The ROM is
#: not flown past it, and the many deep-stall samples would otherwise cost the
#: accuracy of the operating range.
FIT_PAST_STALL_DEG = 8.0


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


def fit_roll_gain(trims, max_degree: int = 2, max_degree_squared: int = 1) -> dict:
    """phi_a,w = (k(u_p) + s(u_p) u_s^2) u_s through the origin on the steered
    trims, k(u_p) = k0 + k1 u_p + ... and s(u_p) = s0 + s1 u_p + ... polynomials
    whose degrees (0..max_degree; none or 0..max_degree_squared) are chosen by
    leave-one-chain-out cross-validation RMSE (a chain's states share one
    tape, so a chain never straddles train and test)."""
    rows = trims[trims.u_s != 0.0]
    if rows.empty:
        raise SystemExit("no steered anchors in the sweep: cannot fit the roll gain")
    fit = _fit_steering_polynomial(rows, rows.phi_a_w.to_numpy(float), max_degree,
                                   max_degree_squared)
    out = {"gain_roll_steering": fit["coef"][0],
           "rms_deg": float(np.rad2deg(fit["resid"].std())),
           "cv_rms_deg": {k: float(np.rad2deg(v)) for k, v in fit["cv"].items()},
           "degree": fit["degree"], "n": int(len(rows))}
    if fit["depower"]:
        out["gain_roll_steering_depower"] = fit["depower"]
    if fit["squared"] is not None:
        out["gain_roll_steering_squared"] = fit["squared"]
    return out


def measure_turn_law(sweep: Path, fit_spec: dict) -> dict:
    """b(u_p), c5, c6 of the turn-rate law from the sweep's turn-law states
    (run_identification_sweep.py --turn-law; awetrim.identification.turn_law),
    b(u_p) of degree ``fit.turn_law.inertia_degree`` (default 1)."""
    states = tl.load_turn_law_states(sweep)
    if states is None or states.empty:
        raise SystemExit(
            f"--steering-law turn_rate needs the sweep's turn-law states ({sweep / tl.TURN_LAW_DIR}); "
            "solve them with run_identification_sweep.py --turn-law")
    table = tl.turn_law_table(states, steering_tape_per_unit_us=fit_spec["steering_tape_per_unit_us"])
    return tl.measure(table, int((fit_spec.get("turn_law") or {}).get("inertia_degree", 1)))


def fit_turn_law(trims, terms: dict, max_degree: int = 2,
                 max_degree_squared: int = 1) -> dict:
    """The turn-rate law of Cayon & Schmehl (in preparation), Eq. 33 (ROM
    steering_law turn_rate), w (v_a + b m v_tau) = a(u_p, u_s) v_a^2 u_s at the
    centre of the window without gravity, on the steered trims.

    ``terms`` is :func:`measure_turn_law`'s: ``inertia_coefficient`` b(u_p)
    [b0, b1, ...] [1/kg], ``gravity_course_coefficient`` /
    ``gravity_radial_coefficient`` c5, c6 [1/kg] and ``density_reference``
    [kg/m3], the density they hold at. They are measured on the sweep's
    turn-law states, not fitted here: at the centre v_tau / v_a is fixed per
    tape, so the chains cannot separate b from a, and their gravity-off
    states carry no c5, c6.

    w is the turn rate about the tether: the trims' course rate, which at the
    centre of the window carries no transport of the course frame.
    a(u_p, u_s) = (a0 + a1 u_p + ...) + (s0 + s1 u_p ...) u_s^2 [rad/m per
    u_s], degrees by leave-one-chain-out CV as for the roll gain."""
    rows = trims[trims.u_s != 0.0]
    if rows.empty:
        raise SystemExit("no steered anchors in the sweep: cannot fit the turn law")
    w, v_a, v_tau, mass, u_p, rho = (rows[c].to_numpy(float) for c in (
        "chi_dot", "v_a", "v_tau", "mass", "u_p", "rho"))
    b_coefs = np.atleast_1d(np.asarray(terms["inertia_coefficient"], dtype=float))
    rho_ref = float(terms.get("density_reference") or np.median(rho))
    b = np.polynomial.polynomial.polyval(u_p, b_coefs) * rho_ref / rho
    denominator = v_a + b * mass * v_tau
    fit = _fit_steering_polynomial(rows, w * denominator / v_a**2, max_degree,
                                   max_degree_squared)
    resid_w = fit["resid"] * v_a**2 / denominator
    out = {"gain_turn_steering": fit["coef"][0],
           "turn_inertia_coefficient": float(b_coefs[0]),
           **({"turn_inertia_coefficient_depower": [float(c) for c in b_coefs[1:]]}
              if b_coefs.size > 1 else {}),
           "turn_gravity_course_coefficient": float(terms.get("gravity_course_coefficient", 0.0)),
           "turn_gravity_radial_coefficient": float(terms.get("gravity_radial_coefficient", 0.0)),
           "turn_density_reference": rho_ref,
           # the measurement behind b, c5, c6 (report only, not ROM parameters)
           "measured": {k: terms[k].to_dict(orient="records")
                        for k in ("inertia", "gravity") if k in terms},
           "rms_rad_s": float(resid_w.std()),
           "rms_relative": float(np.sqrt(np.mean((resid_w / w) ** 2))),
           "inertia_share_mean": float(np.mean(b * mass * v_tau / denominator)),
           "degree": fit["degree"], "n": int(len(rows))}
    if fit["depower"]:
        out["gain_turn_steering_depower"] = fit["depower"]
    if fit["squared"] is not None:
        out["gain_turn_steering_squared"] = fit["squared"]
    return out


def _fit_steering_polynomial(rows, target, max_degree, max_degree_squared) -> dict:
    """target = (k(u_p) + s(u_p) u_s^2) u_s through the origin, the degrees of
    k and s (none for s = -1) by leave-one-chain-out CV RMSE."""
    u_s, u_p = (rows[c].to_numpy(float) for c in ("u_s", "u_p"))
    phi = np.asarray(target, dtype=float)
    chains = rows.chain.to_numpy()
    # at most one fewer coefficient than distinct tapes
    max_degree = min(max_degree, len(np.unique(u_p.round(3))) - 1)

    def design(degrees, mask=slice(None)):
        deg, deg_squared = degrees
        return np.column_stack(
            [u_s[mask] * u_p[mask] ** p for p in range(deg + 1)]
            + [u_s[mask] ** 3 * u_p[mask] ** p for p in range(deg_squared + 1)]
        )

    cv = {}
    for degrees in product(range(max_degree + 1), range(-1, max_degree_squared + 1)):
        sq = []
        for c in np.unique(chains):
            test = chains == c
            coef, *_ = np.linalg.lstsq(design(degrees, ~test), phi[~test], rcond=None)
            sq.append((phi[test] - design(degrees, test) @ coef) ** 2)
        cv[degrees] = float(np.sqrt(np.concatenate(sq).mean()))
    degrees = min(cv, key=cv.get)
    coef, *_ = np.linalg.lstsq(design(degrees), phi, rcond=None)
    deg = degrees[0]
    return {
        "coef": [float(c) for c in coef],
        "resid": phi - design(degrees) @ coef,
        "cv": {f"k{d} s{s}": v for (d, s), v in cv.items()},
        "degree": list(degrees),
        "depower": [float(c) for c in coef[1:deg + 1]],
        "squared": [float(c) for c in coef[deg + 1:]] if degrees[1] >= 0 else None,
    }


def regressors(polar, stall) -> dict[str, np.ndarray]:
    """ROM regressors of the samples, with the stall switch at ``stall``."""
    data = {c: polar[c].to_numpy() for c in ("alpha", "u_p", "u_s", "v_a")}
    data["stall"] = stall_blend(data["alpha"], stall["angle_of_attack_stall"],
                                stall["width_stall"], xp=np)
    return data


def locate_stall(polar, trims, tapes) -> dict:
    """The stall angle: the median onset on the realistic branch of the
    unsteered states on ``tapes``, the tapes near their stall where the onset
    means something (:func:`ds.stall_onset_increment`). Measured, not chosen on
    the fit: the few samples at the lift drop itself would lose to the many past it."""
    states = trims[(trims.u_s == 0.0) & trims.u_p.between(*tapes)]
    onsets = []
    for state in states.itertuples():
        samples = polar[polar.anchor == state.anchor].sort_values("d_alpha_deg")
        increment = ds.stall_onset_increment(samples.d_alpha_deg, samples.CL)
        onsets.append(np.rad2deg(state.alpha_w) + increment)
    onsets = np.array(onsets)[np.isfinite(onsets)]
    if not onsets.size:
        raise ValueError(f"no stall onset in the polars on tapes {tapes} m")
    angle = float(np.median(onsets))
    print(f"stall: {angle:.2f} deg ({np.percentile(onsets, 10):.1f}-{np.percentile(onsets, 90):.1f}, "
          f"{onsets.size} states on {tapes[0]}-{tapes[1]} m), width {STALL_WIDTH_DEG:.1f} deg")
    return {"angle_of_attack_stall": np.deg2rad(angle), "width_stall": np.deg2rad(STALL_WIDTH_DEG)}


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
def rom_document(fits, theta, roll, polar, fit_spec, stall, kite, n_trims, fidelity,
                 turn=None) -> dict:
    aero = ap.build_rom_aerodynamics(
        fits.values(),
        extra_params={
            "angle_pitch_tether_0": theta["angle_pitch_tether_0"],
            "slope_angle_pitch_tether_depower": theta["slope_angle_pitch_tether_depower"],
            "gain_roll_steering": roll["gain_roll_steering"],
            **({"gain_roll_steering_depower": roll["gain_roll_steering_depower"]}
               if "gain_roll_steering_depower" in roll else {}),
            **({"gain_roll_steering_squared": roll["gain_roll_steering_squared"]}
               if "gain_roll_steering_squared" in roll else {}),
            **{k: float(v) for k, v in stall.items()},
            # the turn law (steering_law turn_rate); the roll gain above stays,
            # so setting steering_law: roll_gain flies the same ROM the old way
            **({k: v for k, v in turn.items() if k.startswith(("gain_turn", "turn_")) and k != "turn_law"}
               if turn else {}),
        },
    )
    aero = {"model": aero["model"], "reference_area": float(fit_spec["reference_area"]),
            "kcu_drag_in_coefficients": False,
            **({"steering_law": "turn_rate"} if turn else {}),
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
                **({"turn_rate_rms_rad_s": turn["rms_rad_s"],
                    "turn_rate_rms_relative": turn["rms_relative"]} if turn else {}),
            },
        },
        "controls": fit_spec["controls"],
        "validity": {
            # Up to the stall less the gust margin: what the optimiser may fly,
            # not how far the data reach.
            "angle_of_attack_deg": [float(np.floor(alpha_deg.min())),
                                    round(float(np.rad2deg(stall["angle_of_attack_stall"]))
                                          - float(fit_spec["stall_margin_deg"]), 2)],
            "input_depower": [round(float(polar.u_p.min()), 4), round(float(polar.u_p.max()), 4)],
            "input_steering_abs_max": round(float(polar.u_s.abs().max()), 4),
            # the fit block's own validity limits (e.g. min_tether_force_n,
            # the force margin that depends on the wind and so is no tape limit)
            **(fit_spec.get("validity") or {}),
        },
        "aerodynamics": aero,
        "tether": fit_spec["rom_tether"],
    }


def steering_only(args, fit_spec, trims_raw, polars_raw, out_dir) -> None:
    """Fit the steering law alone and write it into ``args.steering_only``'s
    ROM: the coefficients, bridle law and stall of that ROM stay as they are,
    so two such files differ in the steering law only."""
    _, trims = ds.coefficient_tables(
        trims_raw, polars_raw.iloc[:0], reference_area=fit_spec["reference_area"],
        steering_tape_per_unit_us=fit_spec["steering_tape_per_unit_us"])
    stalled = trims.anchor.map(trims_raw.set_index("anchor").stalled_fraction)
    trims = trims[stalled <= fit_spec["max_stalled_fraction"]].copy()
    doc = yaml.safe_load(args.steering_only.read_text(encoding="utf-8"))
    aero = doc["aerodynamics"]
    if args.steering_law == "roll_gain":
        roll = fit_roll_gain(trims)
        aero["params"].update({k: v for k, v in roll.items() if k.startswith("gain_roll")})
        aero.pop("steering_law", None)
        summary = {"roll_gain": roll}
    else:
        raw = trims_raw.set_index("anchor")
        trims["mass"], trims["rho"] = trims.anchor.map(raw.mass_total), trims.anchor.map(raw.rho)
        turn = fit_turn_law(trims, measure_turn_law(args.sweep, fit_spec))
        aero["params"].update(
            {k: v for k, v in turn.items() if k.startswith(("gain_turn", "turn_")) and k != "turn_law"})
        aero["steering_law"] = "turn_rate"
        doc.setdefault("metadata", {}).setdefault("fit_quality", {}).update(
            turn_rate_rms_rad_s=turn["rms_rad_s"], turn_rate_rms_relative=turn["rms_relative"])
        summary = {"turn_law": turn}
    doc.setdefault("metadata", {})["steering_law_source"] = (
        f"{args.steering_law} re-identified on {date.today().isoformat()} from "
        f"{len(trims)} trims of {args.sweep or 'the default sweep'}; the rest is "
        f"{args.steering_only.name}'s")
    print(json.dumps(summary, indent=2))
    if not args.no_write:
        out = args.out or out_dir / f"{args.steering_only.stem}_{args.steering_law}.yaml"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            f"# {args.steering_only.name} with its steering law re-identified "
            f"({args.steering_law}).\n# GENERATED by "
            "examples/3_rom_identification/identify_rom_aerostructural.py --steering-only.\n"
            + yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
        print(f"\nWrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--kite", default="LEI-V3-KITE", help="folder under data/")
    parser.add_argument("--fidelity", choices=ds.FIDELITIES, default=None,
                        help="structural model of the sweep (default: identification.yaml's)")
    parser.add_argument("--sweep", type=Path, default=None,
                        help="sweep folder (default results/<kite>/identification/<fidelity>/sweep)")
    parser.add_argument("--results", type=Path, default=None,
                        help="results folder (default results/<kite>/identification/<fidelity>)")
    parser.add_argument("--out", type=Path, default=None,
                        help="ROM file to write (default in the output folder)")
    parser.add_argument("--steering-law", choices=("roll_gain", "turn_rate"), default="roll_gain",
                        help="the ROM's steering law; turn_rate also needs the sweep's "
                        "turn-law states (run_identification_sweep.py --turn-law)")
    parser.add_argument("--steering-only", type=Path, default=None, metavar="BASE_ROM",
                        help="re-identify only the steering law (--steering-law) on the "
                        "sweep's trims and write BASE_ROM with it; everything else of "
                        "BASE_ROM is kept")
    parser.add_argument("--no-write", action="store_true",
                        help="fit and report, do not write the ROM file")
    args = parser.parse_args()

    fidelity, fit_spec = ds.identification_settings(
        PROJECT_DIR / "data" / args.kite / "identification.yaml", "fit", args.fidelity
    )
    band = tuple(fit_spec["depower_band"])
    results = args.results or PROJECT_DIR / "results" / args.kite / "identification" / fidelity
    out_dir = results / "rom_aerostructural"
    sweep = args.sweep = args.sweep or results / "sweep"

    trims_raw, polars_raw = ds.load_sweep(sweep)
    ok = trims_raw.converged & trims_raw.success_physical
    trims_raw = trims_raw[ok]
    if args.steering_only is not None:
        steering_only(args, fit_spec, trims_raw, polars_raw, out_dir)
        return
    polar, trims = ds.coefficient_tables(
        trims_raw, polars_raw[polars_raw.anchor.isin(
            trims_raw.anchor[trims_raw.polar_check <= ds.POLAR_CHECK_TOLERANCE])],
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

    stalled = trims.anchor.map(trims_raw.set_index("anchor").stalled_fraction)
    attached_anchors = trims.anchor[stalled <= fit_spec["max_stalled_fraction"]]
    polar = polar[polar.converged & polar.u_p.between(*band) & polar.anchor.isin(attached_anchors)]
    if not (polar.source == "polar").any():
        raise SystemExit("no polar sample survives the polar check, the fit band and the stall cut: "
                         "a fit on the trims alone cannot separate alpha from the depower")
    theta = fit_theta_b(trims, band)
    # The roll law is the attached wing's too.
    roll = fit_roll_gain(trims[trims.anchor.isin(attached_anchors)])
    turn = None
    if args.steering_law == "turn_rate":
        raw = trims_raw.set_index("anchor")
        trims["mass"], trims["rho"] = trims.anchor.map(raw.mass_total), trims.anchor.map(raw.rho)
        turn = fit_turn_law(trims[trims.anchor.isin(attached_anchors)],
                            measure_turn_law(sweep, fit_spec))
    stall = locate_stall(polar, trims, fit_spec["stall_onset_tapes"])
    polar = polar[polar.alpha <= stall["angle_of_attack_stall"] + np.deg2rad(FIT_PAST_STALL_DEG)]
    fits = fit_coefficients(polar, stall)
    check_extrapolation(fits, polar, stall, band)

    print("\ntheta_b = {:.4f} + {:.4f} u_p [rad]  (rms {:.2f}, max {:.2f} deg, n {})".format(
        theta["angle_pitch_tether_0"], theta["slope_angle_pitch_tether_depower"],
        theta["rms_deg"], theta["max_deg"], theta["n"]))
    print("gain_roll_steering_squared =", roll.get("gain_roll_steering_squared"))
    print("gain_roll_steering = {:.4f} {} rad per u_s  (degree {}, rms {:.2f} deg, n {}; "
          "CV rms by degree {})".format(
              roll["gain_roll_steering"],
              "".join(f"{c:+.4f} u_p^{p}" for p, c in
                      enumerate(roll.get("gain_roll_steering_depower", []), start=1)),
              roll["degree"], roll["rms_deg"], roll["n"],
              {d: round(v, 3) for d, v in roll["cv_rms_deg"].items()}))
    if turn:
        print("turn law: b(u_p) {} + {}, c5 {:.5f}, c6 {:.5f} 1/kg at rho {:.4f}".format(
            turn["turn_inertia_coefficient"], turn.get("turn_inertia_coefficient_depower", []),
            turn["turn_gravity_course_coefficient"], turn["turn_gravity_radial_coefficient"],
            turn["turn_density_reference"]))
        print("turn law: a = {:.4f} {} rad/m per u_s, squared {}, b = {:.5f} 1/kg  (degree {}, "
              "rms {:.4f} rad/s = {:.1%}, inertia share {:.2f}, n {})".format(
                  turn["gain_turn_steering"],
                  "".join(f"{c:+.4f} u_p^{p}" for p, c in
                          enumerate(turn.get("gain_turn_steering_depower", []), start=1)),
                  turn.get("gain_turn_steering_squared"), turn["turn_inertia_coefficient"],
                  turn["degree"], turn["rms_rad_s"], turn["rms_relative"],
                  turn["inertia_share_mean"], turn["n"]))
    for target, fit in fits.items():
        print(f"{target}: CV RMSE {fit.cv_rmse:.5f}, in-sample R2 {fit.metrics['r2']:.5f}")
        print(f"    {fit.intercept:+.5f}" + "".join(
            f"\n    {coef:+.5f} {ap.term_label(pm)}" for pm, coef in fit.terms))

    doc = rom_document(fits, theta, roll, polar, fit_spec, stall, args.kite, len(trims),
                       fidelity, turn)
    (out_dir / "identification_report.json").write_text(
        json.dumps({
            "theta_b": theta, "roll": roll, **({"turn_law": turn} if turn else {}),
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

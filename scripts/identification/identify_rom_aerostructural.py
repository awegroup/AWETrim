"""Identify the LEI-V3 reduced-order model from aerostructural simulations only.

The simulation-only counterpart of the semi-empirical ROM
(``data/LEI-V3-KITE/rom_config_semi_empirical.yaml``, Cayon, van Deursen,
Schmehl 2026, WES 11, 1097, Sect. 5.1): every parameter of the ROM of that
paper is taken from coupled VSM + Billow-wireframe solutions at the CENTRE OF
THE WIND WINDOW (azimuth 0, elevation 0, gravity off) -- no flight data.

Inputs (the _billow9 campaign, scripts/personal/wes-quasi-steady/results)
    center_sweep_2019_udp_continuation_billow9_va*        u_p chains, u_s = 0
    center_sweep_2019_us_continuation_reel{out,in}_billow9_va*   u_s chains
  per point: point.json (the coupled trim), alpha_polar.json (the frozen-
  shape alpha sweep of run_center_sweep_alpha_polar.py, -6..+8 deg about the
  trim) and, when present, alpha_polar_stall.json (+9..+22 deg: the stall, so
  the polynomial turns over where the kite does and the ROM's solvers find no
  spurious slow high-alpha equilibrium).

What is identified, against the ROM's own definitions (awetrim.system.kite)
    theta_b(u_p)  paper Eqs. 1-2, alpha_w = alpha_b - theta_b, from the trims:
                  alpha_b is the paper Eq. D11 angle of the bridle resultant
                  F_b the ROM builds (tether + KCU weight, inertia AND drag:
                  the KCU hangs at the bridle point), alpha_w the centre-panel
                  angle. Linear in u_p over the operating band (the ROM's
                  form).
    C_L, C_D      polynomials in (alpha, u_p, u_s, v_a) on the frozen-shape
                  polars: the trims alone tie alpha to u_p almost one to one,
                  the polars separate them. C_D is the WING's: the bridle
                  drag is in, the KCU drag is NOT (kcu_drag_in_coefficients:
                  false -- the ROM adds it from the system file's KCU
                  hardware, so a KCU/turbine swap needs no
                  re-identification), nor the tether (the ROM models it). Terms chosen by
                  forward selection (then backward elimination) on
                  CROSS-VALIDATION RMSE with folds by
                  ANCHOR (one frozen shape never straddles train and test).
    gain_roll_steering  paper Eq. 43, phi_a,w = k u_s: the roll of the lift
                  about v_a relative to the bridle resultant F_b, with F_b
                  built as the ROM builds it (tether + the KCU's share of the
                  inertial force).

Conventions: u_p = power-tape length l_dp [m]; u_s = steering-tape
half-difference / 1.4 m (awetrim.identification.controls, the standardised
flight scale); alpha in rad; coefficients on the LEI-V3 ROM reference area
S_REF (19.75 m2, the EKF/flight area the semi-empirical ROM also uses),
stated in the file as reference_area.

Usage (project root)
    python scripts/identification/identify_rom_aerostructural.py
    python scripts/identification/identify_rom_aerostructural.py --no-write
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from awetrim.identification import aero_polynomial as ap
from awetrim.system.kite import stall_blend
from awetrim.utils.config_paths import (
    LEI_V3_ROM_AEROSTRUCTURAL_CONFIG,
    LEI_V3_ROM_SEMI_EMPIRICAL_CONFIG,
    REPO_ROOT,
)

DEFAULT_RESULTS = REPO_ROOT / "scripts" / "personal" / "wes-quasi-steady" / "results"
OUT_DIR = REPO_ROOT / "results" / "LEI-V3-KITE" / "identification" / "rom_aerostructural"
CHAIN_GLOBS = (
    "center_sweep_2019_udp_continuation_billow9_va*",
    "center_sweep_2019_us_continuation_reelout_billow9_va*",
    "center_sweep_2019_us_continuation_reelin_billow9_va*",
)
#: Reference area of the LEI-V3 ROMs [m2]: the EKF/flight area. Both V3 ROM
#: files state it (aerodynamics.reference_area), so their coefficients are
#: directly comparable and independent of system.yaml's projected area.
S_REF = 19.75
#: Steering-tape half-difference per unit u_s [m] (controls.py standard).
STEERING_GAIN_M = 1.4
#: Operating depower band the ROM is fitted on [m] (flight 1.7 / 2.1 plus
#: margin for the optimizer's excursions).
UP_BAND = (1.6, 2.2)
#: A polar is kept only if, at the coupled trim's alpha, its C_L is within
#: this fraction of the coupled trim's: the frozen-shape re-trim must land on
#: the coupled state. Steered snapshots beyond u_s tape ~0.1 m re-trim to a
#: different roll/rate state (alpha up to 2 deg off, C_L 10-23 % high).
LENS_CL_TOLERANCE = 0.01
#: Forward selection stops when the best term improves CV RMSE by < this.
MIN_RELATIVE_IMPROVEMENT = 0.01
CV_FOLDS = 5

# Candidate library. u_s enters lift only through even powers (left/right
# symmetry); drag also gets |u_s| (the ROM evaluates C_D terms as |term|).
# The ROM's solvers do not bound alpha, so the model must behave outside the
# operating range. No alpha^3: a cubic C_D went negative beyond the data (a
# spurious reel-in root at alpha -31 deg, 178 m/s in the flight validation)
# and a cubic C_L cannot follow the coupled model's stall. The STALL is the
# ``stall`` variable instead (awetrim.system.kite.stall_blend, a smooth step
# at angle_of_attack_stall): terms multiplied by it act only past the stall,
# C = C_attached + stall * (C_separated - C_attached). Without a stall the
# fitted C_L kept rising to 35 deg and the downloop seed settled on a slow
# alpha-35 deg branch.
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
# Drag uses the same library. Every term is a plain, smooth monomial (the ROM
# used to take |term| of every C_D term, which bent the fitted 0.23*alpha into
# a V at alpha = 0 that stalled the reel-out seed march), and steering enters
# through even powers of u_s, symmetric like lift.
CANDIDATES_CL = CANDIDATES_CL + STALL_TERMS
CANDIDATES_CD = list(CANDIDATES_CL)
#: Stall-location grid [deg]: the coupled polars peak at 15.6-16.7 deg in C_L
#: and drop ~0.35 within ~2 deg to a flat ~0.87 plateau.
STALL_ANGLE_GRID_DEG = (15.5, 16.0, 16.5, 17.0, 17.5, 18.0)
STALL_WIDTH_GRID_DEG = (0.5, 1.0, 1.5)
#: Optional per-point cache extending the polars into stall.
STALL_CACHE = "alpha_polar_stall.json"
#: Fit polar samples up to this angle of attack [deg] (the whole extension:
#: the stall terms model the post-stall plateau).
ALPHA_MAX_FIT_DEG = 30.0


# ───────────────────────────────────────────────────────────────── dataset
def _to_course_frame(v) -> np.ndarray:
    """Trim (VSM) frame -> paper C-frame: the kite flies along -e_course there,
    so the course and normal axes flip (right-handedness is kept)."""
    v = np.asarray(v, dtype=float).reshape(3)
    return np.array([-v[0], -v[1], v[2]])


def _roll_about_va(force_c: np.ndarray, va_c: np.ndarray) -> float:
    """Roll of a force about v_a, paper Eqs. D6-D8 (phi = 0: along e_r')."""
    va_chi, va_n, va_r = va_c
    va_tau = np.hypot(va_chi, va_n)
    va = np.linalg.norm(va_c)
    e_n = np.array([-va_n, va_chi, 0.0]) / va_tau
    e_r = np.array([-va_chi * va_r, -va_n * va_r, va_tau**2]) / (va * va_tau)
    return float(np.arctan2(-force_c @ e_n, force_c @ e_r))


def _lift_drag_coefficients(force, va, z_body, rho, area) -> tuple[float, float]:
    """(C_L, C_D) of a force about the apparent wind; lift signed by body up."""
    e_va = va / np.linalg.norm(va)
    drag = float(force @ e_va)
    f_perp = force - drag * e_va
    up = -(z_body - (z_body @ e_va) * e_va)
    lift = np.sign(f_perp @ up) * np.linalg.norm(f_perp)
    q = 0.5 * rho * float(va @ va)
    return lift / (q * area), drag / (q * area)


def _bridle_angle_of_attack(force_bridle_c, va_c) -> float:
    """Paper Eq. D11: alpha_b = atan2(F_b . e_chi', -F_b . e_r'), e_chi' = -va/|va|."""
    va_chi, va_n, va_r = va_c
    va_tau = np.hypot(va_chi, va_n)
    va = np.linalg.norm(va_c)
    e_chi = -np.asarray(va_c) / va
    e_r = np.array([-va_chi * va_r, -va_n * va_r, va_tau**2]) / (va * va_tau)
    return float(np.arctan2(force_bridle_c @ e_chi, -force_bridle_c @ e_r))


def _mass_split(results_root: Path) -> tuple[float, float]:
    """(KCU mass, total mass) [kg] the trims were solved with."""
    for path in results_root.glob(CHAIN_GLOBS[0] + "/udp_*/vw_*/static_stability.json"):
        record = json.loads(path.read_text(encoding="utf-8"))[0]
        return float(record["state"]["kcu_mass"]), float(record["mass"])
    raise FileNotFoundError("no static_stability.json to read the trim masses from")


def build_dataset(results_root: Path, area: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(polar rows, trim rows) from every anchor with an alpha_polar.json."""
    m_kcu, m_total = _mass_split(results_root)
    polar_rows, trim_rows = [], []
    for pattern in CHAIN_GLOBS:
        for cache in sorted(results_root.glob(pattern + "/udp_*/vw_*/alpha_polar.json")):
            chain = cache.parents[2].name
            rec = json.loads(cache.read_text(encoding="utf-8"))
            point = json.loads((cache.parent / "point.json").read_text(encoding="utf-8"))
            if point.get("status") != "ok" or not rec["success_physical"]:
                continue
            anchor = f"{chain}/{cache.parents[1].name}"
            u_p, u_s = float(rec["ldp"]), float(rec["us_tape"]) / STEERING_GAIN_M
            z_body = np.asarray(rec["frd_axes_world"], dtype=float)[2]
            # Wing (+ bridle) force only: the ROM adds the KCU drag itself.
            trim = point["trim"]
            anchor_rows = []
            steps = list(rec["polar"])
            stall = cache.parent / STALL_CACHE
            if stall.exists():
                # Same snapshot and re-trim; its 0-step repeats the trim.
                steps += [st for st in json.loads(stall.read_text(encoding="utf-8"))["polar"]
                          if st["d_alpha_deg"] != 0.0]
            for step in steps:
                va = np.asarray(step["va_world"], dtype=float)
                force = np.asarray(step["F_aero_world"])
                cl, cd = _lift_drag_coefficients(force, va, z_body, rec["rho"], area)
                anchor_rows.append({
                    "anchor": anchor, "chain": chain, "source": "polar",
                    "u_p": u_p, "u_s": u_s, "v_a": float(np.linalg.norm(va)),
                    "alpha": np.deg2rad(step["alpha_deg"]),
                    "d_alpha_deg": step["d_alpha_deg"], "CL": cl, "CD": cd,
                    "converged": bool(step["gamma_converged"]),
                })
            # The coupled trim itself: an exact sample of the target state.
            va_t = np.asarray(trim["va_vel_world"], dtype=float)
            f_t = np.asarray(trim["total_aero_force_vec"])
            cl_t, cd_t = _lift_drag_coefficients(f_t, va_t, z_body, rec["rho"], area)
            alpha_t = np.deg2rad(trim["aoa_deg"])
            order = np.argsort([r["alpha"] for r in anchor_rows])
            cl_lens = np.interp(alpha_t, np.array([anchor_rows[i]["alpha"] for i in order]),
                                np.array([anchor_rows[i]["CL"] for i in order]))
            lens_ok = abs(cl_lens / cl_t - 1.0) <= LENS_CL_TOLERANCE
            for r in anchor_rows:
                r["lens_consistent"] = lens_ok
            polar_rows += anchor_rows
            polar_rows.append({
                "anchor": anchor, "chain": chain, "source": "coupled_trim",
                "u_p": u_p, "u_s": u_s, "v_a": float(np.linalg.norm(va_t)),
                "alpha": alpha_t, "d_alpha_deg": np.nan, "CL": cl_t, "CD": cd_t,
                "converged": True, "lens_consistent": True,
            })

            va_c = _to_course_frame(trim["va_vel_world"])
            f_wing_c = _to_course_frame(trim["total_aero_force_vec"])
            f_kcu_c = _to_course_frame(trim["kcu_drag_force_vsm"])
            f_inertial_c = _to_course_frame(trim["inertial_force"])
            # Equilibrium F_wing + F_kcu_drag + F_t + F_inertial = 0 (gravity
            # off). The ROM's bridle resultant is the tether plus the KCU's
            # drag and inertial share = minus the wing's force and inertia.
            f_bridle_c = -(f_wing_c + (1.0 - m_kcu / m_total) * f_inertial_c)
            e_va_c = va_c / np.linalg.norm(va_c)
            lift_c = f_wing_c - (f_wing_c @ e_va_c) * e_va_c
            opt_x = np.asarray(trim["opt_x"], dtype=float)
            trim_rows.append({
                "anchor": anchor, "chain": chain, "u_p": u_p, "u_s": u_s,
                "v_a": float(np.linalg.norm(va_c)),
                "alpha_w": np.deg2rad(trim["aoa_deg"]),
                "alpha_b": _bridle_angle_of_attack(f_bridle_c, va_c),
                # Tether-referenced alpha_b: the convention of a ROM whose C_D
                # carries the KCU drag (the semi-empirical one).
                "alpha_b_tether": np.deg2rad(trim["aoa_course_deg"]),
                "phi_a": _roll_about_va(lift_c, va_c),
                "phi_a_b": _roll_about_va(-f_bridle_c, va_c),
                "wind_speed": float(rec["wind_speed"]),
                "tension": float(trim["tether_force"]),
                "v_tau": float(opt_x[0]), "chi_dot": float(opt_x[4]),
                # Normal-force balance at elevation 0: F_n = -m v_tau chi_dot.
                "chi_dot_from_force": float(
                    -(f_wing_c[1] + f_kcu_c[1]) / (m_total * opt_x[0])
                ),
            })
    polar = pd.DataFrame(polar_rows)
    trims = pd.DataFrame(trim_rows)
    trims["theta_b"] = trims["alpha_b"] - trims["alpha_w"]
    trims["phi_a_w"] = trims["phi_a"] - trims["phi_a_b"]
    return polar, trims


# ───────────────────────────────────────────────────────────────── fits
def fit_theta_b(trims: pd.DataFrame) -> dict:
    """Linear theta_b(u_p) on the unsteered trims inside the band."""
    rows = trims[(trims.u_s == 0.0) & trims.u_p.between(*UP_BAND)]
    slope, intercept = np.polyfit(rows.u_p, rows.theta_b, 1)
    resid = rows.theta_b - (intercept + slope * rows.u_p)
    return {
        "angle_pitch_tether_0": float(intercept),
        "slope_angle_pitch_tether_depower": float(slope),
        "rms_deg": float(np.rad2deg(resid.std())),
        "max_deg": float(np.rad2deg(resid.abs().max())),
        "n": int(len(rows)),
    }


def fit_roll_gain(trims: pd.DataFrame) -> dict:
    """phi_a,w = k u_s through the origin, on the steering chains."""
    rows = trims[trims.chain.str.contains("us_continuation")]
    k = float((rows.u_s @ rows.phi_a_w) / (rows.u_s @ rows.u_s))
    resid = rows.phi_a_w - k * rows.u_s
    return {"gain_roll_steering": k, "rms_deg": float(np.rad2deg(resid.std())),
            "n": int(len(rows))}


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


def check_extrapolation(fits, polar, stall) -> None:
    """C_D must stay positive over +-30 deg alpha across the fitted u_p, u_s
    and v_a ranges: a ROM solver that wanders outside the data must meet more
    drag, never thrust."""
    alpha = np.deg2rad(np.linspace(-30.0, 30.0, 121))
    worst = np.inf
    for up in np.linspace(*UP_BAND, 7):
        for us in (0.0, float(polar.u_s.abs().max())):
            for va in (float(polar.v_a.min()), float(polar.v_a.max())):
                data = _grid(alpha, up, us, va, stall)
                worst = min(worst, float(fits["CD"].predict(data).min()))
    print(f"extrapolation check: min C_D over +-30 deg = {worst:.4f}")
    positive = alpha >= 0.0
    for up in UP_BAND:
        cl = fits["CL"].predict(_grid(alpha, up, 0.0, 19.0, stall))
        i = int(np.argmax(np.where(positive, cl, -np.inf)))
        print(f"  C_L max at u_p {up} m: {cl[i]:.3f} at alpha {np.rad2deg(alpha[i]):.1f} deg; "
              f"C_L(30 deg) {cl[-1]:.3f}, C_L(-30 deg) {cl[0]:.3f}")


def _grid(alpha, up, us, va, stall):
    data = {"alpha": alpha, "u_p": np.full_like(alpha, up),
            "u_s": np.full_like(alpha, us), "v_a": np.full_like(alpha, va)}
    data["stall"] = stall_blend(alpha, stall["angle_of_attack_stall"],
                                stall["width_stall"], xp=np)
    return data
    if worst <= 0.0:
        raise SystemExit("C_D polynomial goes non-positive outside the data; "
                         "restrict the candidate library")


# ───────────────────────────────────────────────────────────────── output
def rom_document(fits, theta, roll, polar, area, stall) -> dict:
    semi = yaml.safe_load(LEI_V3_ROM_SEMI_EMPIRICAL_CONFIG.read_text(encoding="utf-8"))
    aero = ap.build_rom_aerodynamics(
        fits.values(),
        extra_params={
            "angle_pitch_tether_0": theta["angle_pitch_tether_0"],
            "slope_angle_pitch_tether_depower": theta["slope_angle_pitch_tether_depower"],
            "gain_roll_steering": roll["gain_roll_steering"],
            **{k: float(v) for k, v in stall.items()},
        },
    )
    aero = {"model": aero["model"], "reference_area": float(area),
            "kcu_drag_in_coefficients": False,
            "params": aero["params"], "coefficients": aero["coefficients"]}
    alpha_deg = np.rad2deg(polar.alpha)
    return {
        "metadata": {
            "description": (
                "AEROSTRUCTURAL ROM of the LEI-V3 kite: every parameter "
                "identified from coupled VSM + Billow-wireframe solutions at "
                "the centre of the wind window (gravity off), no flight data. "
                "Counterpart of rom_config_semi_empirical.yaml."
            ),
            "identified_by": "scripts/identification/identify_rom_aerostructural.py",
            "identified_on": date.today().isoformat(),
            "source": (
                "_billow9 centre-window continuation chains (2019 hardware): "
                f"{polar.anchor.nunique()} anchors: their coupled trims plus "
                f"{int((polar.source == 'polar').sum())} frozen-shape polar samples "
                "where the re-trim reproduces the coupled state; VSM main @ "
                "8825a12, AV on, 45 panels"
            ),
            "control_convention": (
                "u_p is the power-tape length l_dp [m]; u_s is the steering-"
                "tape half-difference / 1.4 m (awetrim.identification.controls, "
                "the standardised flight scale)."
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
        "controls": semi["controls"],
        "validity": {
            "angle_of_attack_deg": [float(np.floor(alpha_deg.min())),
                                    float(np.ceil(alpha_deg.max()))],
            "input_depower": [float(polar.u_p.min()), float(polar.u_p.max())],
            "input_steering_abs_max": float(polar.u_s.abs().max()),
        },
        "aerodynamics": aero,
        "tether": semi["tether"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--no-write", action="store_true",
                        help="fit and report, do not write the ROM config")
    args = parser.parse_args()

    area = S_REF

    polar, trims = build_dataset(args.results_root, area)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    polar.to_csv(OUT_DIR / "polar_samples.csv", index=False)
    trims.to_csv(OUT_DIR / "trim_samples.csv", index=False)
    lens = polar[polar.source == "polar"]
    bad = lens[~lens.lens_consistent].anchor.unique()
    print(f"{polar.anchor.nunique()} anchors, {len(lens)} polar samples "
          f"({(~lens.converged).sum()} unconverged); {len(bad)} anchors' polars "
          f"dropped (re-trim off the coupled state, C_L > {LENS_CL_TOLERANCE:.0%}), "
          f"their coupled trims kept; S_ref {area} m2")

    polar = polar[polar.converged & polar.lens_consistent & polar.u_p.between(*UP_BAND)
                  & (polar.alpha <= np.deg2rad(ALPHA_MAX_FIT_DEG))]
    theta = fit_theta_b(trims)
    roll = fit_roll_gain(trims)
    stall = locate_stall(polar)
    fits = fit_coefficients(polar, stall)
    check_extrapolation(fits, polar, stall)

    print("\ntheta_b = {:.4f} + {:.4f} u_p [rad]  (rms {:.2f}, max {:.2f} deg, n {})".format(
        theta["angle_pitch_tether_0"], theta["slope_angle_pitch_tether_depower"],
        theta["rms_deg"], theta["max_deg"], theta["n"]))
    print("gain_roll_steering = {:.4f} rad per u_s  (rms {:.2f} deg, n {})".format(
        roll["gain_roll_steering"], roll["rms_deg"], roll["n"]))
    for target, fit in fits.items():
        print(f"{target}: CV RMSE {fit.cv_rmse:.5f}, in-sample R2 {fit.metrics['r2']:.5f}")
        print(f"    {fit.intercept:+.5f}" + "".join(
            f"\n    {coef:+.5f} {ap.term_label(pm)}" for pm, coef in fit.terms))
    steer = trims[trims.chain.str.contains("us_continuation")]
    print("chi_dot convention check (AS trim vs normal-force balance), max diff "
          f"{(steer.chi_dot - steer.chi_dot_from_force).abs().max():.4f} rad/s")

    doc = rom_document(fits, theta, roll, polar, area, stall)
    (OUT_DIR / "identification_report.json").write_text(
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
        header = (
            "# AEROSTRUCTURAL reduced-order model of the LEI-V3 kite.\n"
            "# GENERATED by scripts/identification/identify_rom_aerostructural.py;\n"
            "# re-run it rather than editing coefficients by hand.\n"
        )
        LEI_V3_ROM_AEROSTRUCTURAL_CONFIG.write_text(
            header + yaml.safe_dump(doc, sort_keys=False), encoding="utf-8"
        )
        print(f"\nWrote {LEI_V3_ROM_AEROSTRUCTURAL_CONFIG}")


if __name__ == "__main__":
    main()

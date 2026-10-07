"""Correct the aerostructural ROM's depower pitch and drag with 2019 flight data.

The aerostructural ROM (``rom_config_aerostructural.yaml``, from coupled VSM +
Billow-wireframe trims only) keeps its lift polynomial and stall.
Flight data identify, in two separate least-squares steps:

1. theta_b(u_p) = theta_0 + theta_1 u_p, the bridle pitch that sets which alpha
   the kite flies at a given power-tape length (angle_pitch_tether_0,
   slope_angle_pitch_tether_depower). It absorbs the as-built vs flown bridle
   geometry and the logged-depower -> tape-length calibration. Identified FROM
   LIFT, with no alpha sensor: at each flown sample the alpha* at which the AS
   C_L(alpha, u_p, u_s, v_a) equals the flight C_L is found, and
   alpha_b - alpha* is the theta_b the flight needs; the correction
   dtheta = dtheta_0 [+ dtheta_1 u_p] to the AS relation is fitted to it, the
   slope kept only if cross-validation asks for it. Drag plays no part, so
   theta_b and the drag correction cannot trade off.
2. dC_D(u_p, u_s) AT EQUAL LIFT, i.e. C_D,AS evaluated at the same alpha*:
   the drag polar C_D(C_L) is corrected, not C_D at the force angle. (alpha_b
   from Eq. D11 is essentially the flight glide angle atan(C_D/C_L) plus
   wing-mass terms, so evaluating C_D at alpha_b - theta_b would let the flight
   drag pick the angle it is compared at -- circular; per sample that angle is
   glide-angle noise of several degrees.) Single-input terms only:

       C_D = C_D,AS(alpha, u_p, u_s, v_a) + dC_D0 + c_p u_p + c_s u_s^2

   (no products, no other powers; u_s enters squared because the drag cannot
   depend on the turn direction, and |u_s| would put a kink at 0).
3. gain_roll_steering k, paper Eq. 43, phi_a,w = k u_s, a STARTING VALUE and
   a diagnostic only: step 3 sets the gain from the turn-rate law (the solved
   steering regressing 1:1 on the logged steering), because this measured
   roll is the wing-inertia identity F_w + F_b = m_w (a - g) -- it scales with
   the wing mass assumed -- and includes the roll needed against the ROM's
   gravity-driven turn, 2.4x the flight's, so it reads ~20 % high (-0.77 vs
   -0.62 from the turn-rate law). phi_a,w is the roll of
   the wing force about v_a relative to the bridle resultant (D6-D8, the
   definition used on the coupled trims), the wing force being minus the
   bridle resultant plus the wing's inertia and weight. Fitted through the
   origin against u_s delayed by STEERING_LAG_S (the EKF's steering-input lag);
   the intercept of a free fit is reported as a check of the steering zero
   offset.

Term selection in both steps is the forward + backward selection of the
aerostructural fit on grouped cross-validation RMSE (folds by CYCLE);
uncertainties from a cycle bootstrap that repeats BOTH steps. The lift residual
left after step 1 is a lift-model error: reported, not corrected. The result is
written as ``rom_config_aerostructural_flight_corrected.yaml``.

Flight samples, from the 2019-10-08 EKF reconstruction (results/LEI-V3-KITE/ekf)
    C_L target  the EKF wing lift coefficient (the KCU and bridle drag act along
                v_a and carry no lift).
    C_D target  the EKF's total aerodynamic drag of wing + bridle + KCU, minus
                the KCU drag of the ROM's own model (kcu_drag.py on the system
                file's KCU, axis e_r as in Kite.force_drag_kcu_for): the ROM's
                C_D convention (kcu_drag_in_coefficients: false). The EKF's own
                split into wing / bridle / KCU drag is therefore irrelevant.
    alpha_b     paper Eq. D11 on the bridle resultant F_b the ROM builds
                (tether + KCU weight, inertia, drag). The EKF's bridle element
                tension is that resultant with the EKF's KCU drag, bridle drag
                and KCU mass; those are swapped for the ROM's (the bridle drag
                is in the ROM's C_D, not in F_b). A force angle, not a vane.
    u_p, u_s    awetrim.identification.controls (power-tape metres;
                standardised steering + FLIGHT_STEERING_ZERO_OFFSET_2019).
    selection   reel-out (v_r > 0.5, u_p < 1.9, elevation < 0.75 rad) and
                reel-in (v_r < -0.5, u_p >= 1.9), alpha_w inside the ROM's
                validity range and below its stall.

``--alpha rom`` / ``--alpha bridle`` skip step 1 (theta_b stays the AS one;
alpha from the ROM reconstruction or the EKF's wing_angle_of_attack_bridle) and
are kept as comparisons; each variant writes to its own results folder.

Cycles 60-67 are HELD OUT (the flight-validation cycles of
validate_quasi_steady_state_v3.py); the fits use every other cycle.

Usage (project root)
    python scripts/identification/identify_rom_flight_correction.py
    python scripts/identification/identify_rom_flight_correction.py --no-write
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
from datetime import date
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.spatial.transform import Rotation

from awetrim.aerodynamics.kcu_drag import force_drag_kcu
from awetrim.identification import aero_polynomial as ap
from awetrim.identification.controls import (
    FLIGHT_STEERING_ZERO_OFFSET_2019,
    flight_dataframe_depower_to_power_tape_length,
    flight_dataframe_steering_to_us,
)
from awetrim.plotting.plotting import PALETTE, set_plot_style
from awetrim.system.factory import create_system_model_from_yaml
from awetrim.system.kite import stall_blend
from awetrim.utils.config_paths import (
    LEI_V3_ROM_AEROSTRUCTURAL_CONFIG,
    LEI_V3_ROM_SEMI_EMPIRICAL_CONFIG,
    LEI_V3_SYSTEM_FLOWN_CONFIG,
)

EKF_FILE = Path("results/LEI-V3-KITE/ekf/LEI-V3 Kite_2019-10-08.h5")
OUT_DIR = Path("results/LEI-V3-KITE/identification/rom_flight_correction")
OUT_CONFIG = LEI_V3_ROM_AEROSTRUCTURAL_CONFIG.with_name(
    "rom_config_aerostructural_flight_corrected.yaml"
)
HOLDOUT_CYCLES = range(60, 68)
#: Step 1 library: dtheta = dtheta_0 [+ dtheta_1 u_p].
CANDIDATES_THETA = [{"u_p": 1}]
#: Step 2 library: single-input terms only (the intercept is always in).
CANDIDATES_DRAG = [{"u_p": 1}, {"u_s": 2}]
CV_FOLDS = 5
MIN_RELATIVE_IMPROVEMENT = 0.01
#: Every n-th 10 Hz sample (consecutive samples are not independent anyway;
#: the folds and the bootstrap are by cycle).
STRIDE = 5
BOOTSTRAP = 500
#: Stay this far below the ROM's stall angle [deg].
STALL_CLEARANCE_DEG = 1.0
#: Lower end of the C_L inversion bracket [deg].
ALPHA_INVERSION_MIN_DEG = -10.0
MIN_APPARENT_SPEED = 8.0
#: Steering-to-roll delay [s]: the EKF's steering_input_lag (2019 config).
STEERING_LAG_S = 0.3
G = np.array([0.0, 0.0, -9.81])
#: EKF ENU -> NED row permutation (awes_ekf.utils.rotate_ENU2NED).
ENU2NED = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])


# ───────────────────────────────────────────────────────────────── data
def load_ekf(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    with h5py.File(path, "r") as hf:
        ekf = pd.DataFrame({k: np.asarray(hf["ekf_output"][k][()])
                            for k in hf["ekf_output"]
                            if hf["ekf_output"][k].dtype.kind == "f"})
        flight = pd.DataFrame({k: np.asarray(hf["flight_data"][k][()])
                               for k in hf["flight_data"]
                               if hf["flight_data"][k].dtype.kind in "fiub"})
        config = {g: dict(hf["config_data"][g].attrs) for g in hf["config_data"]}
    return ekf, flight, config


def bridle_axis(ekf: pd.DataFrame) -> np.ndarray:
    """EKF bridle-frame z axis in ENU (= the bridle-element tension direction).

    awes_ekf writes the Euler angles of rotate_ENU2NED(dcm_b2w) as scipy
    extrinsic "xyz"; inverted here.
    """
    euler = ekf[["kite_roll", "kite_pitch", "kite_yaw"]].to_numpy()
    dcm_ned = Rotation.from_euler("xyz", euler).as_matrix()
    return np.einsum("ij,njk->nik", ENU2NED.T, dcm_ned)[:, :, 2]


def d11_angle_of_attack(force_bridle, va, radial) -> np.ndarray:
    """Paper Eq. D11, row-wise: alpha_b = atan2(F_b . e_chi', -F_b . e_r')."""
    e_va = va / np.linalg.norm(va, axis=1, keepdims=True)
    e_r = radial - np.sum(radial * e_va, axis=1, keepdims=True) * e_va
    e_r /= np.linalg.norm(e_r, axis=1, keepdims=True)
    return np.arctan2(np.sum(force_bridle * -e_va, axis=1),
                      -np.sum(force_bridle * e_r, axis=1))


def roll_about_va(force, va, radial) -> np.ndarray:
    """Paper Eqs. D6-D8, row-wise: roll of a force about v_a (0 along e_r')."""
    e_va = va / np.linalg.norm(va, axis=1, keepdims=True)
    e_n = np.cross(radial, va)
    e_n /= np.linalg.norm(e_n, axis=1, keepdims=True)
    e_r = radial - np.sum(radial * e_va, axis=1, keepdims=True) * e_va
    e_r /= np.linalg.norm(e_r, axis=1, keepdims=True)
    return np.arctan2(-np.sum(force * e_n, axis=1), np.sum(force * e_r, axis=1))


def build_dataset(rom: dict) -> pd.DataFrame:
    with contextlib.redirect_stdout(io.StringIO()):
        factory = create_system_model_from_yaml(
            LEI_V3_SYSTEM_FLOWN_CONFIG, aero_yaml_path=LEI_V3_ROM_AEROSTRUCTURAL_CONFIG,
            steering_control="asymmetric")
    kite = factory.kite
    kcu = kite.kcu_drag_model
    m_kcu_rom = float(kite.mass_kcu)
    ekf, flight, config = load_ekf(EKF_FILE)
    area = float(config["kite"]["area"])
    m_kcu_ekf = float(config["kcu"]["mass"])
    rho = float(kite.rho)
    if abs(area - rom["aerodynamics"]["reference_area"]) > 1e-9:
        raise SystemExit(f"EKF area {area} != ROM reference area")

    r = ekf[["kite_position_x", "kite_position_y", "kite_position_z"]].to_numpy()
    v = ekf[["kite_velocity_x", "kite_velocity_y", "kite_velocity_z"]].to_numpy()
    wind = np.c_[ekf.wind_speed_horizontal * np.cos(ekf.wind_direction),
                 ekf.wind_speed_horizontal * np.sin(ekf.wind_direction),
                 ekf.wind_speed_vertical]
    va = wind - v
    speed = np.linalg.norm(va, axis=1)
    e_va = va / speed[:, None]
    radial = r / np.linalg.norm(r, axis=1, keepdims=True)
    q_area = 0.5 * rho * speed**2 * area
    accel = np.gradient(v, ekf.time.to_numpy(), axis=0)

    # The EKF's bridle-element tension IS F_b with the EKF's KCU/bridle models.
    f_bridle_ekf = ekf.tether_force_kite.to_numpy()[:, None] * bridle_axis(ekf)
    alpha_b_ekf = d11_angle_of_attack(f_bridle_ekf, va, radial)
    f_kcu_rom = np.array([force_drag_kcu(va[i], radial[i], rho, kcu.cd_area_axial,
                                         kcu.cd_area_broadside) for i in range(len(va))])
    cd_kcu_rom = np.sum(f_kcu_rom * e_va, axis=1) / q_area
    cd_removed = ekf.kcu_drag_coefficient + ekf.bridles_drag_coefficient
    f_bridle_rom = (f_bridle_ekf
                    - (cd_removed.to_numpy() * q_area)[:, None] * e_va + f_kcu_rom
                    + (m_kcu_rom - m_kcu_ekf) * (G - accel))

    # Wing aerodynamic force = minus the bridle resultant plus the wing's
    # inertia and weight (F_w + F_b + m_w g - m_w a = 0), for the roll.
    f_wing = -f_bridle_rom + float(kite.mass_wing) * (accel - G)
    phi_a_w = roll_about_va(f_wing, va, radial) - roll_about_va(-f_bridle_rom, va, radial)

    flight["input_depower"] = flight_dataframe_depower_to_power_tape_length(flight)
    data = pd.DataFrame({
        "time": ekf.time, "cycle": flight.cycle.astype(int),
        "speed_radial": flight.tether_reelout_speed, "elevation": flight.kite_elevation,
        "u_p": flight.input_depower.to_numpy(),
        "u_s": np.asarray(flight_dataframe_steering_to_us(flight)
                          + FLIGHT_STEERING_ZERO_OFFSET_2019),
        "v_a": speed,
        "phi_a_w": phi_a_w,
        "alpha_b": d11_angle_of_attack(f_bridle_rom, va, radial),
        "alpha_b_ekf": alpha_b_ekf,
        "alpha_b_ekf_reported": np.deg2rad(ekf.kite_angle_of_attack),
        "alpha_bridle": np.deg2rad(ekf.wing_angle_of_attack_bridle),
        "CL": ekf.wing_lift_coefficient,
        "CD": (ekf.wing_drag_coefficient + cd_removed) - cd_kcu_rom,
        "CD_kcu_rom": cd_kcu_rom, "CD_kcu_ekf": ekf.kcu_drag_coefficient,
    })
    data["u_s_lagged"] = data.u_s.shift(int(round(STEERING_LAG_S / np.median(np.diff(ekf.time)))))
    reel_out = (data.speed_radial > 0.5) & (data.u_p < 1.9) & (data.elevation < 0.75)
    reel_in = (data.speed_radial < -0.5) & (data.u_p >= 1.9)
    data["phase"] = np.where(reel_out, "reel-out", np.where(reel_in, "reel-in", ""))
    return data


def select_phases(data: pd.DataFrame) -> pd.DataFrame:
    keep = ((data.phase != "") & (data.cycle > 0) & (data.v_a > MIN_APPARENT_SPEED)
            & np.isfinite(data[["CL", "CD", "alpha_b"]]).all(axis=1))
    return data[keep].iloc[::STRIDE].reset_index(drop=True)


def theta_as(params: dict, u_p) -> np.ndarray:
    return params["angle_pitch_tether_0"] + params["slope_angle_pitch_tether_depower"] * u_p


def regressors(df: pd.DataFrame, params: dict, alpha=None) -> dict[str, np.ndarray]:
    alpha = df.alpha.to_numpy(dtype=float) if alpha is None else np.asarray(alpha, dtype=float)
    return {"alpha": alpha, "u_p": df.u_p.to_numpy(dtype=float),
            "u_s": df.u_s.to_numpy(dtype=float), "v_a": df.v_a.to_numpy(dtype=float),
            "stall": stall_blend(alpha, params["angle_of_attack_stall"],
                                 params["width_stall"], xp=np)}


# ───────────────────────────────────────────────────────────── step 1: theta_b
def alpha_matching_lift(df: pd.DataFrame, fit_cl: ap.PolynomialFit, params: dict,
                        iterations: int = 50) -> np.ndarray:
    """alpha* with C_L,AS(alpha*, u_p, u_s, v_a) = C_L,flight (vectorised bisection).

    Bracket [ALPHA_INVERSION_MIN_DEG, stall - STALL_CLEARANCE_DEG], the
    pre-stall branch; NaN where the flight C_L is outside the bracket's lift.
    """
    lo = np.full(len(df), np.deg2rad(ALPHA_INVERSION_MIN_DEG))
    hi = np.full(len(df), params["angle_of_attack_stall"] - np.deg2rad(STALL_CLEARANCE_DEG))
    target = df.CL.to_numpy(dtype=float)
    f = lambda a: fit_cl.predict(regressors(df, params, a)) - target  # noqa: E731
    f_lo, f_hi = f(lo), f(hi)
    ok = np.sign(f_lo) * np.sign(f_hi) < 0
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        f_mid = f(mid)
        left = np.sign(f_mid) == np.sign(f_lo)
        lo, f_lo = np.where(left, mid, lo), np.where(left, f_mid, f_lo)
        hi = np.where(left, hi, mid)
    return np.where(ok, 0.5 * (lo + hi), np.nan)


def fit_theta(train: pd.DataFrame, terms=None) -> ap.PolynomialFit:
    """dtheta = theta_needed - theta_AS on [1, u_p]: selected, or fixed ``terms``."""
    g = train[np.isfinite(train.dtheta_needed)]
    data = {"u_p": g.u_p.to_numpy(dtype=float)}
    if terms is not None:
        return ap.fit_terms(data, g.dtheta_needed.to_numpy(), terms, target="CL",
                            regressors=("u_p",))
    return ap.select_model(
        data, g.dtheta_needed.to_numpy(), target="CL", candidate_terms=CANDIDATES_THETA,
        regressors=("u_p",), cv_folds=CV_FOLDS, cv_groups=g.cycle.to_numpy(),
        criterion="cv", min_relative_improvement=MIN_RELATIVE_IMPROVEMENT, backward=True)


def fit_roll_gain(train: pd.DataFrame) -> dict:
    """phi_a,w = k u_s(t - lag) through the origin; free-intercept fit as a check."""
    g = train[np.isfinite(train.u_s_lagged) & np.isfinite(train.phi_a_w)]
    x, y = g.u_s_lagged.to_numpy(), g.phi_a_w.to_numpy()
    k = float(x @ y / (x @ x))
    (k_free, offset), *_ = np.linalg.lstsq(np.c_[x, np.ones_like(x)], y, rcond=None)
    resid = y - k * x
    x0 = g.u_s.to_numpy()
    return {"gain_roll_steering": k, "lag_s": STEERING_LAG_S,
            "r2": float(1 - resid.var() / y.var()),
            "rms_deg": float(np.rad2deg(resid.std())),
            "gain_without_lag": float(x0 @ y / (x0 @ x0)),
            "gain_free_intercept": float(k_free), "offset_deg": float(np.rad2deg(offset)),
            "per_phase": {ph: float(h.u_s_lagged @ h.phi_a_w / (h.u_s_lagged @ h.u_s_lagged))
                          for ph, h in g.groupby("phase")},
            "n": int(len(g))}


def theta_params(params: dict, fit: ap.PolynomialFit | None) -> dict:
    """The ROM's theta_b parameters after the correction (AS ones if no fit)."""
    theta0 = params["angle_pitch_tether_0"]
    theta1 = params["slope_angle_pitch_tether_depower"]
    if fit is not None:
        theta0 += fit.intercept
        theta1 += sum(c for pm, c in fit.terms if pm == {"u_p": 1})
    return {"angle_pitch_tether_0": float(theta0),
            "slope_angle_pitch_tether_depower": float(theta1)}


# ───────────────────────────────────────────────────────────── step 2: drag
def evaluate_as(df: pd.DataFrame, fits_as: dict, params: dict) -> pd.DataFrame:
    reg = regressors(df, params)
    df["stall"] = reg["stall"]
    df["CD_AS"] = fits_as["CD"].predict(reg)
    df["CL_AS"] = fits_as["CL"].predict(reg)
    df["dCD"] = df.CD - df.CD_AS
    df["dCL"] = df.CL - df.CL_AS
    return df


def in_validity(df: pd.DataFrame, rom: dict) -> pd.Series:
    lo, hi = np.deg2rad(rom["validity"]["angle_of_attack_deg"])
    stall = rom["aerodynamics"]["params"]["angle_of_attack_stall"]
    return df.alpha.between(lo, min(hi, stall - np.deg2rad(STALL_CLEARANCE_DEG)))


def fit_drag(train: pd.DataFrame, terms=None) -> ap.PolynomialFit:
    data = {"u_p": train.u_p.to_numpy(dtype=float), "u_s": train.u_s.to_numpy(dtype=float)}
    if terms is not None:
        return ap.fit_terms(data, train.dCD.to_numpy(), terms, target="CD",
                            regressors=("u_p", "u_s"))
    return ap.select_model(
        data, train.dCD.to_numpy(), target="CD", candidate_terms=CANDIDATES_DRAG,
        regressors=("u_p", "u_s"), cv_folds=CV_FOLDS, cv_groups=train.cycle.to_numpy(),
        criterion="cv", min_relative_improvement=MIN_RELATIVE_IMPROVEMENT, backward=True)


def predict_controls(fit: ap.PolynomialFit, df: pd.DataFrame) -> np.ndarray:
    return fit.predict({k: df[k].to_numpy(dtype=float) for k in ("u_p", "u_s")})


# ───────────────────────────────────────────────────────────── uncertainty
def bootstrap(pool: pd.DataFrame, theta_fit, drag_fit, params, seed: int = 0) -> dict:
    """Cycle bootstrap repeating BOTH steps with the selected terms.

    Each draw resamples whole training cycles and refits dtheta (if step 1
    ran) and the drag terms on the same draw. At equal lift the drag residual
    does not depend on theta_b, so the two steps share only the resampling.
    """
    rng = np.random.default_rng(seed)
    cycles = pool.cycle.unique()
    by_cycle = {c: pool[pool.cycle == c] for c in cycles}
    theta_terms = None if theta_fit is None else [pm for pm, _ in theta_fit.terms]
    drag_terms = [pm for pm, _ in drag_fit.terms]
    draws = []
    for _ in range(BOOTSTRAP):
        sample = pd.concat([by_cycle[c] for c in rng.choice(cycles, cycles.size)])
        row = []
        if theta_fit is not None:
            theta = theta_params(params, fit_theta(sample, theta_terms))
            row += [theta["angle_pitch_tether_0"], theta["slope_angle_pitch_tether_depower"]]
        d = fit_drag(sample, drag_terms)
        row += [fit_roll_gain(sample)["gain_roll_steering"]]
        draws.append(row + [d.intercept] + [c for _, c in d.terms])
    draws = np.asarray(draws)
    labels = ((["angle_pitch_tether_0", "slope_angle_pitch_tether_depower"]
               if theta_fit is not None else [])
              + ["gain_roll_steering", "dCD: 1"] + [f"dCD: {ap.term_label(pm)}" for pm in drag_terms])
    return {lab: {"std": float(draws[:, i].std(ddof=1)),
                  "ci95": [float(np.percentile(draws[:, i], 2.5)),
                           float(np.percentile(draws[:, i], 97.5))]}
            for i, lab in enumerate(labels)}


# ───────────────────────────────────────────────────────────── ROM file
def merged_rom(rom: dict, theta: dict, drag_fit: ap.PolynomialFit, roll: dict,
               report: dict) -> dict:
    """The AS ROM with the new theta_b, roll gain and the drag correction in C_D."""
    doc = json.loads(json.dumps(rom))
    aero = doc["aerodynamics"]
    aero["params"].update(theta)
    aero["params"]["gain_roll_steering"] = float(roll["gain_roll_steering"])
    aero["params"]["CD0"] = float(aero["params"]["CD0"] + drag_fit.intercept)
    entries = aero["coefficients"]["CD"]
    for power_map, coef in drag_fit.terms:
        for entry in entries:
            existing = (dict(entry["vars"]) if "vars" in entry
                        else {entry["var"]: int(entry.get("power", 1))})
            if ap.term_key(existing) == ap.term_key(power_map) and not entry.get("abs"):
                entry["coef"] = float(entry["coef"] + coef)
                break
        else:
            entries.append(ap._term_to_rom_entry(dict(power_map), float(coef)))
    doc["metadata"] = {
        **rom["metadata"],
        "description": (
            "AEROSTRUCTURAL ROM of the LEI-V3 kite with FLIGHT corrections: "
            "rom_config_aerostructural.yaml (coupled VSM + Billow-wireframe, centre "
            "of the wind window) with theta_b(u_p) identified from the flight lift "
            "and dC_D(u_p, u_s) fitted by least squares on the 2019-10-08 EKF "
            "reconstruction (cycles 60-67 held out), with the roll gain identified "
            "together with the steering drag. Lift polynomial and "
            "stall are the aerostructural ones."),
        "identified_by": "scripts/identification/identify_rom_flight_correction.py",
        "identified_on": date.today().isoformat(),
        "base_rom": LEI_V3_ROM_AEROSTRUCTURAL_CONFIG.name,
        "flight_correction": report,
    }
    return doc


# ───────────────────────────────────────────────────────────── report
def phase_table(df: pd.DataFrame, column: str) -> dict:
    return {ph: {"median": float(g[column].median()), "mean": float(g[column].mean()),
                 "rms": float(np.sqrt(np.mean(g[column] ** 2))), "n": int(len(g))}
            for ph, g in df.groupby("phase")}


def plot(train, test, theta, drag_fit, params, path: Path,
         x_label: str = r"$\alpha^*$ ($^\circ$, equal lift)") -> None:
    set_plot_style()
    fig, axes = plt.subplots(3, 2, figsize=(11.0, 11.0))
    both = pd.concat([train.assign(split="fit"), test.assign(split="holdout")])
    phases = (("reel-out", PALETTE["Blue"]), ("reel-in", PALETTE["Orange"]))

    ax = axes[0, 0]
    if "theta_needed" in both:
        for ph, colour in phases:
            g = both[both.phase == ph]
            ax.scatter(g.u_p, np.rad2deg(g.theta_needed), s=3, alpha=0.15, color=colour,
                       label=f"flight lift needs, {ph}", rasterized=True)
    semi = yaml.safe_load(LEI_V3_ROM_SEMI_EMPIRICAL_CONFIG.read_text(encoding="utf-8"))
    u = np.linspace(1.6, 2.2, 20)
    for p, colour, label in ((params, PALETTE["Black"], "aerostructural"),
                             (theta, PALETTE["Bluish Green"], "flight-corrected"),
                             (semi["aerodynamics"]["params"], PALETTE["Reddish Purple"],
                              "semi-empirical")):
        ax.plot(u, np.rad2deg(theta_as(p, u)), color=colour, lw=1.8, label=label)
    ax.set_xlabel(r"$u_p$ (m)")
    ax.set_ylabel(r"$\theta_b$ ($^\circ$)")
    ax.legend(frameon=False, fontsize=7, markerscale=4)

    # Lift is matched by construction with step 1 (alpha* from C_L), so this
    # panel shows what theta_b(u_p) leaves: its residual against steering.
    ax = axes[0, 1]
    for ph, colour in phases:
        g = both[both.phase == ph]
        if "theta_needed" in both:
            y = np.rad2deg(g.theta_needed - theta_as(theta, g.u_p.to_numpy()))
            ylabel = r"$\theta_b$ needed $-$ fitted ($^\circ$)"
        else:
            y, ylabel = g.dCL, r"$C_{L,\mathrm{flight}} - C_{L,\mathrm{AS}}$ (not corrected)"
        ax.scatter(g.u_s, y, s=3, alpha=0.15, color=colour, label=ph, rasterized=True)
    ax.axhline(0.0, color=PALETTE["Black"], lw=0.8)
    ax.set_xlabel(r"$u_s$")
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False, fontsize=8, markerscale=4)

    for ax, name in ((axes[1, 0], "u_p"), (axes[1, 1], "u_s")):
        for df, colour, label in ((train, PALETTE["Blue"], "fit cycles"),
                                  (test, PALETTE["Vermillion"], "held out (60-67)")):
            ax.scatter(df[name], df.dCD, s=3, alpha=0.2, color=colour, label=label,
                       rasterized=True)
        ax.axhline(0.0, color=PALETTE["Black"], lw=0.8)
        ax.set_xlabel({"u_p": r"$u_p$ (m)", "u_s": r"$u_s$"}[name])
        ax.set_ylabel(r"$C_{D,\mathrm{flight}} - C_{D,\mathrm{AS}}$")
    grid = pd.DataFrame({"u_p": np.linspace(1.6, 2.2, 50), "u_s": 0.0})
    axes[1, 0].plot(grid.u_p, predict_controls(drag_fit, grid), color=PALETTE["Black"],
                    lw=1.8, label=r"$\Delta C_D$ at $u_s=0$")
    grid = pd.DataFrame({"u_p": 1.7, "u_s": np.linspace(-0.2, 0.2, 50)})
    axes[1, 1].plot(grid.u_s, predict_controls(drag_fit, grid), color=PALETTE["Black"],
                    lw=1.8, label=r"$\Delta C_D$ at $u_p=1.7$")
    for ax in axes[1]:
        ax.legend(frameon=False, fontsize=8, markerscale=4)

    # Drag against alpha (as compared) and the drag polar C_D(C_L).
    for ax, x, xlabel in ((axes[2, 0], np.rad2deg(both.alpha), x_label),
                          (axes[2, 1], both.CL, r"$C_L$ ($S_\mathrm{ref}$ 19.75 m$^2$)")):
        for ph, colour in phases:
            m = (both.phase == ph).to_numpy()
            ax.scatter(x[m], both.CD[m], s=3, alpha=0.15, color=colour,
                       label=f"flight, {ph}", rasterized=True)
        x_as = x if ax is axes[2, 0] else both.CL_AS
        ax.scatter(x_as, both.CD_AS, s=3, alpha=0.15, color=PALETTE["Black"],
                   rasterized=True, label="aerostructural ROM")
        ax.scatter(x_as, both.CD_corrected, s=3, alpha=0.15, color=PALETTE["Bluish Green"],
                   rasterized=True, label="flight-corrected ROM")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"$C_D$ ($S_\mathrm{ref}$ 19.75 m$^2$)")
        ax.legend(frameon=False, fontsize=7, markerscale=4)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(path.with_suffix("." + ext), dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--alpha", choices=("lift", "rom", "bridle"), default="lift",
                        help="lift: identify theta_b from the flight lift (default); "
                             "rom / bridle: keep the AS theta_b (comparisons)")
    parser.add_argument("--no-write", action="store_true",
                        help="fit and report, do not write the ROM config")
    args = parser.parse_args()
    out_dir = OUT_DIR / f"alpha_{args.alpha}"

    rom = yaml.safe_load(LEI_V3_ROM_AEROSTRUCTURAL_CONFIG.read_text(encoding="utf-8"))
    params = rom["aerodynamics"]["params"]
    fits_as = ap.fits_from_rom_aerodynamics(rom["aerodynamics"])
    full = build_dataset(rom)
    check = full[full.phase != ""]
    print("D11 on the EKF bridle force vs the EKF's reported alpha: median "
          f"{np.rad2deg((check.alpha_b_ekf - check.alpha_b_ekf_reported).median()):+.3f} deg; "
          f"ROM F_b moves alpha_b by {np.rad2deg((check.alpha_b - check.alpha_b_ekf).median()):+.2f} deg; "
          f"KCU C_D ROM {check.CD_kcu_rom.median():.4f} vs EKF {check.CD_kcu_ekf.median():.4f}")

    pool = select_phases(full)
    holdout = pool.cycle.isin(HOLDOUT_CYCLES)
    theta_fit, theta = None, theta_params(params, None)
    report: dict = {"alpha_source": args.alpha,
                    "holdout_cycles": [HOLDOUT_CYCLES.start, HOLDOUT_CYCLES.stop - 1]}

    # Step 1: theta_b from the flight lift.
    if args.alpha == "lift":
        alpha_star = alpha_matching_lift(pool, fits_as["CL"], params)
        pool["theta_needed"] = pool.alpha_b - alpha_star
        pool["dtheta_needed"] = pool.theta_needed - theta_as(params, pool.u_p.to_numpy())
        print(f"C_L inversion: {np.isfinite(alpha_star).mean():.1%} of samples invertible")
        theta_fit = fit_theta(pool[~holdout])
        theta = theta_params(params, theta_fit)
        print("theta_b: AS {:+.4f} {:+.4f} u_p -> flight {:+.4f} {:+.4f} u_p [rad]".format(
            params["angle_pitch_tether_0"], params["slope_angle_pitch_tether_depower"],
            theta["angle_pitch_tether_0"], theta["slope_angle_pitch_tether_depower"]))
        print("  selection path:", theta_fit.selection_path,
              f"CV RMSE {np.rad2deg(theta_fit.cv_rmse):.2f} deg")
        report["theta_b"] = {
            "aerostructural": theta_params(params, None), "flight": theta,
            "selection_path": theta_fit.selection_path,
            "cv_rmse_deg": float(np.rad2deg(theta_fit.cv_rmse)),
            "invertible_fraction": float(np.isfinite(alpha_star).mean()),
        }
        # Step 2 compares drag at equal lift: at alpha*, not at alpha_b - theta_b.
        pool["alpha"] = alpha_star
        pool = evaluate_as(pool, fits_as, params)
    else:
        pool["alpha"] = (pool.alpha_b - theta_as(params, pool.u_p.to_numpy())
                         if args.alpha == "rom" else pool.alpha_bridle)
        pool = evaluate_as(pool, fits_as, params)

    # Step 2: drag at that alpha.
    keep = in_validity(pool, rom)
    print(f"{keep.sum()} of {len(pool)} samples inside the ROM's alpha validity "
          f"(below stall - {STALL_CLEARANCE_DEG:.0f} deg)")
    pool = pool[keep].copy()
    holdout = pool.cycle.isin(HOLDOUT_CYCLES)
    train, test = pool[~holdout].copy(), pool[holdout].copy()
    print(f"fit on {train.cycle.nunique()} cycles ({len(train)} samples), "
          f"held out {test.cycle.nunique()} ({len(test)})")
    drag_fit = fit_drag(train)
    for df in (train, test):
        df["CD_corrected"] = df.CD_AS + predict_controls(drag_fit, df)
        df["dCD_after"] = df.CD - df.CD_corrected
    print("\ndC_D = " + f"{drag_fit.intercept:+.5f}" + "".join(
        f" {c:+.5f} {ap.term_label(pm)}" for pm, c in drag_fit.terms))
    print("  selection path:", drag_fit.selection_path, f"CV RMSE {drag_fit.cv_rmse:.5f}")
    roll = fit_roll_gain(train)
    print("roll gain: k {gain_roll_steering:+.4f} (AS {as_gain:+.4f}), R2 {r2:.3f}, lag {lag_s} s; "
          "no lag {gain_without_lag:+.4f}; free intercept {offset_deg:+.2f} deg; per phase "
          "{per_phase}".format(as_gain=params["gain_roll_steering"], **roll))

    ci = bootstrap(train, theta_fit, drag_fit, params)
    print(f"cycle bootstrap ({BOOTSTRAP} draws, both steps):")
    for lab, s in ci.items():
        print(f"  {lab:>34s}: std {s['std']:.5f}, 95% [{s['ci95'][0]:+.5f}, {s['ci95'][1]:+.5f}]")

    report.update({
        "drag_law": "C_D = C_D,AS + dC_D, dC_D single-input terms in u_p, u_s^2",
        "dCD_intercept": float(drag_fit.intercept),
        "dCD_terms": {ap.term_label(pm): float(c) for pm, c in drag_fit.terms},
        "dCD_selection_path": drag_fit.selection_path,
        "dCD_cv_rmse": float(drag_fit.cv_rmse),
        "roll_gain": roll,
        "bootstrap": ci,
        "dCD_before": {"fit": phase_table(train, "dCD"), "holdout": phase_table(test, "dCD")},
        "dCD_after": {"fit": phase_table(train, "dCD_after"),
                      "holdout": phase_table(test, "dCD_after")},
        "dCL_not_corrected": {"fit": phase_table(train, "dCL"),
                              "holdout": phase_table(test, "dCL")},
    })
    for key in ("dCD_before", "dCD_after", "dCL_not_corrected"):
        for split in ("fit", "holdout"):
            for ph, s in report[key][split].items():
                print(f"{key:18s} {split:8s} {ph:9s} median {s['median']:+.4f} "
                      f"rms {s['rms']:.4f} (n {s['n']})")
    print("alpha_w flown (deg) percentiles 5/50/95 per phase:")
    for ph, g in pool.groupby("phase"):
        print(f"  {ph}: {np.percentile(np.rad2deg(g.alpha), [5, 50, 95]).round(1)}")

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.concat([train.assign(split="fit"), test.assign(split="holdout")]).to_csv(
        out_dir / "flight_samples.csv", index=False)
    (out_dir / "correction_report.json").write_text(json.dumps(report, indent=2),
                                                     encoding="utf-8")
    plot(train, test, theta, drag_fit, params, out_dir / "flight_correction",
         **({} if args.alpha == "lift" else {"x_label": rf"$\alpha_w$ ($^\circ$, {args.alpha})"}))
    print(f"Wrote {out_dir}")
    if not args.no_write:
        if args.alpha != "lift":
            raise SystemExit("only --alpha lift writes the ROM config")
        header = (
            "# AEROSTRUCTURAL + FLIGHT-CORRECTED reduced-order model of the LEI-V3 kite.\n"
            "# GENERATED by scripts/identification/identify_rom_flight_correction.py;\n"
            "# re-run it rather than editing coefficients by hand.\n"
        )
        OUT_CONFIG.write_text(header + yaml.safe_dump(
            merged_rom(rom, theta, drag_fit, roll, report), sort_keys=False), encoding="utf-8")
        print(f"Wrote {OUT_CONFIG}")


if __name__ == "__main__":
    main()

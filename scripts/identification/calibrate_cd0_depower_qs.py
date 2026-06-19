# Copyright (c) 2023-2026 Oriol Cayon, Delft University of Technology
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# SPDX-License-Identifier: Apache-2.0

"""Calibrate the ROM aero parameters CD0, angle_pitch_depower_0 and
delta_pitch_depower against the quasi-steady (QS) validation.

The QS trim takes the measured flight kinematics + wind as given and solves the
force balance for [tension_tether_ground, input_steering, speed_tangential]. We
then fit the three aero parameters so that the *predicted* tether force and
tangential speed match the *measured* ones, per phase (powered / depowered).

Why force + speed (rather than the EKF CL/CD)?
  - speed v_tau ~ glide ratio L/D = CL/CD      -> sensitive to the *ratio*
  - tether force ~ CL * v_a^2                   -> sensitive to the lift *level*
  apd (-> alpha -> CL) moves lift; CD0 moves drag only (CL untouched). The two
  observables therefore have different signatures and jointly break the CD0<->apd
  degeneracy. The powered/depowered phase split pins delta_pitch_depower.

This mirrors the solve setup of
``scripts/reduced-order-model/validation/validate_quasi_steady_state_v3.py`` so the
fitted values match what that validator reports.

Run from the project root:
    python scripts/identification/calibrate_cd0_depower_qs.py
"""

from __future__ import annotations

import h5py
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares

from awetrim import SystemModel
from awetrim.system.kite import Kite
from awetrim.system.tether import RigidLumpedTether
from awetrim.system.factory import load_aero_input_from_system_config
from awetrim.environment.Wind import Wind
from awetrim.identification.controls import (
    ROM_POWERED_INPUT_DEPOWER,
    ROM_DEPOWERED_INPUT_DEPOWER,
    flight_dataframe_depower_to_power_tape_length,
    flight_dataframe_steering_to_us,
)
from awetrim.utils.config_paths import LEI_V3_SYSTEM_FLOWN_CONFIG

import yaml

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
YEAR, MONTH, DAY = "2019", "10", "08"
KITE_MODEL = "LEI-V3-Kite"
EKF_RESULTS_DIR = "./results/LEI-V3-KITE/ekf/"

# Cycles used for the fit. Widen for robustness (e.g. range(20, 110)); a small
# range keeps the run fast while you iterate on weights/targets.
CYCLE_RANGE = range(60, 64)
POINT_STRIDE = 4  # raise (3, 4, ...) to subsample points and speed up the fit

# Targets and their weights in the residual (set a weight to 0 to drop it).
WEIGHT_FORCE = 1.0
WEIGHT_SPEED = 0.8

# The two depower DOFs are parametrized so they are ORTHOGONAL in the opti:
#   apd(l_dp) = apd_powered + (l_dp - l_dp_powered)/l_dp_span * delta_swing
#   - apd_powered : pitch at the POWERED state (l_dp = 1.7 m) -- the offset
#   - delta_swing : powered->depowered angle DIFFERENCE [rad] (legacy -0.13)
# The swing basis is zero at the powered anchor, so delta_swing cannot move
# apd_powered (and vice versa) -> no cross-coupling, far better conditioned than
# fitting the l_dp=0 intercept + slope. Converted to the ROM at the boundary:
#   delta_pitch_depower   = delta_swing / l_dp_span
#   angle_pitch_depower_0 = apd_powered - l_dp_powered * delta_pitch_depower
L_DP_POWERED = ROM_POWERED_INPUT_DEPOWER  # 1.7 m, anchor for the powered offset
L_DP_SPAN = ROM_DEPOWERED_INPUT_DEPOWER - ROM_POWERED_INPUT_DEPOWER  # 0.4 m
FIT_DELTA = True
DELTA_FIXED = -0.13  # swing [rad]; physical powered->depowered (legacy -0.13)

# CD steering-drag term: CD += CD_us * |u_s|. In the QS trim u_s is *solved*, so
# only the ratio CD_us/CS affects force/speed (u_s = required_side_force / CS).
# Fitting CD_us with CS fixed therefore identifies the steering-drag-per-side-
# force. It is only separable from the constant CD0 if the residual resolves the
# |u_s| variation -> this script fits PER-POINT (not phase means) when enabled.
FIT_CD_US = False
CD_US_FIXED = 0.0642857

# Initial guess (current rom_config.yaml values).
CD0_0 = 0.1130532
APD_POW_0 = -0.12  # powered apd [rad] at l_dp = 1.7 m (legacy powered pitch)
DELTA_0 = -0.13  # swing [rad]
CD_US_0 = 0.0642857

# Bounds for the fit.
CD0_BOUNDS = (0.0, 0.40)
APD_POW_BOUNDS = (-0.5, 0.5)  # powered apd [rad]
DELTA_BOUNDS = (-0.40, 0.0)  # swing [rad] between powered and depowered apd
CD_US_BOUNDS = (0.0, 0.5)  # CD += CD_us*|u_s| -> keep >= 0 (steering adds drag)

# Penalty residual for a fit point that fails to converge at some parameters.
FAIL_PENALTY = 2.0

WRITE_BACK = True  # if True, print a ready-to-paste YAML block (no file edit)

# Wind model used by the validator (logarithmic, z0 = 0.1 m).
WIND_Z0 = 0.1

# QS solve setup (rigid lumped tether path, as in the validator default).
UNKNOWN_VARS = ["tension_tether_ground", "input_steering", "speed_tangential"]
SOLVER_OPTIONS = {
    "ipopt": {"print_level": 0, "sb": "yes", "max_iter": 400},
    "print_time": False,
}
ACCEPT_RESIDUAL_NORM = 1.0  # ||g|| below this -> accept the QS solution
WIND_WINDOW = 50  # trailing-average window for friction velocity / wind direction


# ---------------------------------------------------------------------------
# Data loading (mirrors validate_quasi_steady_state_v3.py)
# ---------------------------------------------------------------------------
def _read_dict_from_group(group):
    out = {}
    for key, value in group.attrs.items():
        out[key] = value.decode("utf-8") if isinstance(value, bytes) else value
    for sub in group:
        out[sub] = _read_dict_from_group(group[sub])
    return out


def _read_hdf5(hdf5_path):
    with h5py.File(hdf5_path, "r") as hf:

        def frame(grp):
            return pd.DataFrame(
                {
                    col: (
                        grp[col][:].astype(str)
                        if grp[col].dtype.kind == "S"
                        else grp[col][:]
                    )
                    for col in grp
                    if isinstance(grp[col], h5py.Dataset)
                }
            )

        ekf = frame(hf["ekf_output"])
        flight = frame(hf["flight_data"])
        config = _read_dict_from_group(hf["config_data"])
    return ekf, flight, config


def _trailing_mean(values, window):
    """Trailing moving average matching the validator's windowing."""
    out = np.empty(len(values), dtype=float)
    buf = []
    for i, v in enumerate(values):
        buf.append(v)
        if len(buf) > window:
            buf.pop(0)
        out[i] = float(np.mean(buf))
    return out


def load_flight_data():
    """Return a per-row table with everything the QS solve needs."""
    path = f"{EKF_RESULTS_DIR}{KITE_MODEL}_{YEAR}-{MONTH}-{DAY}.h5"
    results, flight, _ = _read_hdf5(path)

    mask = flight.cycle.isin(list(CYCLE_RANGE))
    flight = flight[mask].reset_index(drop=True)
    results = results[mask].reset_index(drop=True)

    position = np.array(
        [results.kite_position_x, results.kite_position_y, results.kite_position_z]
    ).T
    velocity = np.array(
        [results.kite_velocity_x, results.kite_velocity_y, results.kite_velocity_z]
    ).T
    distance_radial = np.linalg.norm(position, axis=1)
    speed_tangential = np.linalg.norm(
        np.cross(position, velocity), axis=1
    ) / np.maximum(distance_radial, 1e-12)
    azimuth = np.arctan2(results.kite_position_y, results.kite_position_x).to_numpy()

    course_rate = np.gradient(np.unwrap(flight.kite_course), flight.time)
    course_rate = gaussian_filter1d(course_rate, sigma=1)

    wind = Wind(wind_model="logarithmic", z0=WIND_Z0, direction_wind=0)
    uf_raw = (
        results.wind_speed_horizontal
        * wind.kappa
        / np.log(results.kite_position_z / wind.z0)
    ).to_numpy()

    df = pd.DataFrame(
        {
            "time": flight.time.to_numpy(),
            "cycle": flight.cycle.to_numpy(),
            "distance_radial": distance_radial,
            "angle_course": flight.kite_course.to_numpy(),
            "speed_radial": flight.tether_reelout_speed.to_numpy(),
            "azimuth": azimuth,
            "angle_elevation": flight.kite_elevation.to_numpy(),
            "timeder_angle_course": course_rate,
            "speed_friction": _trailing_mean(uf_raw, WIND_WINDOW),
            "wind_direction": _trailing_mean(
                results.wind_direction.to_numpy(), WIND_WINDOW
            ),
            "input_depower": flight_dataframe_depower_to_power_tape_length(flight),
            "input_steering_measured": flight_dataframe_steering_to_us(flight),
            "measured_force": flight.ground_tether_force.to_numpy(),
            "measured_speed": speed_tangential,
            "kite_elevation": flight.kite_elevation.to_numpy(),
            "tether_reelout_speed": flight.tether_reelout_speed.to_numpy(),
        }
    )
    return df


# ---------------------------------------------------------------------------
# Phase masks (mirrors the validator)
# ---------------------------------------------------------------------------
def phase_masks(df):
    threshold = 0.5 * (ROM_POWERED_INPUT_DEPOWER + ROM_DEPOWERED_INPUT_DEPOWER)
    mask_pow = (
        (df.input_depower < threshold)
        & (df.kite_elevation < 0.75)
        & (df.tether_reelout_speed > 0.5)
    )
    mask_dep = (df.tether_reelout_speed < -0.5) & (df.input_depower >= threshold)
    return mask_pow.to_numpy(), mask_dep.to_numpy()


# ---------------------------------------------------------------------------
# QS prediction for a given parameter set
# ---------------------------------------------------------------------------
def build_kite_model(cd0, apd0, delta, cd_us):
    with open(LEI_V3_SYSTEM_FLOWN_CONFIG, "r") as f:
        cfg = yaml.safe_load(f)
    aero_input = load_aero_input_from_system_config(
        cfg, config_path=LEI_V3_SYSTEM_FLOWN_CONFIG
    )
    aero_input["params"]["CD0"] = float(cd0)
    aero_input["params"]["angle_pitch_depower_0"] = float(apd0)
    aero_input["params"]["delta_pitch_depower"] = float(delta)
    # CD steering-drag coefficient (kite.py applies it as CD += cd_us * |u_s|).
    for term in aero_input.get("coefficients", {}).get("CD", []):
        if term.get("var") == "u_s":
            term["coef"] = float(cd_us)

    wing = cfg["components"]["kite"]["wing"]["structure"]
    cs = cfg["components"]["kite"].get("control_system", {}).get("structure", {})
    tether_struct = cfg["components"].get("tether", {}).get("structure", {})

    kite = Kite(
        mass_wing=wing.get("mass", 14),
        area_wing=wing.get("projected_surface_area", 20),
        aero_input=aero_input,
        mass_kcu=cs.get("mass", 0.0),
        steering_control="asymmetric",
    )
    model = SystemModel(
        dof=3,
        quasi_steady=True,
        kite=kite,
        tether=RigidLumpedTether(diameter=tether_struct.get("diameter", 0.01)),
        wind_model=Wind(wind_model="logarithmic", z0=WIND_Z0, direction_wind=0),
    )
    model.setup_qs_solver(UNKNOWN_VARS, solver_options=SOLVER_OPTIONS)
    return model


def run_predictions(df, params):
    """Solve the QS trim for every (subsampled) row. Returns predicted force and
    speed arrays aligned to ``df`` (NaN where the trim did not converge)."""
    cd0, apd_pow, delta_swing, cd_us = params
    # Convert the orthogonal opti params (powered offset + swing) to the ROM's
    # l_dp=0 intercept + per-metre slope.
    coef = delta_swing / L_DP_SPAN
    apd0_intercept = apd_pow - L_DP_POWERED * coef
    model = build_kite_model(cd0, apd0_intercept, coef, cd_us)
    qs_inputs = model._qs_inputs

    pred_force = np.full(len(df), np.nan)
    pred_speed = np.full(len(df), np.nan)
    guess = np.array([1e5, 0.0, 60.0])

    for i in range(0, len(df), POINT_STRIDE):
        row = df.iloc[i]
        state = {
            "distance_radial": row.distance_radial,
            "angle_course": row.angle_course,
            "speed_radial": row.speed_radial,
            "angle_azimuth": row.azimuth - row.wind_direction,
            "angle_elevation": row.angle_elevation,
            "speed_friction": row.speed_friction,
            "timeder_angle_course": row.timeder_angle_course,
            "input_depower": row.input_depower,
        }
        guess[1] = row.input_steering_measured
        p = np.asarray([state[name] for name in qs_inputs], dtype=float)
        lbx, ubx, lbg, ubg = model.get_boundaries(state, UNKNOWN_VARS)
        try:
            sol = model._qs_solver(
                x0=guess,
                p=p,
                lbx=np.asarray(lbx, float),
                ubx=np.asarray(ubx, float),
                lbg=np.asarray(lbg, float),
                ubg=np.asarray(ubg, float),
            )
        except RuntimeError:
            guess = np.array([1e10, row.input_steering_measured, 100.0])
            continue
        if float(np.linalg.norm(sol["g"])) < ACCEPT_RESIDUAL_NORM:
            x = np.asarray(sol["x"], float).reshape(-1)
            pred_force[i] = x[0]
            pred_speed[i] = x[2]
            guess = x
        else:
            guess = np.array([1e10, row.input_steering_measured, 100.0])
    return pred_force, pred_speed


# ---------------------------------------------------------------------------
# Residuals and reporting
# ---------------------------------------------------------------------------
def phase_relative_errors(df, masks, pred_force, pred_speed):
    """Per-phase mean relative error for force and speed (NaN-safe)."""
    mask_pow, mask_dep = masks
    out = {}
    for name, m in (("powered", mask_pow), ("depowered", mask_dep)):
        valid = m & np.isfinite(pred_force) & np.isfinite(pred_speed)
        if valid.sum() < 2:
            out[name] = None
            continue
        mf_meas = float(np.mean(df.measured_force.to_numpy()[valid]))
        mf_pred = float(np.mean(pred_force[valid]))
        mv_meas = float(np.mean(df.measured_speed.to_numpy()[valid]))
        mv_pred = float(np.mean(pred_speed[valid]))
        out[name] = {
            "n": int(valid.sum()),
            "force_meas": mf_meas,
            "force_pred": mf_pred,
            "force_rel": (mf_pred - mf_meas) / mf_meas if mf_meas else 0.0,
            "speed_meas": mv_meas,
            "speed_pred": mv_pred,
            "speed_rel": (mv_pred - mv_meas) / mv_meas if mv_meas else 0.0,
        }
    return out


def unpack(x):
    """Map the optimizer vector to (cd0, apd_powered, delta_swing, cd_us)."""
    it = iter(x)
    cd0 = next(it)
    apd_pow = next(it)
    delta = next(it) if FIT_DELTA else DELTA_FIXED
    cd_us = next(it) if FIT_CD_US else CD_US_FIXED
    return (cd0, apd_pow, delta, cd_us)


def pack_x0_bounds():
    """Initial guess and bounds for the currently enabled free parameters."""
    x0 = [CD0_0, APD_POW_0]
    lb = [CD0_BOUNDS[0], APD_POW_BOUNDS[0]]
    ub = [CD0_BOUNDS[1], APD_POW_BOUNDS[1]]
    if FIT_DELTA:
        x0.append(DELTA_0)
        lb.append(DELTA_BOUNDS[0])
        ub.append(DELTA_BOUNDS[1])
    if FIT_CD_US:
        x0.append(CD_US_0)
        lb.append(CD_US_BOUNDS[0])
        ub.append(CD_US_BOUNDS[1])
    return x0, lb, ub


def make_residuals(df, fit_idx):
    """Per-point relative residual on force and speed over ``fit_idx``.

    Per-point (not phase-mean) so the |u_s| variation across turns/straights
    makes CD_us identifiable separately from the constant CD0.
    """
    f_meas = df.measured_force.to_numpy()
    v_meas = df.measured_speed.to_numpy()

    def residuals(x):
        params = unpack(x)
        pf, pv = run_predictions(df, params)
        rF = np.empty(len(fit_idx))
        rV = np.empty(len(fit_idx))
        for k, i in enumerate(fit_idx):
            if np.isfinite(pf[i]) and np.isfinite(pv[i]):
                rF[k] = (pf[i] - f_meas[i]) / f_meas[i] if f_meas[i] else 0.0
                rV[k] = (pv[i] - v_meas[i]) / v_meas[i] if v_meas[i] else 0.0
            else:
                rF[k] = rV[k] = FAIL_PENALTY
        r = np.concatenate([WEIGHT_FORCE * rF, WEIGHT_SPEED * rV])
        print(
            f"  CD0={params[0]:.5f} apd_pow={params[1]:+.4f} delta_sw={params[2]:+.4f} "
            f"cd_us={params[3]:.4f} -> rms={np.sqrt(np.mean(r**2)):.4f}"
        )
        return r

    return residuals


def report(df, masks, params, title):
    pf, pv = run_predictions(df, params)
    errs = phase_relative_errors(df, masks, pf, pv)
    print(f"\n=== {title} ===")
    print(
        f"  CD0={params[0]:.5f}  apd_pow={params[1]:+.4f}  "
        f"delta_swing={params[2]:+.4f} rad  cd_us={params[3]:.4f}"
    )
    for name in ("powered", "depowered"):
        e = errs[name]
        if e is None:
            print(f"  {name:10s}: <2 converged points")
            continue
        print(
            f"  {name:10s} (n={e['n']:4d}): "
            f"F {e['force_pred']:8.0f}/{e['force_meas']:8.0f} N "
            f"({100*e['force_rel']:+5.1f}%) | "
            f"v {e['speed_pred']:5.1f}/{e['speed_meas']:5.1f} m/s "
            f"({100*e['speed_rel']:+5.1f}%)"
        )
    return errs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    df = load_flight_data()
    masks = phase_masks(df)
    mask_pow, mask_dep = masks
    print(
        f"Loaded {len(df)} rows over cycles {min(df.cycle)}..{max(df.cycle)}; "
        f"powered={mask_pow.sum()} depowered={mask_dep.sum()}"
    )

    x0_full = (CD0_0, APD_POW_0, DELTA_0, CD_US_0)
    report(df, masks, x0_full, "Initial (current rom_config)")

    # Fit points: converged at the initial guess and inside a fitted phase.
    pf0, pv0 = run_predictions(df, x0_full)
    fit_mask = (mask_pow | mask_dep) & np.isfinite(pf0) & np.isfinite(pv0)
    fit_idx = np.where(fit_mask)[0]
    print(
        f"\nFitting per-point (force + speed) on {len(fit_idx)} points "
        f"[{', '.join(['CD0', 'apd0'] + (['delta'] if FIT_DELTA else []) + (['cd_us'] if FIT_CD_US else []))}]..."
    )

    x0, lb, ub = pack_x0_bounds()
    sol = least_squares(
        make_residuals(df, fit_idx),
        x0=x0,
        bounds=(lb, ub),
        diff_step=1e-3,
        xtol=1e-4,
        verbose=0,
    )
    fitted = unpack(sol.x)
    report(df, masks, fitted, "Fitted")

    cd0, apd_pow, delta_swing, cd_us = fitted
    delta_coef = delta_swing / L_DP_SPAN
    apd0_intercept = apd_pow - L_DP_POWERED * delta_coef
    print("\nFitted parameters (orthogonal opti vars):")
    print(f"  CD0                   : {cd0:.6f}")
    print(
        f"  apd_powered (l_dp=1.7): {apd_pow:.6f} rad  ({np.degrees(apd_pow):+.2f} deg)"
    )
    print(
        f"  delta swing (pow->dep): {delta_swing:.6f} rad"
        f"  ({'fitted' if FIT_DELTA else 'fixed'})"
    )
    print("  -> ROM coefficients:")
    print(f"     angle_pitch_depower_0: {apd0_intercept:.6f}  [rad] (l_dp=0 intercept)")
    print(
        f"     delta_pitch_depower  : {delta_coef:.6f}  [rad/m] = swing / {L_DP_SPAN:.2f}"
    )
    print(
        f"  CD u_s coef           : {cd_us:.6f}  ({'fitted' if FIT_CD_US else 'fixed'})"
    )

    if WRITE_BACK:
        print("\n--- rom_config.yaml  aerodynamics.params: ---")
        print(f"    CD0: {cd0:.6f}")
        print(f"    angle_pitch_depower_0: {apd0_intercept:.6f}")
        print(f"    delta_pitch_depower: {delta_coef:.6f}")
        print("--- rom_config.yaml  aerodynamics.coefficients.CD: (u_s term) ---")
        print(f"      - var: u_s\n        power: 1\n        coef: {cd_us:.6f}")


if __name__ == "__main__":
    main()

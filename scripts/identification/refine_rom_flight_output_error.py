"""Output-error refinement of the flight-corrected aerostructural ROM.

Step 3 of the flight correction (step 1-2: identify_rom_flight_correction.py).
Steps 1-2 fit theta_b and dC_D to the EKF coefficients at the flown state. The
quasi-steady reconstruction does not land at that state, though: theta_b was
taken from the flown bridle-force direction, which the real (sagging) tether
sets, and the ROM's straight tether pulls along a different line, so the same
theta_b puts the ROM at a different alpha (held-out reel-out: alpha 5.9 vs 4.8
deg, tension +55 %, v_tau +19 %). Here the SAME parameters

    angle_pitch_tether_0, slope_angle_pitch_tether_depower   (theta_b)
    CD0, C_D u_p coefficient, C_D u_s^2 coefficient          (dC_D)
    gain_roll_steering                     (--gain turnlaw, the default)

are refined so that the ROM's own quasi-steady solve reproduces the measured
tether force and tangential speed, with the turn rate prescribed and the
steering solved (validate_quasi_steady_state_v3.py, solved mode, rigid tether:
the model the optimisations use). With the roll gain free, the solved
steering's error against the logged steering (+ the 2019 zero offset) joins
the objective: force and speed hardly see the gain, which mostly sets how much
steering the solve uses, and a weak gain makes the solve over-steer and
over-brake through the u_s^2 drag. That variant (--gain steering) drove the
gain to -4.9 (non-physical): the logged steering explains the flown turn
poorly (lag, gravity and tether turning), so the steering error rewards
switching steering off. The measured aerodynamic roll (--gain joint) holds
the gain at -0.75, but that roll is the wing-inertia identity and includes
the roll needed against the ROM's gravity-driven turn, which is 2.4x the
flight's (point-mass coefficient ~1 vs 0.47 measured): it reads 20 % high
against the turn-rate law chi_dot = K v_a u_s + g-term the paper defines the
steering by (flight K 0.343 rad/m; the ROM with -0.75 gives 0.41). The
default (--gain turnlaw) therefore fits all parameters together with the
turn-rate law as the gain's condition: the solved steering must regress 1:1
on the logged steering (zero offset, 0.3 s lag) over the reel-out training
samples, one residual SCALE_SLOPE (slope - 1) with its exact Jacobian from
the per-sample sensitivities. The steering-drag coefficients are bounded at
zero in total (LOWER_BOUNDS): unbounded, the fit met the slope with the gain
at -0.77 and a NEGATIVE u_s^2 drag (-0.25), drag buying the steering slope
instead of the gain. --gain fixed keeps the file's gain. Lift
coefficients and stall stay the aerostructural ones. The gravity-turn excess
itself is a limitation of the point-mass model, documented, not fitted.

Nested least squares: every sample's root (tension, u_s, v_tau) is found by a
Newton rootfinder, mapped over the samples and warm-started from its last
root; its derivative with respect to the parameters follows from the
implicit-function theorem (the parameters enter the CasADi model as symbols),
so the outer Gauss-Newton (scipy least_squares, trust region) has exact
Jacobians. Errors are scaled by their phase medians, phases weighted equally;
a sample without a valid root (Newton failure, past the stall, non-physical)
is charged a fixed error. A single NLP with every sample's balance as a
constraint is infeasible: reel-in samples sit at the radial-balance fold and
lose their root when the parameters move.

Inputs: the per-sample CSVs the validator writes for TRAINING cycles
(``--out``), run with the step 1-2 ROM, which also seed the per-sample
unknowns. Cycles 60-67 stay held out. The refined values are written into
rom_config_aerostructural_flight_corrected.yaml (the starting values are kept
in its metadata). Re-run identify_rom_flight_correction.py first to start over
from steps 1-2.

Usage (project root)
    python scripts/identification/refine_rom_flight_output_error.py
    python scripts/identification/refine_rom_flight_output_error.py --no-write
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import io
import json
from datetime import date
from pathlib import Path

import casadi as ca
import numpy as np
import pandas as pd
import yaml

from awetrim import SystemModel
from awetrim.environment.Wind import Wind
from awetrim.system.factory import create_system_model_from_yaml
from awetrim.system.kite import Kite
from awetrim.system.tether import RigidLumpedTether
from awetrim.utils.config_paths import (
    LEI_V3_ROM_AEROSTRUCTURAL_FLIGHT_CONFIG,
    LEI_V3_ROM_SEMI_EMPIRICAL_CONFIG,
    LEI_V3_SYSTEM_FLOWN_CONFIG,
)

OUT_DIR = Path("results/LEI-V3-KITE/identification/rom_flight_correction/output_error")
#: Steps 1-2 outputs (flight_samples.csv with the measured roll, the report);
#: the roll samples are ROM-independent measurements.
CORRECTION_DIR = OUT_DIR.parent / "alpha_lift"
TRAIN_GLOB = "train_*.csv"
#: What is refitted, per ROM: name -> ("param", key) or ("CD", var, power, abs).
#: The aerostructural-flight set is the flight correction's; the
#: semi-empirical set is the paper's calibration (CD0, theta_b, roll gain and
#: the |u_s| steering drag), recalibrated the same way on the corrected
#: bridle-angle frame (2026-10-07).
ROM_SPECS = {
    "aerostructural_flight": {
        "path": LEI_V3_ROM_AEROSTRUCTURAL_FLIGHT_CONFIG, "out_dir": OUT_DIR,
        "preserve_comments": False,
        "parameters": {"theta0": ("param", "angle_pitch_tether_0"),
                       "theta1": ("param", "slope_angle_pitch_tether_depower"),
                       "cd0": ("param", "CD0"), "cd_up": ("CD", "u_p", 1, False),
                       "cd_us2": ("CD", "u_s", 2, False),
                       "gain": ("param", "gain_roll_steering")}},
    "semi_empirical": {
        "path": LEI_V3_ROM_SEMI_EMPIRICAL_CONFIG,
        "out_dir": OUT_DIR.with_name("output_error_semi_empirical"),
        "preserve_comments": True,
        "parameters": {"theta0": ("param", "angle_pitch_tether_0"),
                       "theta1": ("param", "slope_angle_pitch_tether_depower"),
                       "cd0": ("param", "CD0"), "cd_us_abs": ("CD", "u_s", 1, True),
                       "gain": ("param", "gain_roll_steering")}},
}
HOLDOUT_CYCLES = range(60, 68)
#: Samples per phase entering the NLP (evenly spread over the training rows).
SAMPLES_PER_PHASE = 600
UNKNOWNS = ("tension_tether_ground", "input_steering", "speed_tangential")
#: Set from ROM_SPECS in main(); the default is the flight correction's.
PARAMETERS = tuple(ROM_SPECS["aerostructural_flight"]["parameters"])
#: Steering-error scale [-] (about the spread of the logged reel-out steering).
SCALE_STEERING = 0.1
#: Turn-rate-law condition (--gain turnlaw): the solved steering must regress
#: 1:1 on the logged steering over the reel-out training samples; one residual
#: SCALE_SLOPE * (slope - 1). At 5 a 2 % slope error costs as much as a 10 %
#: tension error over every sample, so the condition pins the gain while the
#: force rows, which barely see it, keep the rest.
SCALE_SLOPE = 5.0
#: The 0.3 s steering lag of the flight's turn response, in 10 Hz samples.
STEERING_LAG_SAMPLES = 3
#: Physical bounds on the refined parameters: the steering-drag coefficients
#: (the file's TOTAL u_s^2 / |u_s| coefficient) may not go negative -- steering
#: cannot reduce the drag -- or the fit buys the steering slope with drag
#: instead of the roll gain (it did: gain -0.77 with -0.25 u_s^2).
LOWER_BOUNDS = {"cd_us2": 0.0, "cd_us_abs": 0.0}
#: The 2019 rig's steering zero offset, added to the logged u_s (the CSV keeps
#: the logged value; the validator adds it for --steering measured).
from awetrim.identification.controls import FLIGHT_STEERING_ZERO_OFFSET_2019  # noqa: E402
#: Keep the solved alpha this far below the ROM stall [deg].
STALL_CLEARANCE_DEG = 1.0
MIN_SPEED_TANGENTIAL = 0.5
MAX_ABS_STEERING = 0.5
#: Per-sample unknown scales (tension, u_s, v_tau): the NLP works in O(1) units.
UNKNOWN_SCALE = np.array([1000.0, 0.1, 10.0])
#: Scaled error charged to a sample that has no valid root at the parameters.
FAIL_PENALTY = 1.0
SPEC = ROM_SPECS["aerostructural_flight"]


# ───────────────────────────────────────────────────────────────── model
def _cd_term(aero: dict, var: str, power: int, absolute: bool) -> dict:
    hits = [e for e in aero["coefficients"]["CD"]
            if e.get("var") == var and int(e.get("power", 1)) == power
            and bool(e.get("abs", False)) == absolute]
    if len(hits) != 1:
        raise SystemExit(f"C_D needs exactly one {'|' if absolute else ''}{var}^{power}"
                         f"{'|' if absolute else ''} term, found {len(hits)}")
    return hits[0]


def rom_values(aero: dict, spec: dict) -> dict:
    """The refined parameters as stored in a ROM aerodynamics block."""
    out = {}
    for name, where in spec["parameters"].items():
        out[name] = (aero["params"][where[1]] if where[0] == "param"
                     else _cd_term(aero, *where[1:])["coef"])
    return out


def with_values(aero: dict, values: dict, spec: dict) -> dict:
    """A copy of the aerodynamics block with the refined parameters replaced."""
    out = copy.deepcopy(aero)
    for name, where in spec["parameters"].items():
        if where[0] == "param":
            out["params"][where[1]] = values[name]
        else:
            _cd_term(out, *where[1:])["coef"] = values[name]
    return out


def write_values_preserving(path: Path, values: dict, spec: dict, note: str) -> None:
    """Replace the refined numbers in a hand-written ROM file, keeping its comments.

    Params are matched as ``key: <number>`` lines; a C_D term as the ``coef:``
    line that follows its ``var:`` line (with the matching ``abs``). ``note``
    is inserted as ``metadata.recalibration``.
    """
    import re
    text = path.read_text(encoding="utf-8")
    for name, where in spec["parameters"].items():
        value = f"{values[name]:.6f}"
        if where[0] == "param":
            pattern = re.compile(rf"^(\s*{where[1]}:\s*)[-+0-9.eE]+", re.M)
            text, n = pattern.subn(rf"\g<1>{value}", text)
        else:
            var, power, absolute = where[1:]
            # The CD block: from "    CD:" to the next top-level key.
            start = text.index("\n    CD:\n") + 1
            end = text.find("\ntether:", start)
            block = text[start:end]
            entries = re.compile(
                rf"(-\s*var:\s*{var}\n(?:\s*#.*\n)*\s*power:\s*{power}\n(?:\s*#.*\n)*\s*coef:\s*)"
                r"([-+0-9.eE]+)([^\n]*\n)(\s*abs:\s*true)?")
            n = 0
            def repl(m):
                nonlocal n
                if bool(m.group(4)) != absolute:
                    return m.group(0)
                n += 1
                return m.group(1) + value + m.group(3) + (m.group(4) or "")
            block = entries.sub(repl, block)
            text = text[:start] + block + text[end:]
        if n != 1:
            raise SystemExit(f"{name}: expected one match in {path.name}, found {n}")
    if "  recalibration:" not in text:
        text = text.replace("metadata:\n", "metadata:\n  recalibration: >-\n    " + note + "\n", 1)
    path.write_text(text, encoding="utf-8", newline="\n")


def build_functions(aero: dict):
    """(residual, alpha) CasADi functions of (x, state, theta) for the validator model."""
    with contextlib.redirect_stdout(io.StringIO()):
        base = create_system_model_from_yaml(
            LEI_V3_SYSTEM_FLOWN_CONFIG, aero_yaml_path=SPEC["path"],
            steering_control="asymmetric").kite
    theta = {n: ca.MX.sym("rom_" + n) for n in PARAMETERS}
    kite = Kite(mass_wing=base.mass_wing, mass_kcu=base.mass_kcu, length_kcu=base.length_kcu,
                diameter_kcu=base.diameter_kcu, diameter_turbine=base.diameter_turbine,
                thrust_coefficient_turbine=base.thrust_coefficient_turbine,
                area_wing=base.area_wing, aero_input=with_values(aero, theta, SPEC),
                steering_control="asymmetric")
    model = SystemModel(dof=3, quasi_steady=True, kite=kite,
                        tether=RigidLumpedTether(diameter=0.01),
                        wind_model=Wind(wind_model="logarithmic", z0=0.1, direction_wind=0))
    model.establish_residual()
    g = model.residual
    alpha = kite.angle_of_attack_for(model)
    x = ca.vertcat(*[getattr(model, n) for n in UNKNOWNS])
    names_x = {s.name() for s in ca.symvar(x)}
    names_t = {s.name() for s in theta.values()}
    state = [s for s in ca.symvar(ca.vertcat(g, alpha))
             if s.name() not in names_x | names_t]
    t = ca.vertcat(*[theta[n] for n in PARAMETERS])
    s = ca.vertcat(*state)
    # Expanded to SX: the per-sample functions are small, and SX derivatives
    # are an order of magnitude faster than the MX graph over 1000+ samples.
    residual = ca.Function("residual", [x, s, t], [g]).expand()
    alpha_fun = ca.Function("alpha", [x, s, t], [alpha]).expand()
    return residual, alpha_fun, [v.name() for v in state]


# ───────────────────────────────────────────────────────────────── data
def load_training(stride_seed: int = 0) -> pd.DataFrame:
    frames = [pd.read_csv(p) for p in sorted(SPEC["out_dir"].glob(TRAIN_GLOB))]
    if not frames:
        raise SystemExit(f"no {SPEC['out_dir'] / TRAIN_GLOB}: run the validator on training cycles "
                         "with --out (see the module docstring)")
    d = pd.concat(frames, ignore_index=True)
    # The logged steering the turn-rate law is fitted against: zero offset
    # and the 0.3 s steering lag, on the contiguous 10 Hz rows BEFORE any
    # row is dropped or subsampled.
    d["steering_logged_lagged"] = (
        d.measured_input_steering + FLIGHT_STEERING_ZERO_OFFSET_2019
    ).groupby(d.cycle).shift(STEERING_LAG_SAMPLES)
    d = d[~d.cycle.isin(HOLDOUT_CYCLES) & d.predicted_tension.notna()
          & d.steering_logged_lagged.notna()]
    reel_out = (d.speed_radial > 0.5) & (d.input_depower < 1.9) & (d.kite_elevation < 0.75)
    reel_in = (d.speed_radial < -0.5) & (d.input_depower >= 1.9)
    d = d.assign(phase=np.where(reel_out, "reel-out", np.where(reel_in, "reel-in", "")))
    picks = []
    for ph in ("reel-out", "reel-in"):
        g = d[d.phase == ph]
        idx = np.linspace(0, len(g) - 1, min(SAMPLES_PER_PHASE, len(g))).astype(int)
        picks.append(g.iloc[idx])
    return pd.concat(picks, ignore_index=True)


def state_matrix(d: pd.DataFrame, names: list[str]) -> np.ndarray:
    columns = {"distance_radial": d.distance_radial, "angle_course": d.angle_course,
               "speed_radial": d.speed_radial, "angle_azimuth": d.angle_azimuth,
               "angle_elevation": d.angle_elevation, "speed_friction": d.speed_friction,
               "timeder_angle_course": d.measured_course_rate,
               "input_depower": d.input_depower}
    return np.vstack([np.asarray(columns[n], dtype=float) for n in names])


# ───────────────────────────────────────────────────────────────── fit
class SampleSolver:
    """Every sample's quasi-steady root at given parameters, plus d(root)/d(theta).

    A Newton rootfinder on the 3 force rows in (tension, u_s, v_tau), mapped over
    the samples; its parameter sensitivities come from the implicit-function
    theorem (CasADi differentiates the rootfinder). Each sample is warm-started
    from its last accepted root; a sample whose Newton fails, lands past the
    stall or at a non-physical point is flagged invalid for that evaluation.
    """

    def __init__(self, residual, alpha_fun, states: np.ndarray, seeds: np.ndarray,
                 alpha_max: float):
        n = states.shape[1]
        x = ca.SX.sym("x", 3)
        s = ca.SX.sym("s", states.shape[0])
        t = ca.SX.sym("t", len(PARAMETERS))
        scale = ca.DM(UNKNOWN_SCALE)
        # Scaled unknowns keep Newton well conditioned (N vs rad vs m/s).
        f = ca.Function("f", [x, ca.vertcat(s, t)], [residual(x * scale, s, t) / 1000.0])
        root = ca.rootfinder("root", "newton", f, {
            "error_on_fail": False, "abstol": 1e-10, "max_iter": 40})
        x0 = ca.MX.sym("x0", 3)
        sm = ca.MX.sym("s", states.shape[0])
        tm = ca.MX.sym("t", len(PARAMETERS))
        xr = root(x0, ca.vertcat(sm, tm))
        xr_phys = xr * scale
        outputs = [xr_phys, ca.jacobian(xr_phys, tm),
                   ca.norm_2(residual(xr_phys, sm, tm)), alpha_fun(xr_phys, sm, tm)]
        one = ca.Function("one", [x0, sm, tm], outputs)
        self.one = one
        self.batch = one.map(n, "thread", 8)
        self.states = states
        self.last = seeds.copy()
        self.seeds = seeds.copy()
        self.alpha_max = alpha_max

    def _evaluate(self, x0, tm):
        """The mapped solve; sample by sample if one sample throws inside it."""
        try:
            return tuple(np.asarray(v, dtype=float) for v in self.batch(x0, self.states, tm))
        except RuntimeError:
            n, k = self.states.shape[1], len(PARAMETERS)
            x, jac = np.full((3, n), np.nan), np.zeros((3, k * n))
            res, alpha = np.full((1, n), np.inf), np.full((1, n), np.inf)
            for i in range(n):
                try:
                    out = self.one(x0[:, i], self.states[:, i], tm[:, i])
                except RuntimeError:
                    continue
                x[:, i] = np.asarray(out[0]).ravel()
                jac[:, k * i:k * (i + 1)] = np.asarray(out[1])
                res[0, i], alpha[0, i] = float(out[2]), float(out[3])
            return x, jac, res, alpha

    def __call__(self, theta: np.ndarray):
        n = self.states.shape[1]
        tm = np.repeat(np.asarray(theta, float)[:, None], n, axis=1)
        best = None
        # Warm start from the last accepted root, then from the step 1-2 seed.
        for start in (self.last, self.seeds):
            x, jac, res, alpha = self._evaluate(start / UNKNOWN_SCALE[:, None], tm)
            res, alpha = res.ravel(), alpha.ravel()
            ok = ((res < 1e-3) & (alpha < self.alpha_max) & (x[0] > 0.0)
                  & (x[2] > MIN_SPEED_TANGENTIAL) & (np.abs(x[1]) < MAX_ABS_STEERING)
                  & np.isfinite(x).all(axis=0))
            if best is None:
                best = (x, jac, ok)
            else:
                take = ok & ~best[2]
                best = (np.where(take, x, best[0]),
                        np.where(np.repeat(take, len(PARAMETERS))[None, :], jac, best[1]),
                        best[2] | ok)
        x, jac, ok = best
        self.last = np.where(ok[None, :], x, self.last)
        # jac is 3 x (5 n): per sample a 3 x 5 block, laid out sample-major.
        jac = jac.reshape(3, n, len(PARAMETERS))
        return x, jac, ok


def load_roll() -> tuple[pd.DataFrame, float]:
    """Training samples of the measured roll (steps 1-2) and its residual rms [rad]."""
    flight = pd.read_csv(CORRECTION_DIR / "flight_samples.csv")
    flight = flight[(flight.split == "fit") & np.isfinite(flight.phi_a_w)
                    & np.isfinite(flight.u_s_lagged)].reset_index(drop=True)
    report = json.loads((CORRECTION_DIR / "correction_report.json").read_text(encoding="utf-8"))
    return flight, float(np.deg2rad(report["roll_gain"]["rms_deg"]))


def fit(d: pd.DataFrame, residual, alpha_fun, names, start: dict, alpha_stall: float,
        free: tuple[str, ...], gain_mode: str = "fixed"):
    """Gauss-Newton (scipy least_squares, trust region) over the ``free`` parameters.

    ``gain_mode`` says what holds the roll gain when it is free:
    ``"joint"`` -- the measured aerodynamic roll of steps 1-2, phi_a,w - k u_s
    on every training sample (scaled by its rms), next to tension and speed:
    all six parameters in one fit; ``"steering"`` -- the solved steering's
    error against the logged steering (REJECTED: the gain runs away).
    """
    from scipy.optimize import least_squares

    n = len(d)
    seeds = np.vstack([d.predicted_tension, d.predicted_input_steering,
                       d.predicted_speed_tangential]).astype(float)
    solver = SampleSolver(residual, alpha_fun, state_matrix(d, names), seeds,
                          alpha_stall - np.deg2rad(STALL_CLEARANCE_DEG))
    w = np.zeros(n)
    scale_t, scale_v = np.ones(n), np.ones(n)
    for ph in ("reel-out", "reel-in"):
        m = (d.phase == ph).to_numpy()
        if m.any():
            w[m] = 0.5 / m.sum()
            scale_t[m] = np.median(d.measured_tension[m])
            scale_v[m] = np.median(d.measured_speed_tangential[m])
    sw = np.sqrt(w)
    t_meas = d.measured_tension.to_numpy(float)
    v_meas = d.measured_speed_tangential.to_numpy(float)
    s_meas = d.measured_input_steering.to_numpy(float) + FLIGHT_STEERING_ZERO_OFFSET_2019
    with_steering = "gain" in free and gain_mode == "steering"
    assert set(free) <= set(PARAMETERS), free
    with_roll = "gain" in free and gain_mode == "joint"
    with_slope = "gain" in free and gain_mode == "turnlaw"
    if with_slope:
        # Reel-out samples only: the turn-rate law is a reel-out relation.
        slope_mask = (d.phase == "reel-out").to_numpy()
        s_logged = d.steering_logged_lagged.to_numpy(float)
    if with_roll:
        roll, roll_rms = load_roll()
        roll_w = np.zeros(len(roll))
        for ph in ("reel-out", "reel-in"):
            m = (roll.phase == ph).to_numpy()
            roll_w[m] = 0.5 / m.sum()
        roll_sw = np.sqrt(roll_w) / roll_rms
        roll_phi = roll.phi_a_w.to_numpy(float)
        roll_us = roll.u_s_lagged.to_numpy(float)
        gain_index = PARAMETERS.index("gain")
    index = [PARAMETERS.index(k) for k in free]
    full0 = np.array([start[k] for k in PARAMETERS])
    cache = {}

    def full(theta_free):
        out = full0.copy()
        out[index] = theta_free
        return out

    def evaluate(theta_free):
        theta = full(theta_free)
        key = tuple(np.round(theta, 14))
        if key not in cache:
            cache.clear()
            cache[key] = solver(theta)
        return cache[key]

    def fun(theta):
        x, _, ok = evaluate(theta)
        e_t = np.where(ok, (x[0] - t_meas) / scale_t, FAIL_PENALTY)
        e_v = np.where(ok, (x[2] - v_meas) / scale_v, FAIL_PENALTY)
        parts = [sw * e_t, sw * e_v]
        if with_steering:
            parts.append(sw * np.where(ok, (x[1] - s_meas) / SCALE_STEERING, FAIL_PENALTY))
        if with_roll:
            parts.append(roll_sw * (roll_phi - full(theta)[gain_index] * roll_us))
        if with_slope:
            m = slope_mask & ok
            slope = (x[1][m] @ s_logged[m]) / (s_logged[m] @ s_logged[m])
            parts.append(np.array([SCALE_SLOPE * (slope - 1.0)]))
        return np.concatenate(parts)

    def jac(theta):
        _, J, ok = evaluate(theta)
        jt = np.where(ok[:, None], J[0] / scale_t[:, None], 0.0) * sw[:, None]
        jv = np.where(ok[:, None], J[2] / scale_v[:, None], 0.0) * sw[:, None]
        rows = [jt, jv]
        if with_steering:
            rows.append(np.where(ok[:, None], J[1] / SCALE_STEERING, 0.0) * sw[:, None])
        if with_roll:
            jr = np.zeros((len(roll_phi), len(PARAMETERS)))
            jr[:, gain_index] = -roll_sw * roll_us
            rows.append(jr)
        if with_slope:
            m = slope_mask & ok
            # d slope / d theta = sum_i (du_s,i/d theta) u_logged,i / sum u_logged^2
            rows.append(SCALE_SLOPE * (s_logged[m] @ J[1][m]) [None, :]
                        / (s_logged[m] @ s_logged[m]))
        return np.vstack(rows)[:, index]

    t0 = full0[index]
    lower = np.array([LOWER_BOUNDS.get(k, -np.inf) for k in free])
    t0 = np.maximum(t0, lower + 1e-6)
    r0 = fun(t0)
    ok0 = evaluate(t0)[2].mean()
    result = least_squares(fun, t0, jac=jac, method="trf", x_scale="jac",
                           bounds=(lower, np.full(len(free), np.inf)),
                           max_nfev=60, ftol=1e-6, verbose=2)
    x, _, ok = evaluate(result.x)
    values = dict(zip(PARAMETERS, map(float, full(result.x))))
    extra = {}
    if with_roll:
        k = values["gain"]
        extra["roll_rms_deg"] = float(np.rad2deg(np.sqrt(np.mean((roll_phi - k * roll_us) ** 2))))
        extra["roll_samples"] = int(len(roll_phi))
    if with_slope:
        m = slope_mask & ok
        extra["steering_slope"] = float((x[1][m] @ s_logged[m]) / (s_logged[m] @ s_logged[m]))
        extra["steering_slope_samples"] = int(m.sum())
    samples = np.where(ok[None, :], x, np.nan).T
    return values, samples, {"success": bool(result.success), "message": result.message,
                             "evaluations": int(result.nfev),
                             "objective_start": float(r0 @ r0),
                             "objective": float(2 * result.cost),
                             "valid_start": float(ok0), "valid": float(ok.mean()), **extra}


def phase_errors(d: pd.DataFrame, tension, speed, steering) -> dict:
    out = {}
    for ph in ("reel-out", "reel-in"):
        m = (d.phase == ph).to_numpy()
        e_t = 100 * (tension[m] / d.measured_tension.to_numpy()[m] - 1)
        e_v = 100 * (speed[m] / d.measured_speed_tangential.to_numpy()[m] - 1)
        if not m.any():
            continue
        out[ph] = {"tension_median_%": float(np.nanmedian(e_t)),
                   "v_tau_median_%": float(np.nanmedian(e_v)), "n": int(m.sum()),
                   "steering_rms": float(np.sqrt(np.nanmean(
                       (steering[m] - d.measured_input_steering.to_numpy()[m]
                        - FLIGHT_STEERING_ZERO_OFFSET_2019) ** 2))),
                   "solved": int(np.isfinite(e_t).sum())}
    return out


def main() -> None:
    global SPEC, PARAMETERS
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--no-write", action="store_true")
    parser.add_argument("--phases", choices=("both", "reel-out"), default="both",
                        help="phases the objective uses (reel-in states sit at the "
                             "radial-balance fold and may have no root)")
    parser.add_argument("--gain", choices=("turnlaw", "joint", "fixed", "steering"),
                        default="turnlaw",
                        help="turnlaw (default): all parameters together, the roll gain "
                             "set by the turn-rate law, i.e. the solved steering must "
                             "regress 1:1 on the logged steering (reel-out); joint: the "
                             "gain held by the measured aerodynamic roll instead (that "
                             "roll is the wing-inertia identity and includes the ROM's "
                             "2.4x too strong gravity turn, so it reads 20 %% high); "
                             "fixed: keep the file's gain; steering: free gain against a "
                             "per-sample steering error (REJECTED: runs to -4.9)")
    parser.add_argument("--free", default=None,
                        help="comma list of the parameters to refit (subset of "
                             + ",".join(PARAMETERS) + "); default: all parameters with --gain turnlaw, "
                             "the first five with --gain fixed")
    parser.add_argument("--report-name", default="refinement_report.json")
    parser.add_argument("--rom", choices=sorted(ROM_SPECS), default="aerostructural_flight",
                        help="which ROM to refit (its parameter set in ROM_SPECS)")
    args = parser.parse_args()
    SPEC = ROM_SPECS[args.rom]
    PARAMETERS = tuple(SPEC["parameters"])
    rom_path = SPEC["path"]

    doc = yaml.safe_load(rom_path.read_text(encoding="utf-8"))
    aero = doc["aerodynamics"]
    start = rom_values(aero, SPEC)
    residual, alpha_fun, names = build_functions(aero)
    d = load_training()
    if args.phases == "reel-out":
        d = d[d.phase == "reel-out"].reset_index(drop=True)
    print(f"{len(d)} training samples ({d.cycle.nunique()} cycles); start {start}")
    before = phase_errors(d, d.predicted_tension.to_numpy(),
                          d.predicted_speed_tangential.to_numpy(),
                          d.predicted_input_steering.to_numpy())
    free = PARAMETERS if args.gain != "fixed" else tuple(k for k in PARAMETERS if k != "gain")
    if args.free:
        free = tuple(k for k in PARAMETERS if k in args.free.split(","))
        if args.gain in ("joint", "turnlaw") and "gain" not in free:
            args.gain = "fixed"
    values, samples, stats = fit(d, residual, alpha_fun, names, start,
                                 float(aero["params"]["angle_of_attack_stall"]), free,
                                 args.gain)
    after = phase_errors(d, samples[:, 0], samples[:, 2], samples[:, 1])
    print(json.dumps({"stats": stats, "start": start, "refined": values,
                      "training_errors_before": before, "training_errors_after": after},
                     indent=2))

    report = {"rom": args.rom, "phases": args.phases, "free_parameters": list(free),
              "gain_mode": args.gain,
              "method": "output error: quasi-steady tension and v_tau, solved steering, "
                        "rigid tether, phases weighted equally",
              "training_cycles": sorted(int(c) for c in d.cycle.unique()),
              "samples": int(len(d)), "solver": stats,
              "start_values": start, "refined_values": values,
              "training_errors_before": before, "training_errors_after": after}
    SPEC["out_dir"].mkdir(parents=True, exist_ok=True)
    (SPEC["out_dir"] / args.report_name).write_text(json.dumps(report, indent=2),
                                                     encoding="utf-8")
    if not stats["success"]:
        raise SystemExit("IPOPT did not converge; ROM file left unchanged")
    if not args.no_write:
        if SPEC["preserve_comments"]:
            note = (f"{date.today().isoformat()}: {', '.join(free)} recalibrated by "
                    "scripts/identification/refine_rom_flight_output_error.py --rom "
                    f"{args.rom} (output error on the quasi-steady tension and v_tau of "
                    f"{len(report['training_cycles'])} 2019 training cycles, cycles 60-67 "
                    "held out, roll gain held by the measured aerodynamic roll) on the "
                    "corrected bridle-angle frame of Kite._force_in_wind_frame; the "
                    "paper's values were calibrated on the pre-fix frame. Report: "
                    f"{SPEC['out_dir'] / args.report_name}")
            write_values_preserving(rom_path, values, SPEC, note)
        else:
            doc["aerodynamics"] = with_values(aero, values, SPEC)
            doc["metadata"]["output_error_refinement"] = {
                **report, "refined_by": "scripts/identification/refine_rom_flight_output_error.py",
                "refined_on": date.today().isoformat()}
            text = rom_path.read_text(encoding="utf-8")
            header = "".join(line + "\n" for line in text.splitlines() if line.startswith("#"))
            rom_path.write_text(header + yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
        print(f"Wrote {rom_path}")


if __name__ == "__main__":
    main()

"""
Turn rate law identification from flight data.

Implements the three formulations from:
  Cayon & Schmehl, "Quasi-Steady Mechanics of Tethered Flight"

  Eq. (41)  Simple:       chi_dot_b = gk * va * us
    Eq. (40)  Two-term:     chi_dot_b = c1*(va*us) + c2*(sin(chi)*cos(beta)/va)
    Eq. (38)  Full rational: chi_dot_b = -(k1*va^2*us + m*g*sin(chi)*cos(beta))
                                                                             / (k4*m*v_tau + k2*va)
    Eq. (39)  Full + course term: chi_dot_b = -(k1*va^2*us + m*g*sin(chi)*cos(beta)
                                                                                         + k5*cos(chi)*cos(beta)
                                                                                         + k6*sin(beta))
                                                                             / (k4*m*v_tau + k2*va)

Parameters identified by least squares (Eq. 41, 40) or nonlinear fit (Eq. 38):
  gk  = -0.5*rho*S*K_us / (m + 0.25*rho*S*b*K_rhat)   [kinematic turn gain]
  c1  = gk
  c2  = -m*g / (m + 0.25*rho*S*b*K_rhat)
  k1  = 0.5*rho*S*K_us
  k2  = 0.25*rho*S*b*K_rhat

All three fits are performed for each flight phase separately.

The course angle χ is obtained by projecting the EKF velocity states onto the
spherical unit vectors (zero differentiations). The identification target is
the RELATIVE course-turning rate χ̇_turn = −a·e_n/v_τ (with e_n = e_r × e_χ),
not the absolute chart rate d/dt[atan2(v_φ, v_β)]: the two differ by the
great-circle transport of the spherical basis (≈ φ̇·sinβ), which the turn-rate
laws deliberately exclude. The acceleration is a Savitzky–Golay derivative of
the EKF velocity (smoothing and differentiation in one step per contiguous
time segment).

Flights that declare a screened, healthy gyro (``gyro_yaw_rate_col``; 2019
only) also get that raw rate channel as a third cloud in the scatter. It is
plotted, never fitted, and it calibrates how much of the differentiated
heading rate is measurement: the two are the same physical quantity, so where
the differentiated cloud reaches further than the gyro's it is differentiating
a stepping angle channel, not recording a faster turn. On 2019 reel-out the
despiked derivative still runs 5x more samples past |chi_dot| = 1.8 rad/s than
the gyro (1198 vs 206) while correlating 0.94 with it; before despiking it
reaches 4.9 rad/s against the gyro's 2.2 (30.9 for the same-IMU
d/dt(kite_yaw_1), which regresses onto kite_yaw_rate_1 with slope 1.03).
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter
from pathlib import Path
from awes_ekf.setup.settings import load_config
from awes_ekf.load_data.read_data import read_results_from_hdf5
from awes_ekf.plotting.color_palette import get_color_list, set_plot_style_no_latex
from awetrim.identification.controls import (
    FLIGHT_STEERING_KCU_NORM_2019,
    FLIGHT_STEERING_KCU_NORM_2025,
    flight_steering_to_us,
)
import h5py

plt.close("all")
set_plot_style_no_latex()

# ── Configuration ─────────────────────────────────────────────────────────────
MASS = 50.0  # kite + lines mass [kg]
G = 9.81  # gravity [m/s²]
CUT = 10  # trim edges of the dataset
SMOOTH_WIN = 15  # moving-average window applied to yaw-rate signals [samples]
SAVGOL_WIN = 15  # Savitzky–Golay window for differentiating v_kite [samples, odd]
SAVGOL_ORDER = 3  # Savitzky–Golay polynomial order
# True (recommended): fits, RMSEs and the scatter use the raw (despiked-only)
# turn-rate target, so the identified coefficients are filter-independent —
# the SavGol/boxcar smoothing then only affects the plotted time series.
# False: fits use the smoothed signals; the identified gains then inherit the
# filter's peak attenuation (a few % low here, much worse for heavy filters).
FIT_ON_RAW = True
# Metric shown in the bar-chart panel (a): "R2" (fraction of turn-rate
# variance explained; comparable across phases, higher = better) or "RMSE"
# (absolute residual [rad/s]; can flatter phases with little turning, e.g.
# reel-in, where a small residual is easy while explaining almost nothing).
PLOT_METRIC = "R2"

# Asymmetry handling per law: "fit", "off", "fixed", "from_simple", "from_two_term"
ASYM_MODE_SIMPLE = "fit"
ASYM_MODE_TWO_TERM = "fit"
ASYM_MODE_FULL = "fit"  # warm-start full rational with two-term's fitted asymmetry
K_ASYM_FIXED = 0.0
# Turn-rate signal used for fitting: "yaw_dot" or "chi_dot".
TURN_RATE_SOURCE = "chi_dot"
# Which sensor index to use for the heading-rate signal (kite_yaw_rate_<x>
# when trusted, otherwise the kite_yaw_<x> angle). Sensor 0 gives the better
# angle in both flights (corr with chi_dot_turn 0.91/0.90 vs 0.86/0.73).
YAW_RATE_SENSOR_ID = 0
# False: ignore kite_yaw_rate_<id> and differentiate the kite_yaw_<id> angle
# instead (unwrapped, per-segment). Default False for cross-flight work: the
# gyro channels are not consistently usable, so differentiating the angle is
# the only signal comparable between datasets. Always screen a new dataset's
# rate channel before trusting it — regressed against chi_dot_turn on
# reel-out: 2019 gyro_1 slope 1.15 / corr 0.96 (healthy, and the best single
# channel there), 2019 gyro_0 corr 0.04 (dead), both 2025 gyros corr ≈ −0.20
# (broken or sign/frame-flipped). Set True only for a single-flight study on
# a screened, healthy rate channel.
YAW_RATE_SENSOR_RELIABLE = False
SCATTER_PHASE = 1  # phase shown in the representative scatter panel (reel-out)
PHASE_NAME = {1: "reel-out", 2: "rori", 3: "reel-in", 4: "riro"}
PHASES_TO_FIT = [1, 3]  # only reel-out (1) and reel-in (3)
PALETTE = get_color_list()

# ── Flights ───────────────────────────────────────────────────────────────────
# Each flight is analysed and plotted separately. `scatter_cycles` are the
# cycles the representative scatter always samples (on top of an even spread
# across the dataset).
CYCLES = range(2, 70)  # cycles to include
FLIGHTS = [
    {
        "label": "2019-10-08",
        "path": "results/LEI-V3-KITE/ekf/LEI-V3-Kite_2019-10-08.h5",
        "scatter_cycles": [62, 63, 64],
        # Standardised u_s (1.4*u_s = tape half-difference): the 2019 KCU's
        # kcu/100 moves only half the nominal tape, so divide by 200.
        "steering_kcu_norm": FLIGHT_STEERING_KCU_NORM_2019,
        # Healthy gyro channel, plotted RAW in the scatter as a reference
        # cloud (never fitted, see GYRO_REFERENCE_COL). IMU 1 is the screened
        # one on this flight; IMU 0's rate channel is dead (corr 0.04).
        "gyro_yaw_rate_col": "kite_yaw_rate_1",
    },
    {
        "label": "2025-10-09",
        "path": "results/LEI-V3-KITE/ekf/LEI-V3 Kite_2025-10-09.h5",
        "scatter_cycles": [2],
        # 2025 rig hypothesised already on the standardised scale (kcu/100).
        "steering_kcu_norm": FLIGHT_STEERING_KCU_NORM_2025,
        # No usable gyro on this flight (both rate channels corr ~ -0.20
        # against chi_dot_turn), so no reference cloud is drawn.
        "gyro_yaw_rate_col": None,
    },
]


def remove_outliers(sig, size=21, threshold=3.0):
    """Remove spike outliers by replacing values far from the local median."""
    med = median_filter(sig, size=size)
    diff = np.abs(sig - med)
    std = np.std(diff)
    if std > 0:
        mask = diff > threshold * std
        sig_clean = sig.copy()
        sig_clean[mask] = med[mask]
        return sig_clean
    return sig


# The dataset is stitched from selected cycles, so each contiguous time
# segment is differentiated separately to avoid smearing across the gaps.
def savgol_derivative(sig, time, window=SAVGOL_WIN, polyorder=SAVGOL_ORDER):
    """First time derivative of sig, per contiguous time segment.

    window=None skips the Savitzky–Golay smoothing and returns the plain
    (unsmoothed) finite-difference derivative of each segment.
    """
    dt = np.diff(time)
    dt_med = np.median(dt)
    breaks = np.flatnonzero(dt > 1.5 * dt_med) + 1
    out = np.full(sig.shape, np.nan)
    for s0, s1 in zip(np.r_[0, breaks], np.r_[breaks, sig.size]):
        if s1 - s0 < 2:
            continue
        if window is not None and s1 - s0 > window:
            out[s0:s1] = savgol_filter(
                sig[s0:s1], window, polyorder, deriv=1, delta=dt_med
            )
        else:
            out[s0:s1] = np.gradient(sig[s0:s1], time[s0:s1])
    return out


def heading_rate(flight_data, time):
    """Heading-rate signal (fit variant, display variant, source label).

    Uses the kite_yaw_rate_<id> gyro channel when it is declared trustworthy,
    otherwise differentiates the unwrapped kite_yaw_<id> angle.
    """
    yaw_rate_col = f"kite_yaw_rate_{YAW_RATE_SENSOR_ID}"
    if YAW_RATE_SENSOR_RELIABLE and yaw_rate_col in flight_data.columns:
        source, raw = yaw_rate_col, flight_data[yaw_rate_col].to_numpy()
    elif YAW_RATE_SENSOR_RELIABLE and "kite_yaw_rate" in flight_data.columns:
        source, raw = "kite_yaw_rate", flight_data["kite_yaw_rate"].to_numpy()
    else:
        yaw_angle_col = f"kite_yaw_{YAW_RATE_SENSOR_ID}"
        if yaw_angle_col not in flight_data.columns:
            yaw_angle_col = "kite_yaw_0"
        yaw_angle = flight_data[yaw_angle_col].to_numpy().astype(float)
        # Unit heuristic: the median per-sample increment ≈ typical yaw
        # rate × dt (~0.05 rad at 10 Hz vs ~3 deg). Range-based checks
        # misclassify unwrapped-radian logs (e.g. 2025 file spans ±10 rad).
        if np.nanmedian(np.abs(np.diff(yaw_angle))) > 1.0:  # degrees
            yaw_angle = np.deg2rad(yaw_angle)
        source = f"d/dt({yaw_angle_col})"
        raw = savgol_derivative(np.unwrap(yaw_angle), time, window=None)
    fit_variant = remove_outliers(raw)
    display = np.convolve(fit_variant, np.ones(SMOOTH_WIN) / SMOOTH_WIN, mode="same")
    return fit_variant, display, source


def gyro_reference(flight, flight_data):
    """Raw gyro heading rate of a flight that declares a healthy rate channel.

    Returned UNCONDITIONED -- no despiking, no smoothing, no unit heuristic --
    because its only job is to show what the differentiated yaw ANGLE would
    look like if differentiation added nothing. It is never fitted: on the 2019
    file, d/dt(kite_yaw_1) and kite_yaw_rate_1 are the same sensor and regress
    onto each other with slope 1.03, yet the derivative reaches 30.9 rad/s
    against the gyro's 2.2 -- the outliers are differentiation artefacts of a
    stepping angle channel, not flight events. ``None`` when the flight has no
    trustworthy rate channel (both 2025 gyros).
    """
    column = flight.get("gyro_yaw_rate_col")
    if not column or column not in flight_data.columns:
        return None, None
    return flight_data[column].to_numpy().astype(float), column


def prepare_flight(flight):
    """Load one flight and derive every signal the fits and plots need."""
    results, flight_data, _ = read_results_from_hdf5(flight["path"])
    results = results[CUT:-CUT].reset_index(drop=True)
    flight_data = flight_data[CUT:-CUT].reset_index(drop=True)
    keep = flight_data["cycle"].isin(CYCLES)
    results = results[keep].reset_index(drop=True)
    flight_data = flight_data[keep].reset_index(drop=True)

    time = flight_data["time"].to_numpy()
    us = np.asarray(
        flight_steering_to_us(
            flight_data["kcu_actual_steering"], norm=flight["steering_kcu_norm"]
        ),
        float,
    )
    va = results["kite_apparent_windspeed"].to_numpy()
    position = np.array([results[f"kite_position_{c}"] for c in "xyz"])
    v_kite = np.array([results[f"kite_velocity_{c}"] for c in "xyz"])

    r = np.linalg.norm(position, axis=0)
    r_hat = position / np.maximum(r, 1e-6)
    beta = np.arctan2(position[2], np.hypot(position[0], position[1]))  # elevation
    phi = np.arctan2(position[1], position[0])  # azimuth in wind window
    v_r = np.sum(r_hat * v_kite, axis=0)  # radial (tether) speed, signed

    # Course angle χ using spherical wind-frame coordinates. χ is measured
    # from the direction of increasing elevation β (toward zenith) within the
    # tangential plane τ. The EKF velocity states are projected onto the
    # spherical unit vectors, which are built algebraically from position — no
    # differentiation anywhere, so χ is as clean as the velocity estimate.
    #   e_β = (−sinβ·cosφ, −sinβ·sinφ, cosβ)   toward zenith (χ=0 reference)
    #   e_φ = (−sinφ, cosφ, 0)                 clockwise in window (χ=90°)
    sin_b, cos_b = np.sin(beta), np.cos(beta)
    sin_p, cos_p = np.sin(phi), np.cos(phi)
    e_beta = np.array([-sin_b * cos_p, -sin_b * sin_p, cos_b])
    e_phi = np.array([-sin_p, cos_p, np.zeros_like(phi)])
    v_beta = np.sum(v_kite * e_beta, axis=0)  # = r·dβ/dt, toward zenith
    v_phi = np.sum(v_kite * e_phi, axis=0)  # = r·cosβ·dφ/dt, clockwise
    v_tau = np.hypot(v_beta, v_phi)  # tangential speed (always >= 0)
    chi = np.unwrap(np.arctan2(v_phi, v_beta))  # course angle χ [rad]

    # Relative course-turning rate χ̇_turn from the lateral acceleration in the
    # tangential plane (see paper, "Experimental indication of quasi-steady
    # turning"):  e_n = e_r × e_chi,  χ̇_turn = −a·e_n/|v_τ|. Unlike
    # d/dt[atan2(v_φ, v_β)] this removes the great-circle transport of the
    # spherical basis and matches the χ̇_turn variable of the turn-rate laws.
    v_tau_vec = v_kite - v_r * r_hat
    e_n = np.cross(r_hat, v_tau_vec / np.maximum(v_tau, 1e-6), axis=0)

    def project(a_kite):
        return -np.sum(a_kite * e_n, axis=0) / np.maximum(v_tau, 1e-6)

    # Display variant: SavGol-differentiated velocity (smooth).
    chi_dot_display = project(
        np.vstack([savgol_derivative(row, time) for row in v_kite])
    )
    # Fit variant: plain finite-difference derivative, despiked only. Target-
    # side noise averages out of least squares without bias, whereas smoothing
    # attenuates the real peaks and drags the identified gains down.
    chi_dot_fit = remove_outliers(
        project(
            np.vstack([savgol_derivative(row, time, window=None) for row in v_kite])
        )
    )

    yaw_dot_fit, yaw_dot_display, yaw_source = heading_rate(flight_data, time)
    print(f"[{flight['label']}] heading rate from {yaw_source}")
    gyro_yaw_dot, gyro_source = gyro_reference(flight, flight_data)
    if gyro_yaw_dot is not None:
        print(
            f"[{flight['label']}] gyro reference cloud from {gyro_source} "
            "(raw, plotted only -- never fitted)"
        )

    # `meas` is the fit/RMSE/scatter target (raw when FIT_ON_RAW), `display`
    # the smoothed variant. `alt_*` is the other measurement chain, refit for
    # the paired bars as a cross-sensor check.
    if TURN_RATE_SOURCE == "chi_dot":
        primary_fit, primary_display = chi_dot_fit, chi_dot_display
        alt_fit, alt_display = yaw_dot_fit, yaw_dot_display
        primary_label, alt_label = "Course rate", "Heading rate"
    else:
        primary_fit, primary_display = yaw_dot_fit, yaw_dot_display
        alt_fit, alt_display = chi_dot_fit, chi_dot_display
        primary_label, alt_label = "Heading rate", "Course rate"

    return {
        "flight": flight,
        "flight_data": flight_data,
        "time": time,
        "us": us,
        "va": va,
        "r": r,
        "v_r": v_r,
        "beta": beta,
        "chi": chi,
        "v_tau": v_tau,
        "meas": primary_fit if FIT_ON_RAW else primary_display,
        "meas_display": primary_display,
        "alt": alt_fit if FIT_ON_RAW else alt_display,
        "alt_display": alt_display,
        "gyro_yaw_dot": gyro_yaw_dot,
        "gyro_source": gyro_source,
        "turn_rate_label": primary_label,
        "alt_label": alt_label,
    }


# ── Turn rate law functions ────────────────────────────────────────────────────


def fit_simple(chi_dot, us, va, asym_mode="fit", k_asym_fixed=0.0):
    """
    Eq. (41): chi_dot_b = gk * va * (us - k_asymmetry)
    Reformulated as linear: chi_dot_b = c1*(va*us) + c2*(va)
    where c1 = gk and c2 = -gk*k_asymmetry, so k_asymmetry = -c2/c1
    Returns: (gk, k_asymmetry), chi_dot_est
    """
    term1 = va * us
    if asym_mode == "off":
        A = term1.reshape(-1, 1)
    elif asym_mode == "fixed":
        A = (va * (us - k_asym_fixed)).reshape(-1, 1)
    else:
        term2 = va
        A = np.vstack([term1, term2]).T
    valid = np.isfinite(A).all(axis=1) & np.isfinite(chi_dot)
    coeffs = calculate_weighted_squares_1d(chi_dot[valid], A[valid])
    if asym_mode == "off":
        gk = coeffs[0]
        k_asymmetry = 0.0
    elif asym_mode == "fixed":
        gk = coeffs[0]
        k_asymmetry = k_asym_fixed
    else:
        c1, c2 = coeffs[0], coeffs[1]
        gk = c1
        k_asymmetry = -c2 / c1 if abs(c1) > 1e-10 else 0.0

    return (gk, k_asymmetry), A @ coeffs


def fit_two_term(chi_dot, us, va, chi, beta, asym_mode="fit", k_asym_fixed=0.0):
    """
    Eq. (40): chi_dot_b = c1*(va*(us - k_asymmetry)) + c2*(sin(chi)*cos(beta)/va)
    Reformulated as linear: chi_dot_b = coeff1*(va*us) + coeff2*(va) + coeff3*(sin(chi)*cos(beta)/va)
    where coeff1 = c1, coeff2 = -c1*k_asymmetry, coeff3 = c2
    Returns: (c1, c2, k_asymmetry), chi_dot_est
    gk = c1,  and from c2 = -m*g/(m + 0.25*rho*S*b*K_rhat)
    """
    term1 = va * us
    term_gravity = np.sin(chi) * np.cos(beta) / np.maximum(va, 1e-6)
    if asym_mode == "off":
        A = np.vstack([term1, term_gravity]).T
    elif asym_mode == "fixed":
        term1_fixed = va * (us - k_asym_fixed)
        A = np.vstack([term1_fixed, term_gravity]).T
    else:
        term2 = va
        A = np.vstack([term1, term2, term_gravity]).T
    valid = np.isfinite(A).all(axis=1) & np.isfinite(chi_dot)
    coeffs = calculate_weighted_squares_1d(chi_dot[valid], A[valid])
    if asym_mode == "off":
        c1, c2 = coeffs[0], coeffs[1]
        k_asymmetry = 0.0
    elif asym_mode == "fixed":
        c1, c2 = coeffs[0], coeffs[1]
        k_asymmetry = k_asym_fixed
    else:
        coeff1, coeff2, coeff3 = coeffs[0], coeffs[1], coeffs[2]
        c1 = coeff1
        c2 = coeff3
        k_asymmetry = -coeff2 / coeff1 if abs(coeff1) > 1e-10 else 0.0
    return (c1, c2, k_asymmetry), A @ coeffs


def fit_full_rational(
    chi_dot,
    us,
    va,
    r,
    v_r,
    v_tau,
    chi,
    beta,
    mass=MASS,
    g=G,
    x0=None,
    asym_mode="fit",
    k_asym_fixed=0.0,
    gravity_mode="fit",
):
    """
    Eq. (38): chi_dot_b = -(k1*va^2*(us - k_asymmetry) + m*g*sin(chi)*cos(beta))
                                                    / (k4*m*v_tau + k2*va)
    Nonlinear least-squares fit for (k1, k2, k3, k_asymmetry).
      k1  ~ 0.5*rho*S*K_us        (steering aerodynamic gain)
      k2  ~ 0.25*rho*S*b*K_rhat   (yaw-damping aerodynamic gain)
            k3  ~ gravity numerator gain
            k4  ~ radial-rate denominator gain
      k_asymmetry ~ asymmetry in steering input
    gravity_mode: "fit" keeps the gravity term coefficient k3 in the numerator;
        "off" fixes k3 = 0 and fits a gravity-free model.
    x0: initial guess for the physical coefficients.
    Returns: (k1, k2, k3, k_asymmetry), chi_dot_est
    """
    if x0 is None:
        if gravity_mode == "fit":
            x0 = [1, 8, 8, 0] if asym_mode == "fit" else [1, 8, 8, 8]
        else:
            x0 = [1, 8, 8, 0] if asym_mode == "fit" else [1, 8, 8]

    with_asym = asym_mode == "fit"
    with_gravity = gravity_mode == "fit"

    def to_internal_x0(x_phys):
        x_arr = np.asarray(x_phys, dtype=float).reshape(-1)

        def get(index, default=0.0):
            return x_arr[index] if index < x_arr.size else default

        k1_0 = get(0, 1.0)
        k2_0 = get(1, 8.0)
        k3_0 = get(2, 0.0) if with_gravity else 0.0

        if with_gravity and with_asym:
            k4_0 = get(3, k2_0)
            k_asym_0 = get(4, 0.0)
        elif with_gravity and not with_asym:
            k4_0 = get(3, k2_0)
            k_asym_0 = 0.0
        elif not with_gravity and with_asym:
            if x_arr.size >= 5:
                k4_0 = get(3, k2_0)
                k_asym_0 = get(4, 0.0)
            else:
                k4_0 = get(2, k2_0)
                k_asym_0 = get(3, 0.0)
        else:
            k4_0 = get(2, k2_0)
            k_asym_0 = 0.0

        if with_gravity and with_asym:
            return np.array([k1_0, k2_0, k3_0, k4_0, k_asym_0], dtype=float)
        if with_gravity and not with_asym:
            return np.array([k1_0, k2_0, k3_0, k4_0], dtype=float)
        if not with_gravity and with_asym:
            return np.array([k1_0, k2_0, k4_0, k_asym_0], dtype=float)
        return np.array([k1_0, k2_0, k4_0], dtype=float)

    def predict(k1, k2, k3, k4, k_asymmetry):
        gravity = k3 * mass * g * np.sin(chi) * np.cos(beta) if with_gravity else 0.0
        num = k1 * va**2 * (us - k_asymmetry) + gravity
        radial_rate = v_tau * mass  # * v_r / np.maximum(r, 1e-6)
        den = np.maximum(k4 * radial_rate + k2 * va, 1e-6)
        return -num / den

    def residuals(x):
        if with_gravity and with_asym:
            k1, k2, k3, k4, k_asymmetry = x
        elif with_gravity and not with_asym:
            k1, k2, k3, k4 = x
            k_asymmetry = k_asym_fixed
        elif not with_gravity and with_asym:
            k1, k2, k4, k_asymmetry = x
            k3 = 0.0
        else:
            k1, k2, k4 = x
            k3 = 0.0
            k_asymmetry = k_asym_fixed
        return predict(k1, k2, k3, k4, k_asymmetry) - chi_dot

    valid = (
        np.isfinite(chi_dot)
        & np.isfinite(va)
        & np.isfinite(v_tau)
        & np.isfinite(us)
        & np.isfinite(chi)
        & np.isfinite(beta)
    )
    if with_gravity and with_asym:
        bounds = ([-1e2, -1e2, -1e2, -1e2, -0.1], [1e2, 1e2, 1e2, 1e2, 0.1])
    elif with_gravity and not with_asym:
        bounds = ([-1e2, -1e2, -1e2, -1e2], [1e2, 1e2, 1e2, 1e2])
    elif not with_gravity and with_asym:
        bounds = ([-1e2, -1e2, -1e2, -0.1], [1e2, 1e2, 1e2, 0.1])
    else:
        bounds = ([-1e2, -1e2, -1e2], [1e2, 1e2, 1e2])
    res = least_squares(
        lambda x: residuals(x)[valid],
        x0=to_internal_x0(x0),
        bounds=bounds,
    )
    if with_gravity and with_asym:
        k1, k2, k3, k4, k_asymmetry = res.x
    elif with_gravity and not with_asym:
        k1, k2, k3, k4 = res.x
        k_asymmetry = k_asym_fixed
    elif not with_gravity and with_asym:
        k1, k2, k4, k_asymmetry = res.x
        k3 = 0.0
    else:
        k1, k2, k4 = res.x
        k3 = 0.0
        k_asymmetry = k_asym_fixed
    return (k1, k2, k3, k4, k_asymmetry), predict(k1, k2, k3, k4, k_asymmetry)


def fit_full_rational_course_term(
    chi_dot,
    us,
    va,
    r,
    v_r,
    v_tau,
    chi,
    beta,
    mass=MASS,
    g=G,
    x0=None,
    asym_mode="fit",
    k_asym_fixed=0.0,
    gravity_mode="fit",
):
    """
        Eq. (39): chi_dot_b = -(k1*va^2*(us - k_asymmetry) + m*g*sin(chi)*cos(beta)
                        + k5*cos(chi)*cos(beta) + k6*sin(beta))
                        / (k4*m*v_tau + k2*va)
        Nonlinear least-squares fit for (k1, k2, k3, k4, k5, k6, k_asymmetry).
      k1  ~ 0.5*rho*S*K_us        (steering aerodynamic gain)
      k2  ~ 0.25*rho*S*b*K_rhat   (yaw-damping aerodynamic gain)
            k3  ~ gravity numerator gain
            k4  ~ radial-rate denominator gain
            k5  ~ course-term numerator gain
            k6  ~ elevation-term numerator gain
      k_asymmetry ~ asymmetry in steering input
    gravity_mode: "fit" keeps the gravity term coefficient k3 in the numerator;
        "off" fixes k3 = 0 and fits a gravity-free model.
    x0: initial guess for the physical coefficients.
    Returns: (k1, k2, k3, k4, k5, k6, k_asymmetry), chi_dot_est
    """
    if x0 is None:
        if gravity_mode == "fit":
            x0 = [1, 8, 8, 1, 0, 0, 0] if asym_mode == "fit" else [1, 8, 8, 1, 0, 0]
        else:
            x0 = [1, 8, 1, 0, 0, 0] if asym_mode == "fit" else [1, 8, 1, 0, 0]

    with_asym = asym_mode == "fit"
    with_gravity = gravity_mode == "fit"

    def to_internal_x0(x_phys):
        x_arr = np.asarray(x_phys, dtype=float).reshape(-1)

        def get(index, default=0.0):
            return x_arr[index] if index < x_arr.size else default

        k1_0 = get(0, 1.0)
        k2_0 = get(1, 8.0)
        k3_0 = get(2, 0.0) if with_gravity else 0.0

        if with_gravity and with_asym:
            k4_0 = get(3, k2_0)
            k5_0 = get(4, 0.0)
            k6_0 = get(5, 0.0)
            k_asym_0 = get(6, 0.0)
        elif with_gravity and not with_asym:
            k4_0 = get(3, k2_0)
            k5_0 = get(4, 0.0)
            k6_0 = get(5, 0.0)
            k_asym_0 = 0.0
        elif not with_gravity and with_asym:
            if x_arr.size >= 7:
                k4_0 = get(3, k2_0)
                k5_0 = get(4, 0.0)
                k6_0 = get(5, 0.0)
                k_asym_0 = get(6, 0.0)
            else:
                k4_0 = get(2, k2_0)
                k5_0 = get(3, 0.0)
                k6_0 = get(4, 0.0)
                k_asym_0 = get(5, 0.0)
        else:
            k4_0 = get(2, k2_0)
            k5_0 = get(3, 0.0)
            k6_0 = get(4, 0.0)
            k_asym_0 = 0.0

        if with_gravity and with_asym:
            return np.array([k1_0, k2_0, k3_0, k4_0, k5_0, k6_0, k_asym_0], dtype=float)
        if with_gravity and not with_asym:
            return np.array([k1_0, k2_0, k3_0, k4_0, k5_0, k6_0], dtype=float)
        if not with_gravity and with_asym:
            return np.array([k1_0, k2_0, k4_0, k5_0, k6_0, k_asym_0], dtype=float)
        return np.array([k1_0, k2_0, k4_0, k5_0, k6_0], dtype=float)

    def predict(k1, k2, k3, k4, k5, k6, k_asymmetry):
        gravity = k3 * mass * g * np.sin(chi) * np.cos(beta) if with_gravity else 0.0
        course_term = k5 * np.cos(chi) * np.cos(beta)
        elev_term = k6 * np.sin(beta)
        num = k1 * va**2 * (us - k_asymmetry) + gravity + course_term + elev_term
        radial_rate = v_tau * mass  # * v_r / np.maximum(r, 1e-6)
        den = np.maximum(k4 * radial_rate + k2 * va, 1e-6)
        return -num / den

    def residuals(x):
        if with_gravity and with_asym:
            k1, k2, k3, k4, k5, k6, k_asymmetry = x
        elif with_gravity and not with_asym:
            k1, k2, k3, k4, k5, k6 = x
            k_asymmetry = k_asym_fixed
        elif not with_gravity and with_asym:
            k1, k2, k4, k5, k6, k_asymmetry = x
            k3 = 0.0
        else:
            k1, k2, k4, k5, k6 = x
            k3 = 0.0
            k_asymmetry = k_asym_fixed
        return predict(k1, k2, k3, k4, k5, k6, k_asymmetry) - chi_dot

    valid = (
        np.isfinite(chi_dot)
        & np.isfinite(va)
        & np.isfinite(v_tau)
        & np.isfinite(us)
        & np.isfinite(chi)
        & np.isfinite(beta)
    )
    if with_gravity and with_asym:
        bounds = (
            [-1e2, -1e2, -1e2, -1e2, -1e2, -1e2, -0.1],
            [1e2, 1e2, 1e2, 1e2, 1e2, 1e2, 0.1],
        )
    elif with_gravity and not with_asym:
        bounds = (
            [-1e2, -1e2, -1e2, -1e2, -1e2, -1e2],
            [1e2, 1e2, 1e2, 1e2, 1e2, 1e2],
        )
    elif not with_gravity and with_asym:
        bounds = (
            [-1e2, -1e2, -1e2, -1e2, -1e2, -0.1],
            [1e2, 1e2, 1e2, 1e2, 1e2, 0.1],
        )
    else:
        bounds = ([-1e2, -1e2, -1e2, -1e2, -1e2], [1e2, 1e2, 1e2, 1e2, 1e2])
    res = least_squares(
        lambda x: residuals(x)[valid],
        x0=to_internal_x0(x0),
        bounds=bounds,
    )
    if with_gravity and with_asym:
        k1, k2, k3, k4, k5, k6, k_asymmetry = res.x
    elif with_gravity and not with_asym:
        k1, k2, k3, k4, k5, k6 = res.x
        k_asymmetry = k_asym_fixed
    elif not with_gravity and with_asym:
        k1, k2, k4, k5, k6, k_asymmetry = res.x
        k3 = 0.0
    else:
        k1, k2, k4, k5, k6 = res.x
        k3 = 0.0
        k_asymmetry = k_asym_fixed
    return (k1, k2, k3, k4, k5, k6, k_asymmetry), predict(
        k1, k2, k3, k4, k5, k6, k_asymmetry
    )


def calculate_weighted_squares_1d(y, A):
    """Unweighted least squares (normal equations via lstsq)."""
    return np.linalg.lstsq(A, y, rcond=None)[0]


def rmse(y_true, y_pred):
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    return np.sqrt(np.mean((y_true[valid] - y_pred[valid]) ** 2))


def r_squared(y_true, y_pred):
    """Coefficient of determination, 1 − SS_res/SS_tot (a.k.a. VAF).

    Defined for any predictor, not just linear fits; can go negative for a
    model worse than the target's mean. With a raw (unsmoothed) target the
    attainable maximum is < 1 because the noise floor is unexplainable by
    construction — comparisons between models on the same target remain
    like-for-like.
    """
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    ss_res = np.sum((y_true[valid] - y_pred[valid]) ** 2)
    ss_tot = np.sum((y_true[valid] - np.mean(y_true[valid])) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan


# ── Per-phase fitting ─────────────────────────────────────────────────────────

TABLE_HEADER = (
    f"{'Phase':<22} {'Model':<16} {'k1/gk/c1':>10} {'k2/c2':>10} {'k3':>10} "
    f"{'k4':>10} {'k5':>10} {'k6':>10} {'k_asym':>10} {'RMSE':>8} {'R2':>7}"
)


def fit_one_phase(ctx, phase, phase_col, x0_full):
    """Fit every law for one flight phase.

    Returns (result_dict or None, updated warm-start x0_full).
    """
    flight_data = ctx["flight_data"]
    mask = (flight_data[phase_col] == phase).to_numpy()
    n = mask.sum()
    if n < 50:
        return None, x0_full

    fd = ctx["meas"][mask]
    yd = ctx["alt"][mask] if ctx["alt"] is not None else None
    gyro = ctx["gyro_yaw_dot"][mask] if ctx.get("gyro_yaw_dot") is not None else None
    u = ctx["us"][mask]
    v = ctx["va"][mask]
    vt = ctx["v_tau"][mask]
    c = ctx["chi"][mask]
    b = ctx["beta"][mask]
    rr = ctx["r"][mask]
    vr = ctx["v_r"][mask]

    # Eq. (41) simple
    simple_mode = (
        ASYM_MODE_SIMPLE if ASYM_MODE_SIMPLE in ["fit", "off", "fixed"] else "fit"
    )
    (gk, k_asym_41), est_simple = fit_simple(
        fd,
        u,
        v,
        asym_mode=simple_mode,
        k_asym_fixed=K_ASYM_FIXED,
    )
    r_simple = rmse(fd, est_simple)

    # Eq. (41) simple symmetric reference (no asymmetry term)
    (gk_sym, _), est_simple_sym = fit_simple(
        fd,
        u,
        v,
        asym_mode="off",
        k_asym_fixed=0.0,
    )
    r_simple_sym = rmse(fd, est_simple_sym)

    # Eq. (40) two-term
    if ASYM_MODE_TWO_TERM == "from_simple":
        two_mode = "fixed"
        two_k_asym = k_asym_41
    elif ASYM_MODE_TWO_TERM == "off":
        two_mode = "off"
        two_k_asym = 0.0
    elif ASYM_MODE_TWO_TERM == "fixed":
        two_mode = "fixed"
        two_k_asym = K_ASYM_FIXED
    else:
        two_mode = "fit"
        two_k_asym = K_ASYM_FIXED
    (c1, c2, k_asym_40), est_two = fit_two_term(
        fd,
        u,
        v,
        c,
        b,
        asym_mode=two_mode,
        k_asym_fixed=two_k_asym,
    )
    r_two = rmse(fd, est_two)

    # Eq. (38) gravity-free pass used to warm-start the full model
    if ASYM_MODE_FULL == "from_simple":
        full_ng_mode = "fixed"
        full_ng_k_asym = k_asym_41
    elif ASYM_MODE_FULL == "from_two_term":
        full_ng_mode = "fixed"
        full_ng_k_asym = k_asym_40
    elif ASYM_MODE_FULL == "off":
        full_ng_mode = "fixed"
        full_ng_k_asym = 0.0
    elif ASYM_MODE_FULL == "fixed":
        full_ng_mode = "fixed"
        full_ng_k_asym = K_ASYM_FIXED
    else:
        full_ng_mode = "fit"
        full_ng_k_asym = K_ASYM_FIXED

    (k1_ng, k2_ng, k3_ng, k4_ng, k_asym_ng), est_full_ng = fit_full_rational(
        fd,
        u,
        v,
        rr,
        vr,
        vt,
        c,
        b,
        x0=x0_full,
        asym_mode=full_ng_mode,
        k_asym_fixed=full_ng_k_asym,
        gravity_mode="off",
    )

    x0_full = [k1_ng, k2_ng, 5, k4_ng, k_asym_ng]

    # Eq. (38) full rational — warm-started from the gravity-free solution
    if ASYM_MODE_FULL == "from_simple":
        full_mode = "fixed"
        full_k_asym = k_asym_41
    elif ASYM_MODE_FULL == "from_two_term":
        full_mode = "fixed"
        full_k_asym = k_asym_40
    elif ASYM_MODE_FULL == "off":
        full_mode = "fixed"
        full_k_asym = 0.0
    elif ASYM_MODE_FULL == "fixed":
        full_mode = "fixed"
        full_k_asym = K_ASYM_FIXED
    else:
        full_mode = "fit"
        full_k_asym = K_ASYM_FIXED

    (k1, k2, k3, k4, k_asym_38), est_full = fit_full_rational(
        fd,
        u,
        v,
        rr,
        vr,
        vt,
        c,
        b,
        x0=x0_full,
        asym_mode=full_mode,
        k_asym_fixed=full_k_asym,
        gravity_mode="fit",
    )
    norm_coeffs = np.sqrt(k1**2 + k2**2 + k3**2)
    if norm_coeffs < 1e-2:
        k1 = k1 / norm_coeffs
        k2 = k2 / norm_coeffs
        k3 = k3 / norm_coeffs
    r_full = rmse(fd, est_full)

    # Eq. (39) full rational + course term — warm-started from full model
    (k1_ct, k2_ct, k3_ct, k4_ct, k5_ct, k6_ct, k_asym_39), est_full_ct = (
        fit_full_rational_course_term(
            fd,
            u,
            v,
            rr,
            vr,
            vt,
            c,
            b,
            x0=[k1, k2, k3, k4, 0.0, 0.0, k_asym_38],
            asym_mode=full_mode,
            k_asym_fixed=full_k_asym,
            gravity_mode="fit",
        )
    )
    r_full_ct = rmse(fd, est_full_ct)

    # Eq. (39) refit against yaw-rate measurement (for bar-chart comparison).
    full_yaw_block = None
    if yd is not None:
        (
            (k1_yw, k2_yw, k3_yw, k4_yw, k5_yw, k6_yw, k_asym_yw),
            est_full_yaw,
        ) = fit_full_rational_course_term(
            yd,
            u,
            v,
            rr,
            vr,
            vt,
            c,
            b,
            x0=[k1_ct, k2_ct, k3_ct, k4_ct, k5_ct, k6_ct, k_asym_39],
            asym_mode=full_mode,
            k_asym_fixed=full_k_asym,
            gravity_mode="fit",
        )
        full_yaw_block = {
            "k1": k1_yw,
            "k2": k2_yw,
            "k3": k3_yw,
            "k4": k4_yw,
            "k5": k5_yw,
            "k6": k6_yw,
            "k_asymmetry": k_asym_yw,
            "RMSE": rmse(yd, est_full_yaw),
            "R2": r_squared(yd, est_full_yaw),
            "est": est_full_yaw,
        }

    # Every law refit against the yaw-rate signal: paired bars in panel (a)
    # compare each formula on both measurement chains (EKF-velocity χ̇_turn vs
    # gyro heading rate) — a cross-sensor robustness check of the law
    # structure. Coefficients are NOT interchangeable between the two targets
    # (heading and course dynamics differ by the sideslip/crab transient).
    yaw_fits = None
    if yd is not None:
        (gk_y, k_asym_y), est_simple_y = fit_simple(
            yd, u, v, asym_mode=simple_mode, k_asym_fixed=K_ASYM_FIXED
        )
        (gk_sym_y, _), est_simple_sym_y = fit_simple(
            yd, u, v, asym_mode="off", k_asym_fixed=0.0
        )
        (c1_y, c2_y, k_asym_40_y), est_two_y = fit_two_term(
            yd, u, v, c, b, asym_mode=two_mode, k_asym_fixed=two_k_asym
        )
        (k1_y, k2_y, k3_y, k4_y, k_asym_38_y), est_full_y = fit_full_rational(
            yd,
            u,
            v,
            rr,
            vr,
            vt,
            c,
            b,
            x0=[k1, k2, k3, k4, k_asym_38],
            asym_mode=full_mode,
            k_asym_fixed=full_k_asym,
            gravity_mode="fit",
        )
        yaw_fits = {
            "simple": {
                "gk": gk_y,
                "k_asymmetry": k_asym_y,
                "RMSE": rmse(yd, est_simple_y),
                "R2": r_squared(yd, est_simple_y),
            },
            "simple_symmetric": {
                "gk": gk_sym_y,
                "RMSE": rmse(yd, est_simple_sym_y),
                "R2": r_squared(yd, est_simple_sym_y),
            },
            "two_term": {
                "c1": c1_y,
                "c2": c2_y,
                "RMSE": rmse(yd, est_two_y),
                "R2": r_squared(yd, est_two_y),
            },
            "full": {
                "RMSE": rmse(yd, est_full_y),
                "R2": r_squared(yd, est_full_y),
            },
            "full_course_term": {
                "RMSE": full_yaw_block["RMSE"],
                "R2": full_yaw_block["R2"],
            },
        }

    result = {
        "yaw_fits": yaw_fits,
        "yaw_dot_meas": yd,
        # Raw gyro rate for this phase: scatter reference only, no fit uses it.
        "gyro_yaw_dot": gyro,
        "simple": {
            "gk": gk,
            "k_asymmetry": k_asym_41,
            "RMSE": r_simple,
            "R2": r_squared(fd, est_simple),
            "est": est_simple,
            "meas": fd,
        },
        "simple_symmetric": {
            "gk": gk_sym,
            "k_asymmetry": 0.0,
            "RMSE": r_simple_sym,
            "R2": r_squared(fd, est_simple_sym),
            "est": est_simple_sym,
        },
        "two_term": {
            "c1": c1,
            "c2": c2,
            "k_asymmetry": k_asym_40,
            "RMSE": r_two,
            "R2": r_squared(fd, est_two),
            "est": est_two,
        },
        "full": {
            "k1": k1,
            "k2": k2,
            "k3": k3,
            "k4": k4,
            "k_asymmetry": k_asym_38,
            "RMSE": r_full,
            "R2": r_squared(fd, est_full),
            "est": est_full,
        },
        "full_course_term": {
            "k1": k1_ct,
            "k2": k2_ct,
            "k3": k3_ct,
            "k4": k4_ct,
            "k5": k5_ct,
            "k6": k6_ct,
            "k_asymmetry": k_asym_39,
            "RMSE": r_full_ct,
            "R2": r_squared(fd, est_full_ct),
            "est": est_full_ct,
        },
        "full_course_term_yaw": full_yaw_block,
        "full_no_gravity": {
            "k1": k1_ng,
            "k2": k2_ng,
            "k3": k3_ng,
            "k4": k4_ng,
            "k_asymmetry": k_asym_ng,
            "RMSE": rmse(fd, est_full_ng),
            "R2": r_squared(fd, est_full_ng),
            "est": est_full_ng,
        },
        "signals": {"u": u, "v": v, "vt": vt, "c": c, "b": b},
        "us_va": u * v,
        "time": ctx["time"][mask],
        "n": n,
    }

    print(
        f"{str(phase):<22} {'Eq.(41)':<12} {gk:>10.4f} {'—':>10} {'—':>10} {'—':>10} {'—':>10} {'—':>10} {k_asym_41:>10.4f} {r_simple:>8.4f} {r_squared(fd, est_simple):>7.3f}"
    )
    print(
        f"{'':22} {'Eq.(41) sym':<12} {gk_sym:>10.4f} {'—':>10} {'—':>10} {'—':>10} {'—':>10} {'—':>10} {0.0:>10.4f} {r_simple_sym:>8.4f} {r_squared(fd, est_simple_sym):>7.3f}"
    )
    print(
        f"{'':22} {'two_fit':<16} {c1:>10.4f} {c2:>10.4f} {'—':>10} {'—':>10} {'—':>10} {'—':>10} {k_asym_40:>10.4f} {r_two:>8.4f} {r_squared(fd, est_two):>7.3f}"
    )
    print(
        f"{'':22} {'full':<16} {k1:>10.4f} {k2:>10.4f} {k3:>10.4f} {k4:>10.4f} {'—':>10} {'—':>10} {k_asym_38:>10.4f} {r_full:>8.4f} {r_squared(fd, est_full):>7.3f}"
    )
    print(
        f"{'':22} {'Eq.(39)':<12} {k1_ct:>10.4f} {k2_ct:>10.4f} {k3_ct:>10.4f} {k4_ct:>10.4f} {k5_ct:>10.4f} {k6_ct:>10.4f} {k_asym_39:>10.4f} {r_full_ct:>8.4f} {r_squared(fd, est_full_ct):>7.3f}"
    )
    print(
        f"{'':22} {'Eq.(38) NG':<12} {k1_ng:>10.4f} {k2_ng:>10.4f} {k3_ng:>10.4f} {k4_ng:>10.4f} {'—':>10} {'—':>10} {k_asym_ng:>10.4f} {rmse(fd, est_full_ng):>8.4f} {r_squared(fd, est_full_ng):>7.3f}"
    )
    print()
    return result, x0_full


def fit_flight(ctx):
    """Fit every law on every requested phase of one flight."""
    flight_data = ctx["flight_data"]
    phase_col = (
        "flight_phase_index" if "flight_phase_index" in flight_data.columns else "cycle"
    )
    phases = [p for p in flight_data[phase_col].unique() if p in PHASES_TO_FIT]

    print(f"\n[{ctx['flight']['label']}] fitting turn rate laws per {phase_col}")
    print("=" * 118)
    print(TABLE_HEADER)
    print("-" * 118)

    phase_results = {}
    x0_full = None  # warm-start, carried across phases
    for phase in sorted(phases, key=str):
        result, x0_full = fit_one_phase(ctx, phase, phase_col, x0_full)
        if result is not None:
            phase_results[phase] = result
    print("=" * 118)
    return phase_results, phase_col


# ── Plots ─────────────────────────────────────────────────────────────────────


def phase_label(p):
    try:
        return PHASE_NAME.get(int(float(p)), str(p))
    except (ValueError, TypeError):
        return str(p)


def make_figure(ctx, phase_results, phase_col):
    """Two panels for one flight: (a) metric bars, (b) representative scatter."""
    flight = ctx["flight"]
    if not phase_results:
        print(f"[{flight['label']}] no phases with enough data to plot.")
        return

    flight_data = ctx["flight_data"]
    turn_rate_label = ctx["turn_rate_label"]
    alt_label = ctx["alt_label"]

    fig, (ax_bar, ax_scatter) = plt.subplots(1, 2, figsize=(12, 4.8))

    # --- (a) metric per phase, per law ---
    phase_keys = sorted(
        phase_results,
        key=lambda p: int(float(p)) if str(p).replace(".", "", 1).isdigit() else str(p),
    )
    x = np.arange(len(phase_keys))
    sym_linear_color = PALETTE[1]
    course_color = PALETTE[2]
    alt_color = PALETTE[4]

    def metric(model_key, source="primary"):
        if source == "primary":
            return [phase_results[p][model_key][PLOT_METRIC] for p in phase_keys]
        return [
            phase_results[p]["yaw_fits"][model_key][PLOT_METRIC] for p in phase_keys
        ]

    has_course_term = all("full_course_term" in phase_results[p] for p in phase_keys)
    bar_items = [
        ("Linear", "simple", PALETTE[6]),
        ("Sym. linear", "simple_symmetric", sym_linear_color),
        ("Weight-corr.", "two_term", PALETTE[5]),
        ("Reduced full", "full", PALETTE[7] if len(PALETTE) > 7 else PALETTE[4]),
    ]
    if has_course_term:
        bar_items.append(("Full", "full_course_term", course_color))

    # Solid bar = primary target, hatched twin = the other measurement chain —
    # a cross-sensor robustness check of the law structure.
    has_yaw_fits = all(phase_results[p].get("yaw_fits") is not None for p in phase_keys)
    width = 0.8 / max(len(bar_items), 1)
    for idx, (label, model_key, color) in enumerate(bar_items):
        offset = (idx - (len(bar_items) - 1) / 2) * width
        if has_yaw_fits:
            ax_bar.bar(
                x + offset - width / 4,
                metric(model_key),
                width / 2,
                label=label,
                color=color,
            )
            ax_bar.bar(
                x + offset + width / 4,
                metric(model_key, "yaw"),
                width / 2,
                color=color,
                hatch="///",
                edgecolor="white",
                linewidth=0,
            )
        else:
            ax_bar.bar(x + offset, metric(model_key), width, label=label, color=color)
    if has_yaw_fits:
        # Two neutral swatches showing the fill styles, so the legend explains
        # the encoding by example instead of naming the pattern.
        ax_bar.bar(
            0, 0, width=0, color="0.8", label=f"Fit to {turn_rate_label.lower()}"
        )
        ax_bar.bar(
            0,
            0,
            width=0,
            color="0.8",
            hatch="///",
            edgecolor="white",
            label=f"Fit to {alt_label.lower()}",
        )

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(
        [phase_label(p) for p in phase_keys], rotation=30, ha="center"
    )
    if PLOT_METRIC == "R2":
        ax_bar.set_ylabel(r"$R^2$  [—]")
        ax_bar.set_ylim(0, 1)
    else:
        ax_bar.set_ylabel("RMSE [rad/s]")
    ax_bar.grid(True, axis="y")

    # --- (b) representative scatter: raw signals + the two linear laws ---
    phase_target = next(
        (
            p
            for p in phase_keys
            if str(p).replace(".", "", 1).isdigit() and int(float(p)) == SCATTER_PHASE
        ),
        None,
    )
    if phase_target is not None:
        pr = phase_results[phase_target]
        phase_mask = (flight_data[phase_col] == phase_target).to_numpy()
        cycles_in_phase = flight_data.loc[phase_mask, "cycle"].to_numpy()
        unique_cycles = np.unique(cycles_in_phase)
        n_scatter_cycles = 5
        if len(unique_cycles) > n_scatter_cycles:
            sel = np.linspace(0, len(unique_cycles) - 1, n_scatter_cycles).astype(int)
            selected = np.union1d(
                unique_cycles[sel], np.array(flight["scatter_cycles"])
            )
        else:
            selected = unique_cycles
        sub = np.isin(cycles_in_phase, selected)

        xdata = pr["us_va"][sub]
        ax_scatter.scatter(
            xdata,
            pr["simple"]["meas"][sub],
            s=15,
            alpha=0.5,
            color=PALETTE[0],
            label=turn_rate_label,
        )
        if pr.get("yaw_dot_meas") is not None:
            ax_scatter.scatter(
                xdata,
                pr["yaw_dot_meas"][sub],
                s=15,
                alpha=0.5,
                color=PALETTE[3],
                label=alt_label,
            )
        # Raw gyro rate, same quantity as the differentiated yaw angle but
        # measured directly. The differentiated cloud is already despiked, so
        # the gross artefacts never reach the figure; what is left to see is
        # its tail -- it reaches beyond the gyro's at both ends of the turn,
        # which is differentiation noise, not a faster turn. Plotted only.
        if pr.get("gyro_yaw_dot") is not None:
            ax_scatter.scatter(
                xdata,
                pr["gyro_yaw_dot"][sub],
                s=12,
                alpha=0.35,
                marker="x",
                linewidths=0.7,
                color=PALETTE[6],
                label=f"Heading rate (gyro, {ctx['gyro_source']})",
            )
        # Linear law (Eq. 41) for both targets. As a function of x = u_s·v_a the
        # asymmetry term is −gk·k_asym·v_a; the line uses the phase-mean v_a.
        x_line = np.linspace(np.nanmin(xdata), np.nanmax(xdata), 200)
        va_mean = np.nanmean(pr["signals"]["v"][sub])
        ax_scatter.plot(
            x_line,
            pr["simple"]["gk"] * (x_line - pr["simple"]["k_asymmetry"] * va_mean),
            color=sym_linear_color,
            lw=2.0,
            label=f"Linear ({turn_rate_label.lower()})",
        )
        if pr.get("yaw_fits") is not None:
            yf = pr["yaw_fits"]["simple"]
            ax_scatter.plot(
                x_line,
                yf["gk"] * (x_line - yf.get("k_asymmetry", 0.0) * va_mean),
                color=alt_color,
                lw=2.0,
                label=f"Linear ({alt_label.lower()})",
            )
        ax_scatter.set_xlabel(r"$u_s \cdot v_a$  [m/s]")
        ax_scatter.set_ylabel(r"$\dot{\chi}$  [rad/s]")
        ax_scatter.grid(True)

    handles, labels = [], []
    for ax in (ax_bar, ax_scatter):
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h)
                labels.append(l)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.10),
        ncol=min(len(labels), 4),
    )
    fig.tight_layout()

    out = Path("results") / "plots_paper" / flight["label"] / "turn_rate_composite.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, format="pdf", bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
    print(f"[{flight['label']}] saved {out}")


# ── Run every flight ──────────────────────────────────────────────────────────

for _flight in FLIGHTS:
    _ctx = prepare_flight(_flight)
    _phase_results, _phase_col = fit_flight(_ctx)
    make_figure(_ctx, _phase_results, _phase_col)

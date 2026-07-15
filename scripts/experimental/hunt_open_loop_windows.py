"""Hunt for quasi-open-loop windows in EKF-reconstructed flight data.

A window is quasi-open-loop when the steering input is held approximately
constant, so the course loop applies no corrective action through u_s and the
lateral dynamics evolve (nearly) uncontrolled. Within each window the course
rate is compared against the two-term turn-rate-law baseline, and the growth
rate of the residual is fitted. A positive fitted rate sigma corresponds to a
divergence time 1/sigma and a time to double ln(2)/sigma, directly comparable
with Re(lambda_u) of the unstable lateral mode from the modal stability
analysis (run_modal_stability.py in scripts/personal/wes-quasi-steady).

Method per flight:
  1. Reconstruct u_s, v_a, elevation, course angle chi and kinematic course
     rate chi_dot from the EKF results (same recipe as the turn-rate-law
     identification in awes_ekf.plotting.plot_aerodynamics).
  2. Fit the two-term law chi_dot ~ g_k*va*us + c_g*sin(chi)*cos(beta)/va per
     flight phase; the residual is the lateral deviation signal.
  3. Detect windows where the rolling range of u_s stays within --us-tol for
     at least --min-duration seconds, without crossing a phase boundary.
  4. In each window, estimate the residual growth rate sigma from the RMS
     envelope (last third vs first third of the window); an AR(2) fit on
     samples decimated past the smoothing correlation length adds the
     oscillation frequency and a secondary rate estimate for long windows.
  5. Repeat the growth and peak-frequency fits for the IMU roll and yaw rates,
     the channels where the lateral mode's eigenvector is largest.
  6. Write a CSV summary, a histogram of sigma per phase, median spectra of
     all channels over the quiet windows (spectra.pdf, peak location vs the
     model frequency Im(lambda_u)/2pi), and detail plots of the largest-sigma
     windows to results/<kite>/open_loop_windows/<flight>/.

Caveats (interpret distributions, not single windows):
  - Constant u_s opens the steering loop only; winch and depower still act.
  - Measurement noise biases AR-fitted sigma towards stability.
  - Over windows short compared with the divergence time, exponential and
    linear drift are nearly indistinguishable; the fitted rate is still the
    right statistic, but its shape says nothing about exponentiality.
  - Turbulence excites the same channels; a positive-sigma tail that
    concentrates in one phase (e.g. reel-in) is the signature to look for,
    not any individual event.

Run from the project root:
    python scripts/experimental/hunt_open_loop_windows.py --selftest
    python scripts/experimental/hunt_open_loop_windows.py --results results/LEI-V3-KITE/ekf/v3_2019-10-08.h5
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PHASE_NAME = {0: "single-phase", 1: "reel-out", 2: "rori", 3: "reel-in", 4: "riro"}
SMOOTH_SAMPLES = 21  # same moving average as the turn-rate identification


# ---------------------------------------------------------------------------
# Signal reconstruction (mirrors awes_ekf plot_turn_rate_identification)
# ---------------------------------------------------------------------------

def build_signals(ekf_output_df, flight_data_df):
    """Assemble the signals needed for window hunting into one DataFrame."""
    fd, res = flight_data_df, ekf_output_df
    time = fd["time"].to_numpy(dtype=float)

    position = np.array(
        [res["kite_position_x"], res["kite_position_y"], res["kite_position_z"]]
    )
    r_norm = np.linalg.norm(position, axis=0)
    beta = np.arctan2(position[2], np.hypot(position[0], position[1]))
    phi = np.arctan2(position[1], position[0])

    dbeta_dt = np.gradient(beta, time)
    dphi_dt = np.gradient(phi, time)
    v_beta = r_norm * dbeta_dt
    v_phi = r_norm * np.cos(beta) * dphi_dt
    v_tau = np.hypot(v_beta, v_phi)
    chi = np.unwrap(np.arctan2(v_phi, v_beta))

    v_beta_dot = np.gradient(v_beta, time)
    v_phi_dot = np.gradient(v_phi, time)
    kernel = np.ones(SMOOTH_SAMPLES) / SMOOTH_SAMPLES
    chi_dot = np.convolve(
        (v_beta * v_phi_dot - v_phi * v_beta_dot) / np.maximum(v_tau**2, 1e-6),
        kernel,
        mode="same",
    )

    signals = pd.DataFrame(
        {
            "time": time,
            "us": -fd["kcu_actual_steering"].to_numpy(dtype=float) / 100.0,
            "va": res["kite_apparent_windspeed"].to_numpy(dtype=float),
            "beta": beta,
            "chi": chi,
            "chi_dot": chi_dot,
        }
    )
    if "flight_phase_index" in fd.columns:
        signals["phase"] = fd["flight_phase_index"].to_numpy()
    else:
        print("Warning: no 'flight_phase_index' column; treating flight as one phase.")
        signals["phase"] = 0
    signals["cycle"] = fd["cycle"].to_numpy() if "cycle" in fd.columns else -1

    # IMU body rates: the lateral mode's eigenvector is dominated by roll/yaw,
    # so these carry the mode more directly than the course rate.
    for ch, cols in {
        "roll_rate": ["kite_roll_rate_0", "kite_roll_rate_1"],
        "yaw_rate": ["kite_yaw_rate_0", "kite_yaw_rate_1"],
    }.items():
        col = next((c for c in cols if c in fd.columns), None)
        signals[ch] = fd[col].to_numpy(dtype=float) if col else np.nan
        if col is None:
            print(f"Warning: no {ch} column found; channel skipped.")
    return signals


def fit_turn_rate_baseline(signals):
    """Per-phase two-term turn-rate-law fit; returns the residual chi_dot signal.

    chi_dot ~ g_k * va * us + c_g * sin(chi) * cos(beta) / va
    The residual is the lateral deviation whose growth the windows probe.
    """
    residual = np.full(len(signals), np.nan)
    coeffs = {}
    for phase, idx in signals.groupby("phase").groups.items():
        sub = signals.loc[idx]
        ok = np.isfinite(sub[["chi_dot", "va", "us"]]).all(axis=1) & (sub["va"] > 1.0)
        if ok.sum() < 50:
            continue
        s = sub[ok]
        A = np.column_stack(
            [
                s["va"] * s["us"],
                np.sin(s["chi"]) * np.cos(s["beta"]) / s["va"],
            ]
        )
        x, *_ = np.linalg.lstsq(A, s["chi_dot"].to_numpy(), rcond=None)
        coeffs[phase] = x
        A_all = np.column_stack(
            [
                sub["va"] * sub["us"],
                np.sin(sub["chi"]) * np.cos(sub["beta"]) / np.maximum(sub["va"], 1.0),
            ]
        )
        residual[idx] = sub["chi_dot"].to_numpy() - A_all @ x
    return residual, coeffs


# ---------------------------------------------------------------------------
# Window detection and growth fitting
# ---------------------------------------------------------------------------

def find_windows(signals, us_tol, min_duration):
    """Index ranges where u_s stays within us_tol for at least min_duration.

    Windows never cross a flight-phase boundary. Returns a list of
    (start_index, end_index) pairs, end exclusive.
    """
    dt = float(np.median(np.diff(signals["time"])))
    width = max(int(round(min_duration / dt)), 5)
    us = signals["us"]
    us_range = us.rolling(width, center=True).max() - us.rolling(width, center=True).min()
    quiet = (us_range <= us_tol).to_numpy() & np.isfinite(us.to_numpy())

    phase = signals["phase"].to_numpy()
    quiet &= np.r_[True, phase[1:] == phase[:-1]]

    windows = []
    edges = np.flatnonzero(np.diff(np.r_[False, quiet, False]))
    for start, stop in edges.reshape(-1, 2):
        if signals["time"].iloc[stop - 1] - signals["time"].iloc[start] >= min_duration:
            windows.append((int(start), int(stop)))
    return windows


def fit_growth_rate(time, residual):
    """Growth rate of the residual; returns (sigma_env, sigma_ar, frequency_hz, r2).

    The envelope estimate (RMS of last third vs first third) is the primary
    statistic: it is insensitive to the autocorrelation that the moving-average
    smoothing imprints on the residual. The AR(2) fit provides the mode
    frequency and a secondary rate estimate; it decimates to the smoothing
    correlation length, so it needs windows of roughly 8x that length and
    returns NaN otherwise.
    """
    if not np.isfinite(residual).all() or len(residual) < 12:
        return np.nan, np.nan, np.nan, np.nan

    # Envelope estimate on the full-rate residual.
    n3 = len(residual) // 3
    rms_a = np.sqrt(np.mean(residual[:n3] ** 2))
    rms_b = np.sqrt(np.mean(residual[-n3:] ** 2))
    t_gap = time[-n3:].mean() - time[:n3].mean()
    sigma_env = np.log(rms_b / rms_a) / t_gap if rms_a > 0 and t_gap > 0 else np.nan

    # AR(2) on samples decimated past the smoothing correlation length,
    # otherwise the fit tracks the moving-average filter, not the dynamics.
    dt = float(np.median(np.diff(time)))
    step = SMOOTH_SAMPLES
    r = residual[::step]
    if len(r) < 8:
        return sigma_env, np.nan, np.nan, np.nan

    A = np.column_stack([r[1:-1], r[:-2], np.ones(len(r) - 2)])
    y = r[2:]
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ coef
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    roots = np.roots([1.0, -coef[0], -coef[1]])
    z = roots[np.argmax(np.abs(roots))]
    if np.abs(z) < 1e-12:
        return sigma_env, np.nan, np.nan, r2
    lam = np.log(z.astype(complex)) / (step * dt)
    return sigma_env, float(lam.real), float(abs(lam.imag)) / (2 * np.pi), r2


# Channels probed per window: the course-rate residual plus the IMU body
# rates, where the lateral mode's eigenvector has its largest components.
CHANNELS = {"chi_dot": "residual", "roll_rate": "roll_rate", "yaw_rate": "yaw_rate"}


def _detrend_smooth(x):
    """Moving-average smoothing followed by linear detrending."""
    kernel = np.ones(SMOOTH_SAMPLES) / SMOOTH_SAMPLES
    xs = np.convolve(x, kernel, mode="same")
    idx = np.arange(len(xs), dtype=float)
    ok = np.isfinite(xs)
    if ok.sum() < 12:
        return np.full_like(xs, np.nan)
    p = np.polyfit(idx[ok], xs[ok], 1)
    return xs - np.polyval(p, idx)


def peak_frequency(x, dt, f_max=0.45):
    """Dominant frequency [Hz] from a Hann-windowed periodogram.

    f_max stays below 0.48 Hz because both the logged IMU rates (pre-smoothed
    upstream with a ~2 s box-car; notches at multiples of 0.5 Hz in all three
    axes) and the reconstructed course rate (21-sample moving average; notches
    at multiples of 0.48 Hz) carry no usable information above the first
    filter notch -- the apparent bumps between notches are shaped noise.
    """
    x = x[np.isfinite(x)]
    if len(x) < 32:
        return np.nan
    x = x - x.mean()
    taper = np.hanning(len(x))
    psd = np.abs(np.fft.rfft(x * taper)) ** 2
    f = np.fft.rfftfreq(len(x), dt)
    sel = (f > 0) & (f <= f_max)
    if not sel.any() or not np.any(psd[sel] > 0):
        return np.nan
    return float(f[sel][np.argmax(psd[sel])])


def analyse_windows(signals, windows):
    rows = []
    for start, stop in windows:
        w = signals.iloc[start:stop]
        time = w["time"].to_numpy()
        dt = float(np.median(np.diff(time)))
        chan = {}
        sigma_env = sigma_ar = r2 = np.nan
        for name, col in CHANNELS.items():
            sig = w[col].to_numpy()
            if col != "residual":
                sig = _detrend_smooth(sig)
            s_env, s_ar, _, r2_ch = fit_growth_rate(time, sig)
            chan[f"sigma_{name}"] = s_env
            chan[f"fpeak_{name}"] = peak_frequency(sig, dt)
            if name == "chi_dot":
                sigma_env, sigma_ar, r2 = s_env, s_ar, r2_ch
        rows.append(
            {
                "t_start": w["time"].iloc[0],
                "t_end": w["time"].iloc[-1],
                "duration": w["time"].iloc[-1] - w["time"].iloc[0],
                "phase": int(w["phase"].iloc[0]),
                "phase_name": PHASE_NAME.get(int(w["phase"].iloc[0]), "?"),
                "cycle": int(w["cycle"].iloc[0]),
                "us_mean": w["us"].mean(),
                "va_mean": w["va"].mean(),
                "beta_mean_deg": np.degrees(w["beta"].mean()),
                "sigma": sigma_env,
                "sigma_ar": sigma_ar,
                **chan,
                "ar2_r2": r2,
                "i_start": start,
                "i_stop": stop,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def summarize(table, model_sigma=None):
    if table.empty:
        print("No quasi-open-loop windows found. Try loosening --us-tol or "
              "shortening --min-duration.")
        return
    print(f"\n{len(table)} quasi-open-loop windows found.")
    for phase, sub in table.groupby("phase"):
        sig = sub["sigma"].dropna()
        if sig.empty:
            continue
        med = sig.median()
        line = (
            f"  {PHASE_NAME.get(phase, phase):>12}: {len(sub):3d} windows, "
            f"total {sub['duration'].sum():6.0f} s, median sigma {med:+.4f} 1/s"
        )
        if med > 0:
            line += f"  (tau_div {1 / med:.1f} s, T2 {np.log(2) / med:.1f} s)"
        print(line)
        for name in CHANNELS:
            if name == "chi_dot":
                continue
            s = sub[f"sigma_{name}"].dropna()
            f_pk = sub[f"fpeak_{name}"].dropna()
            if s.empty:
                continue
            print(f"  {'':>12}  {name}: median sigma {s.median():+.4f} 1/s, "
                  f"median peak freq {f_pk.median():.3f} Hz")
    if model_sigma is not None:
        print(f"  model Re(lambda_u) = {model_sigma:+.4f} 1/s "
              f"(tau_div {1 / model_sigma:.1f} s, T2 {np.log(2) / model_sigma:.1f} s)")


def plot_histogram(table, out_dir, model_sigma=None):
    fig, ax = plt.subplots(figsize=(7, 4))
    for phase, sub in table.groupby("phase"):
        ax.hist(
            sub["sigma"].dropna(),
            bins=30,
            alpha=0.6,
            label=f"{PHASE_NAME.get(phase, phase)} (n={len(sub)})",
        )
    ax.axvline(0, color="k", lw=0.8)
    if model_sigma is not None:
        ax.axvline(model_sigma, color="tab:red", ls="--", label=r"model $\Re(\lambda_u)$")
    ax.set_xlabel(r"envelope growth rate $\sigma$ [1/s]")
    ax.set_ylabel("windows")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "sigma_histogram.pdf")
    plt.close(fig)


def _window_psd(x, dt, f_grid):
    """Unit-power Hann periodogram interpolated onto a common frequency grid."""
    x = x[np.isfinite(x)]
    if len(x) < 64:
        return None
    x = x - x.mean()
    taper = np.hanning(len(x))
    psd = np.abs(np.fft.rfft(x * taper)) ** 2
    f = np.fft.rfftfreq(len(x), dt)
    total = np.trapezoid(psd, f)
    if total <= 0:
        return None
    return np.interp(f_grid, f, psd / total, left=np.nan, right=np.nan)


def plot_spectra(signals, table, out_dir, model_freq=None):
    """Median spectrum of each channel over the quiet windows, per phase.

    A common peak in roll rate and yaw rate that also appears in the
    course-rate residual is the frequency-domain fingerprint of the lateral
    mode; compare its location with the model value Im(lambda_u)/(2*pi).
    """
    f_grid = np.linspace(0.02, 0.45, 120)  # capped below the first filter notch
    dt = float(np.median(np.diff(signals["time"])))
    phases = [p for p, sub in table.groupby("phase") if len(sub) >= 5]
    if not phases:
        return
    fig, axes = plt.subplots(
        1, len(phases), figsize=(5 * len(phases), 4), squeeze=False
    )
    for ax, phase in zip(axes[0], phases):
        sub = table[table["phase"] == phase]
        for name, col in CHANNELS.items():
            psds = []
            for _, row in sub.iterrows():
                w = signals.iloc[int(row["i_start"]) : int(row["i_stop"])]
                p = _window_psd(w[col].to_numpy(), dt, f_grid)
                if p is not None:
                    psds.append(p)
            if psds:
                med = np.nanmedian(np.vstack(psds), axis=0)
                ax.semilogy(f_grid, med, label=f"{name} (n={len(psds)})")
        if model_freq is not None:
            ax.axvline(model_freq, color="tab:red", ls="--",
                       label=r"model $\Im(\lambda_u)/2\pi$")
        ax.set_title(PHASE_NAME.get(phase, str(phase)))
        ax.set_xlabel("frequency [Hz]")
        ax.set_ylabel("normalized PSD [1/Hz]")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "spectra.pdf")
    plt.close(fig)


def plot_window_details(signals, table, out_dir, top_n):
    ranked = table.dropna(subset=["sigma"]).nlargest(top_n, "sigma")
    for rank, (_, row) in enumerate(ranked.iterrows(), start=1):
        w = signals.iloc[int(row["i_start"]) : int(row["i_stop"])]
        t0 = w["time"].iloc[0]
        fig, axes = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
        axes[0].plot(w["time"] - t0, w["us"], color="tab:blue")
        axes[0].set_ylabel(r"$u_s$ [-]")
        axes[1].plot(w["time"] - t0, w["residual"], color="tab:gray", lw=0.8)
        if row["sigma"] > 0:
            rms0 = np.sqrt(np.mean(w["residual"].iloc[: len(w) // 3] ** 2))
            env = np.sqrt(2) * rms0 * np.exp(row["sigma"] * (w["time"] - t0))
            axes[1].plot(w["time"] - t0, env, "r--", lw=1.0)
            axes[1].plot(w["time"] - t0, -env, "r--", lw=1.0,
                         label=rf"$\sigma$={row['sigma']:+.3f} 1/s")
            axes[1].legend()
        axes[1].set_ylabel(r"$\delta\dot{\chi}$ [rad/s]")
        axes[1].set_xlabel("time in window [s]")
        axes[0].set_title(
            f"{row['phase_name']}, cycle {row['cycle']}, "
            f"t = {row['t_start']:.0f}-{row['t_end']:.0f} s, "
            f"va = {row['va_mean']:.1f} m/s"
        )
        fig.tight_layout()
        fig.savefig(out_dir / f"window_{rank:02d}.pdf")
        plt.close(fig)


# ---------------------------------------------------------------------------
# Self-test on synthetic data
# ---------------------------------------------------------------------------

def selftest():
    """Synthetic flight: verifies detection and growth-rate recovery."""
    rng = np.random.default_rng(7)
    dt, t_end = 0.1, 1200.0
    time = np.arange(0.0, t_end, dt)
    n = len(time)

    # Steering: figure-eight activity in reel-out, quiet holds in reel-in.
    phase = np.where((time // 150) % 2 == 0, 1, 3)
    us = np.where(
        phase == 1,
        0.15 * np.sin(2 * np.pi * time / 20),
        0.02 * np.round(np.sin(2 * np.pi * time / 300)),
    )
    va = np.where(phase == 1, 25.0, 10.0) + rng.normal(0, 0.3, n)
    beta = np.radians(np.where(phase == 1, 30.0, 60.0))
    chi = np.where(phase == 1, np.pi / 2, 0.0)

    sigma_true = 0.06  # divergence time ~17 s, planted in reel-in holds
    g_k, c_g = 0.1, 1.0
    residual_true = np.zeros(n)
    for start, stop in _contiguous(phase == 3):
        t_loc = time[start:stop] - time[start]
        residual_true[start:stop] = 0.002 * np.exp(sigma_true * t_loc) * np.cos(
            2 * np.pi * t_loc / 25 + rng.uniform(0, 2 * np.pi)
        )
    chi_dot = g_k * va * us + c_g * np.sin(chi) * np.cos(beta) / va + residual_true
    chi_dot += rng.normal(0, 5e-4, n)

    # Body rates: growing 0.15 Hz oscillation planted in the reel-in holds.
    roll_rate = rng.normal(0, 0.003, n)
    yaw_rate = rng.normal(0, 0.003, n)
    for start, stop in _contiguous(phase == 3):
        t_loc = time[start:stop] - time[start]
        osc = np.exp(sigma_true * t_loc) * np.sin(2 * np.pi * 0.15 * t_loc)
        roll_rate[start:stop] += 0.01 * osc
        yaw_rate[start:stop] += 0.005 * osc

    signals = pd.DataFrame(
        {"time": time, "us": us, "va": va, "beta": beta, "chi": chi,
         "chi_dot": chi_dot, "phase": phase, "cycle": (time // 300).astype(int),
         "roll_rate": roll_rate, "yaw_rate": yaw_rate}
    )
    signals["residual"], _ = fit_turn_rate_baseline(signals)

    windows = find_windows(signals, us_tol=0.01, min_duration=20.0)
    table = analyse_windows(signals, windows)
    reelin = table[(table["phase"] == 3) & table["sigma"].notna()]
    assert len(reelin) >= 4, f"expected >=4 reel-in windows, got {len(reelin)}"
    med = reelin["sigma"].median()
    err = abs(med - sigma_true) / sigma_true
    print(f"selftest: {len(reelin)} reel-in windows, median sigma {med:+.4f} 1/s "
          f"(true {sigma_true:+.4f}, error {100 * err:.0f}%)")
    assert err < 0.35, "recovered sigma deviates >35% from planted value"

    med_roll = reelin["sigma_roll_rate"].median()
    fpk_roll = reelin["fpeak_roll_rate"].median()
    print(f"selftest: roll-rate median sigma {med_roll:+.4f} 1/s, "
          f"peak freq {fpk_roll:.3f} Hz (true {sigma_true:+.4f} 1/s, 0.150 Hz)")
    assert abs(med_roll - sigma_true) / sigma_true < 0.35, "roll-rate sigma off"
    assert 0.10 <= fpk_roll <= 0.20, "roll-rate peak frequency off"

    reelout = table[(table["phase"] == 1) & table["sigma"].notna()]
    print(f"selftest: {len(reelout)} reel-out windows (steering active, expect few/none)")
    print("selftest passed.")


def _contiguous(mask):
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False]))
    return edges.reshape(-1, 2)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0], epilog="Run from the project root."
    )
    parser.add_argument("--results", type=Path, default=None,
                        help="EKF results HDF5 (results/<kite>/ekf/*.h5). "
                             "Omit to list available files.")
    parser.add_argument("--us-tol", type=float, default=0.01,
                        help="max allowed u_s range within a window [-] (default 0.01)")
    parser.add_argument("--min-duration", type=float, default=8.0,
                        help="minimum window duration [s] (default 8)")
    parser.add_argument("--top-n", type=int, default=6,
                        help="number of detail plots for the largest-sigma windows")
    parser.add_argument("--model-sigma", type=float, default=None,
                        help="model Re(lambda_u) [1/s] to overlay on the histogram")
    parser.add_argument("--model-freq", type=float, default=None,
                        help="model Im(lambda_u)/2pi [Hz] to overlay on the spectra")
    parser.add_argument("--selftest", action="store_true",
                        help="run on synthetic data and exit")
    args = parser.parse_args()

    if args.selftest:
        selftest()
        return

    if args.results is None:
        candidates = sorted(Path("results").glob("*/ekf/*.h5"))
        if not candidates:
            sys.exit("No EKF results found under results/*/ekf/. Run "
                     "run_analysis_ekf.py first, or pass --results.")
        print("Available EKF results (pass one via --results):")
        for c in candidates:
            print(f"  {c}")
        return

    from awes_ekf.load_data.read_data import read_results_from_hdf5

    ekf_output_df, flight_data_df, _ = read_results_from_hdf5(args.results)
    signals = build_signals(ekf_output_df, flight_data_df)
    signals["residual"], coeffs = fit_turn_rate_baseline(signals)
    for phase, x in coeffs.items():
        print(f"baseline fit {PHASE_NAME.get(phase, phase)}: "
              f"g_k = {x[0]:.4f} 1/m, c_g = {x[1]:.4f} m/s^2")

    windows = find_windows(signals, args.us_tol, args.min_duration)
    table = analyse_windows(signals, windows)

    kite_name = (args.results.parent.parent.name
                 if args.results.parent.name == "ekf" else "unknown-kite")
    out_dir = Path("results") / kite_name / "open_loop_windows" / args.results.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    table.drop(columns=["i_start", "i_stop"]).to_csv(
        out_dir / "open_loop_windows.csv", index=False
    )
    summarize(table, args.model_sigma)
    if not table.empty:
        plot_histogram(table, out_dir, args.model_sigma)
        plot_spectra(signals, table, out_dir, args.model_freq)
        plot_window_details(signals, table, out_dir, args.top_n)
    print(f"\nOutputs written to {out_dir}")


if __name__ == "__main__":
    main()

"""The turn rate of each LEI-V3 ROM against the flight: the check of the roll gain.

Left: the turn rate each ROM solves with the logged steering as input
(validate_quasi_steady_state_v3.py --steering measured) against the measured
turn rate, reel-out of the held-out cycles. Right: the turn-rate law
chi_dot = K v_a u_s + c_g g cos(beta) sin(chi) / v_tau (Fechner's law with
the gravity term), binned in v_a u_s with the gravity term removed, for the
flight and for each ROM, with the fitted K. The flight's K is what the roll
gain is identified against (refine_rom_flight_output_error.py --gain turnlaw);
the gravity coefficient c_g is a point-mass property the ROMs cannot match
(they give ~1.1-1.2, the kite 0.47).

Usage (project root; run the validator in --steering measured first)
    python scripts/identification/plot_rom_turn_rate_law.py
"""

from __future__ import annotations

from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from awetrim.identification.controls import FLIGHT_STEERING_ZERO_OFFSET_2019
from awetrim.plotting.plotting import PALETTE, set_plot_style

ROOT = Path("results/LEI-V3-KITE/rom_validation")
EKF_FILE = Path("results/LEI-V3-KITE/ekf/LEI-V3 Kite_2019-10-08.h5")
ROMS = {
    "semi_empirical": ("Semi-empirical ROM", PALETTE["Vermillion"]),
    "aerostructural": ("Aerostructural ROM", PALETTE["Blue"]),
    "aerostructural_flight": ("Aerostructural + flight correction", PALETTE["Bluish Green"]),
}
LAG_SAMPLES = 3  # the flight's 0.3 s steering response, at 10 Hz


def reel_out_samples(rom: str) -> pd.DataFrame:
    with h5py.File(EKF_FILE, "r") as hf:
        ekf = pd.DataFrame({"time": np.asarray(hf["ekf_output"]["time"][()]),
                            "va": np.asarray(hf["ekf_output"]["kite_apparent_windspeed"][()]),
                            "course": np.asarray(hf["flight_data"]["kite_course"][()])})
    d = pd.read_csv(ROOT / rom / "qs_validation_measured_steering.csv").merge(ekf, on="time")
    d = d[(d.speed_radial > 0.5) & (d.input_depower < 1.9) & (d.kite_elevation < 0.75)].copy()
    u = (d.measured_input_steering + FLIGHT_STEERING_ZERO_OFFSET_2019).shift(LAG_SAMPLES)
    d["x_turn"] = d.va * u
    d["x_grav"] = 9.81 * np.cos(d.kite_elevation) * np.sin(d.course) / d.measured_speed_tangential
    return d.dropna(subset=["x_turn"])


def turn_law(d: pd.DataFrame, y: np.ndarray) -> tuple[float, float]:
    A = np.c_[d.x_turn, d.x_grav]
    (k, cg), *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(k), float(cg)


def binned(x, y, edges):
    idx = np.digitize(x, edges)
    centres, medians = [], []
    for i in range(1, len(edges)):
        m = idx == i
        if m.sum() >= 30:
            centres.append(np.median(x[m])); medians.append(np.median(y[m]))
    return np.array(centres), np.array(medians)


def main() -> None:
    set_plot_style()
    fig, (ax_s, ax_l) = plt.subplots(1, 2, figsize=(11.5, 5.0))
    edges = np.linspace(-5.0, 5.0, 21)
    flight_done = False
    for rom, (label, colour) in ROMS.items():
        path = ROOT / rom / "qs_validation_measured_steering.csv"
        if not path.exists():
            continue
        d = reel_out_samples(rom)
        ok = d.predicted_course_rate.notna()
        h = d[ok]
        x, y = h.measured_course_rate.to_numpy(), h.predicted_course_rate.to_numpy()
        slope = np.polyfit(x, y, 1)[0]
        ax_s.scatter(x, y, s=2, alpha=0.12, color=colour, rasterized=True)
        k, cg = turn_law(h, y)
        xb, yb = binned(h.x_turn.to_numpy(), y - cg * h.x_grav.to_numpy(), edges)
        ax_l.plot(xb, yb, "o-", color=colour, ms=4, lw=1.4,
                  label=f"{label}: $K$ {k:.3f}, $c_g$ {cg:.2f}")
        ax_s.plot([], [], color=colour, lw=2, label=f"{label}: slope {slope:.2f}, "
                  f"corr {np.corrcoef(x, y)[0, 1]:.2f}")
        if not flight_done:
            kf, cgf = turn_law(d, d.measured_course_rate.to_numpy())
            xb, yb = binned(d.x_turn.to_numpy(),
                            (d.measured_course_rate - cgf * d.x_grav).to_numpy(), edges)
            ax_l.plot(xb, yb, "s-", color=PALETTE["Black"], ms=4, lw=1.6,
                      label=f"Flight (EKF): $K$ {kf:.3f}, $c_g$ {cgf:.2f}")
            ax_l.plot(edges, kf * edges, ":", color=PALETTE["Black"], lw=1.0)
            flight_done = True
    lim = 2.5
    ax_s.plot([-lim, lim], [-lim, lim], color=PALETTE["Black"], lw=0.8)
    ax_s.set_xlim(-lim, lim); ax_s.set_ylim(-lim, lim); ax_s.set_aspect("equal")
    ax_s.set_xlabel(r"measured $\dot\chi$ (rad s$^{-1}$)")
    ax_s.set_ylabel(r"ROM $\dot\chi$ with the logged $u_s$ in (rad s$^{-1}$)")
    ax_s.legend(frameon=False, fontsize=7.5, loc="upper left")
    ax_s.set_title("Turn rate, held-out reel-out", fontsize=10)
    ax_l.axhline(0.0, color=PALETTE["Black"], lw=0.6); ax_l.axvline(0.0, color=PALETTE["Black"], lw=0.6)
    ax_l.set_xlabel(r"$v_a u_s$ (m s$^{-1}$), logged $u_s$, 0.3 s lag")
    ax_l.set_ylabel(r"$\dot\chi - c_g\,g\cos\beta\sin\chi / v_\tau$ (rad s$^{-1}$)")
    ax_l.legend(frameon=False, fontsize=7.5, loc="upper left")
    ax_l.set_title(r"Turn-rate law $\dot\chi = K v_a u_s + c_g\, g\cos\beta\sin\chi/v_\tau$ (binned medians)",
                   fontsize=10)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(ROOT / f"rom_turn_rate_law.{ext}", dpi=200)
    print(f"Wrote {ROOT / 'rom_turn_rate_law'}.png/.pdf")


if __name__ == "__main__":
    main()

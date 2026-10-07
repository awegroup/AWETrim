"""Flight validation of the two LEI-V3 ROMs: measured vs predicted, 2019.

Reads the per-sample CSVs written by
``scripts/reduced-order-model/validation/validate_quasi_steady_state_v3.py
--rom <name> [--steering measured]`` (results/LEI-V3-KITE/rom_validation/<rom>/
qs_validation[_measured_steering].csv) and draws, for one cycle, the tether
force, tangential speed and the third quantity of the mode against the
measurement -- the steering the ROM needs (``solved``) or the turn rate it
predicts from the logged steering (``measured``) -- plus per-phase error
statistics over every cycle in the files (all converged roots; pre-stall
and speed-stable shares reported alongside).

Usage (project root)
    python scripts/identification/plot_rom_flight_validation.py --cycle 64
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from awetrim.identification.controls import FLIGHT_STEERING_ZERO_OFFSET_2019 as STEERING_ZERO_OFFSET
from awetrim.plotting.plotting import PALETTE, set_plot_style

ROOT = Path("results/LEI-V3-KITE/rom_validation")
ROMS = {
    "semi_empirical": ("Semi-empirical ROM", dict(color=PALETTE["Vermillion"], lw=1.4, ls="--")),
    "aerostructural": ("Aerostructural ROM", dict(color=PALETTE["Blue"], lw=1.4, ls="-")),
    "aerostructural_flight": ("Aerostructural + flight correction",
                              dict(color=PALETTE["Bluish Green"], lw=1.4, ls="-.")),
}
MEASURED = dict(color=PALETTE["Black"], lw=1.0)
#: Phase masks of validate_quasi_steady_state_v3.py (power-tape length 1.9 m
#: separates powered from depowered).
DEPOWER_SPLIT = 1.9
#: Stall angle [rad] of each ROM (aerodynamics.params.angle_of_attack_stall).
ROM_FILES = {"semi_empirical": "rom_config_semi_empirical.yaml",
             "aerostructural": "rom_config_aerostructural.yaml",
             "aerostructural_flight": "rom_config_aerostructural_flight_corrected.yaml"}
STALL_ANGLE = {
    rom: float(yaml.safe_load(
        (Path("data/LEI-V3-KITE") / name).read_text(encoding="utf-8")
    )["aerodynamics"]["params"]["angle_of_attack_stall"])
    for rom, name in ROM_FILES.items() if (Path("data/LEI-V3-KITE") / name).exists()
}


def phase_masks(d):
    reel_out = (d.speed_radial > 0.5) & (d.input_depower < DEPOWER_SPLIT) & (d.kite_elevation < 0.75)
    reel_in = (d.speed_radial < -0.5) & (d.input_depower >= DEPOWER_SPLIT)
    return {"reel-out": reel_out, "reel-in": reel_in}


def error_table(frames) -> pd.DataFrame:
    rows = []
    for rom, d in frames.items():
        for phase, mask in phase_masks(d).items():
            g = d[mask]
            # Every converged root is scored: the validator already rejects
            # post-stall roots where a pre-stall one exists.
            ok = g.predicted_tension.notna()
            alpha_stall = STALL_ANGLE[rom]
            pre_stall = (g.predicted_angle_of_attack[ok] < alpha_stall).mean()
            speed_stable = (
                (g.predicted_speed_stability[ok] < 0.0).mean()
                if "predicted_speed_stability" in g else np.nan
            )
            e_f = 100 * (g.predicted_tension[ok] / g.measured_tension[ok] - 1)
            e_v = 100 * (g.predicted_speed_tangential[ok] / g.measured_speed_tangential[ok] - 1)
            # The turn rate is only a prediction in --steering measured (in
            # the solved mode it is the measured input copied through).
            predicts_turn_rate = (
                "predicted_course_rate" in g
                and g.predicted_course_rate[ok].notna().any()
                and not np.allclose(g.predicted_course_rate[ok], g.measured_course_rate[ok])
            )
            if predicts_turn_rate:
                e_chi = (g.predicted_course_rate[ok] - g.measured_course_rate[ok]).abs()
                chi_corr = np.corrcoef(g.predicted_course_rate[ok], g.measured_course_rate[ok])[0, 1]
            else:
                e_chi, chi_corr = pd.Series([np.nan]), np.nan
            # In the solved mode the steering is the prediction: regress it on
            # the logged steering (+ the rig's zero offset). Medians of |u_s|
            # are NOT a valid check here -- thousands of near-zero-steering
            # samples make a ratio of medians read 2x for a ROM whose slope
            # is 1.0 (that misread cost half a day on 2026-10-06).
            if not predicts_turn_rate and "measured_input_steering" in g:
                x = (g.measured_input_steering[ok] + STEERING_ZERO_OFFSET).to_numpy()
                y = g.predicted_input_steering[ok].to_numpy()
                slope, intercept = np.polyfit(x, y, 1) if ok.sum() > 2 else (np.nan, np.nan)
                us_rms = float(np.sqrt(np.mean((y - slope * x - intercept) ** 2)))
                us_corr = float(np.corrcoef(x, y)[0, 1])
            else:
                slope, us_rms, us_corr = np.nan, np.nan, np.nan
            rows.append({
                "rom": rom, "phase": phase, "samples": int(mask.sum()),
                "solved_%": 100 * ok.mean(),
                "pre_stall_%": 100 * pre_stall,
                "speed_stable_%": 100 * speed_stable,
                "tension_err_median_%": e_f.median(),
                "tension_err_IQR_%": f"{e_f.quantile(.25):+.0f}..{e_f.quantile(.75):+.0f}",
                "v_tau_err_median_%": e_v.median(),
                "chi_dot_abs_err_median_rad_s": e_chi.median(),
                "chi_dot_corr": chi_corr,
                "us_slope": slope, "us_rms": us_rms, "us_corr": us_corr,
            })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cycle", type=int, default=64)
    parser.add_argument("--steering", choices=("solved", "measured"), default="solved")
    parser.add_argument("--tether", choices=("rigid", "williams"), default="rigid",
                        help="which validator run (validate_quasi_steady_state_v3.py --tether)")
    args = parser.parse_args()
    set_plot_style()
    suffix = (("_williams" if args.tether == "williams" else "")
              + ("_measured_steering" if args.steering == "measured" else ""))
    frames = {rom: pd.read_csv(ROOT / rom / f"qs_validation{suffix}.csv") for rom in ROMS
              if (ROOT / rom / f"qs_validation{suffix}.csv").exists()}

    table = error_table(frames)
    print(table.round(1).to_string(index=False))
    table.round(3).to_csv(ROOT / f"rom_flight_validation_errors{suffix}.csv", index=False)

    fig, axes = plt.subplots(3, 1, figsize=(10.0, 8.0), sharex=True)
    base = next(iter(frames.values()))  # any ROM: the measured columns are the same
    cyc = base.cycle == args.cycle
    t = base.time[cyc] - base.time[cyc].iloc[0]
    for ax, (col, label, scale) in zip(axes, (
        ("tension", r"$F_\mathrm{t,g}$ (kN)", 1e-3),
        ("speed_tangential", r"$v_\tau$ (m s$^{-1}$)", 1.0),
        ("course_rate", r"$\dot{\chi}$ (rad s$^{-1}$)", 1.0) if args.steering == "measured"
        else ("input_steering", r"$u_s$", 1.0),
    )):
        ax.plot(t, scale * base[f"measured_{col}"][cyc], label="Flight (EKF)", **MEASURED)
        for rom, (name, style) in ROMS.items():
            if rom not in frames:
                continue
            d = frames[rom]
            ax.plot(t, scale * d[f"predicted_{col}"][d.cycle == args.cycle], label=name, **style)
        ax.set_ylabel(label)
        for m, shade in ((phase_masks(base)["reel-in"][cyc], 0.08),):
            ax.fill_between(t, 0, 1, where=m.to_numpy(), transform=ax.get_xaxis_transform(),
                            color=PALETTE["Black"], alpha=shade, lw=0)
    axes[0].legend(frameon=False, fontsize=9, ncol=3, loc="upper right")
    axes[-1].set_xlabel("Time in cycle (s)")
    mode = ("logged steering in, turn rate out" if args.steering == "measured"
            else "turn rate in, steering out")
    fig.suptitle(f"LEI-V3 2019 flight, cycle {args.cycle}: quasi-steady ROM reconstruction, "
                 f"{mode} (shaded: reel-in)", fontsize=10)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(ROOT / f"rom_flight_validation_cycle{args.cycle}{suffix}.{ext}", dpi=200)
    print(f"Wrote {ROOT / f'rom_flight_validation_cycle{args.cycle}{suffix}'}.png/.pdf")


if __name__ == "__main__":
    main()

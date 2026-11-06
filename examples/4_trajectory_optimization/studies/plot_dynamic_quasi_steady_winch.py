# Copyright (c) 2023-2026 Oriol Cayon, Delft University of Technology
# SPDX-License-Identifier: Apache-2.0
"""Plot compare_dynamic_quasi_steady_winch.py's runs: one row per case.

Usage (from the repository root):
    python examples/4_trajectory_optimization/studies/plot_dynamic_quasi_steady_winch.py \\
        [--dir results/LEI-V3-KITE/winch_dynamics]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from awetrim.system.winch import Drivetrain

MODELS = [
    ("qs", "quasi-steady (friction)", "#2a78d6", "-"),
    ("qs_drum", "quasi-steady kite + drum inertia", "#1baf7a", "-."),
    ("dyn", "full dynamic (kite + drum)", "#eb6834", "--"),
]
CASES = [("eight", "One figure eight (reel-out)"), ("cycle", "Whole pumping cycle")]


def drivetrain_from_system(system_yaml):
    drum = yaml.safe_load(open(system_yaml, encoding="utf-8"))["components"]["ground_station"]["drums"][0]
    return Drivetrain(
        inertia_mass=float(drum["inertia_equivalent_mass"]),
        friction_coulomb=float(drum["friction_coulomb"]),
        friction_viscous=float(drum["friction_viscous"]),
    )


def metrics(df, drivetrain, with_inertia):
    t = df["t"].to_numpy()
    v = df["speed_radial"].to_numpy()
    T = df["tension_tether_ground"].to_numpy()
    a = np.r_[0.0, np.diff(v) / np.diff(t)]  # backward difference, as the march
    m = drivetrain.inertia_mass if with_inertia else 0.0
    f_motor = T - drivetrain.friction_force(v) - m * a
    shaft = f_motor * v
    span = t[-1] - t[0]
    return dict(
        t=t - t[0], v=v, T=T, a=a, f_motor=f_motor, shaft=shaft,
        vtau=df["speed_tangential"].to_numpy(),
        duration=span,
        p_tether=np.trapezoid(T * v, t) / span,
        p_shaft=np.trapezoid(shaft, t) / span,
        t_max=T.max(), f_motor_max=f_motor.max(),
        r_change=df["distance_radial"].iloc[-1] - df["distance_radial"].iloc[0],
        nodes=len(df),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dir", default="results/LEI-V3-KITE/winch_dynamics")
    parser.add_argument("--system", default="data/LEI-V3-KITE/system.yaml")
    args = parser.parse_args()
    folder = Path(args.dir)
    drivetrain = drivetrain_from_system(args.system)

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": "#e4e4e0", "grid.linewidth": 0.6})
    fig, axs = plt.subplots(len(CASES), 4, figsize=(16, 7.5), constrained_layout=True,
                            squeeze=False)
    rows = []
    for row, (case, title) in enumerate(CASES):
        for name, label, color, ls in MODELS:
            path = folder / f"{case}_{name}.csv"
            if not path.exists():
                continue
            d = metrics(pd.read_csv(path), drivetrain, with_inertia=name != "qs")
            rows.append((title, label, d))
            kw = dict(color=color, ls=ls, lw=1.5)
            axs[row, 0].plot(d["t"], d["vtau"], label=label, **kw)
            axs[row, 1].plot(d["t"], d["v"], **kw)
            axs[row, 2].plot(d["t"], d["T"] / 1e3, **kw)
            axs[row, 3].plot(d["t"], d["shaft"] / 1e3,
                             label=f"{d['p_shaft'] / 1e3:.2f} kW", **kw)
        axs[row, 0].set(ylabel="tangential speed v_τ (m s$^{-1}$)", title=f"{title}: tangential speed")
        axs[row, 1].set(ylabel="reel speed v_r (m s$^{-1}$)", title="Reel speed")
        axs[row, 2].set(ylabel="tension at ground T (kN)", title="Tether tension")
        axs[row, 3].set(ylabel="shaft power (kW)", title="Shaft power (legend: mean)")
        axs[row, 2].axhline(8.4, color="#6b6a63", lw=0.9, ls=":")
        axs[row, 3].axhline(0, color="#6b6a63", lw=0.8)
        axs[row, 3].legend(fontsize=8, frameon=False)
        for ax in axs[row]:
            ax.set_xlabel("time (s)")
    axs[0, 0].legend(fontsize=8, frameon=False)
    fig.suptitle("LEI-V3: full dynamic vs quasi-steady forward simulation, drivetrain friction "
                 "in all (no optimisation)", fontsize=11)
    out = folder / "dynamic_vs_quasi_steady.png"
    fig.savefig(out, dpi=150)

    print(f"{'case':28s} {'model':34s} {'time s':>7s} {'P_tether kW':>11s} {'P_shaft kW':>10s}"
          f" {'T max kN':>9s} {'F_motor max kN':>14s} {'dr m':>7s} {'nodes':>6s}")
    for title, label, d in rows:
        print(f"{title:28s} {label:34s} {d['duration']:7.1f} {d['p_tether'] / 1e3:11.2f}"
              f" {d['p_shaft'] / 1e3:10.2f} {d['t_max'] / 1e3:9.2f}"
              f" {d['f_motor_max'] / 1e3:14.2f} {d['r_change']:7.1f} {d['nodes']:6d}")
    print(out)


if __name__ == "__main__":
    main()

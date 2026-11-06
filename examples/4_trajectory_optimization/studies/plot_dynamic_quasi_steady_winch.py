# Copyright (c) 2023-2026 Oriol Cayon, Delft University of Technology
# SPDX-License-Identifier: Apache-2.0
"""Plot compare_dynamic_quasi_steady_winch.py's runs: one figure per comparison.

- ``tether_inertia``: quasi-steady without vs with tether inertia (ideal winch)
- ``dynamic_ideal_winch``: with tether inertia, quasi-steady vs dynamic, no winch
  friction (the dynamic run with the drum's inertia)
- ``dynamic_winch_friction``: the same with winch friction

Rows: one figure eight, the whole cycle. A run that leaves the physical branch
(tension below the ROM's floor or steering beyond its box, where the march
jumped to the low-lift root because the kite could not follow the path) is cut
at that node and marked.

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

BLUE, ORANGE = "#2a78d6", "#eb6834"
# figure: (title, [(model, label, colour, style)])
COMPARISONS = {
    "tether_inertia": (
        "Quasi-steady, ideal winch: without vs with tether inertia",
        [("qs", "quasi-steady", BLUE, "-"),
         ("qs_ti", "quasi-steady + tether inertia", ORANGE, "--")],
    ),
    "dynamic_ideal_winch": (
        "With tether inertia, no winch friction: quasi-steady vs dynamic",
        [("qs_ti", "quasi-steady", BLUE, "-"),
         ("dyn_ti", "dynamic (kite, tether, drum)", ORANGE, "--")],
    ),
    "dynamic_winch_friction": (
        "With tether inertia and winch friction: quasi-steady vs dynamic",
        [("qs_ti_f", "quasi-steady", BLUE, "-"),
         ("dyn_ti_f", "dynamic (kite, tether, drum)", ORANGE, "--")],
    ),
}
# what each model's shaft power charges: (drum inertia, friction)
DRIVETRAIN_PARTS = {
    "qs": (False, False), "qs_ti": (False, False), "dyn_ti": (True, False),
    "qs_ti_f": (False, True), "dyn_ti_f": (True, True),
}
CASES = [("eight", "One figure eight (reel-out)"), ("cycle", "Whole pumping cycle")]
TENSION_FLOOR = 602.0  # N, the ROM's validity floor (rom_config validity)
STEERING_BOX = 0.357  # the ROM's steering validity


def system_drivetrain(system_yaml):
    drum = yaml.safe_load(open(system_yaml, encoding="utf-8"))["components"]["ground_station"]["drums"][0]
    return float(drum["inertia_equivalent_mass"]), float(drum["friction_coulomb"]), float(drum["friction_viscous"])


def load(path, name, drivetrain_values):
    df = pd.read_csv(path)
    t = df["t"].to_numpy()
    v = df["speed_radial"].to_numpy()
    T = df["tension_tether_ground"].to_numpy()
    u_s = df["input_steering"].to_numpy()
    # cut where the march left the physical branch (it cannot follow the path)
    off = (T < TENSION_FLOOR) | (np.abs(u_s) > STEERING_BOX) | ~np.isfinite(t)
    off[0] = False
    cut = int(np.argmax(off)) if off.any() else len(df)
    t, v, T = t[:cut], v[:cut], T[:cut]
    drum, friction = DRIVETRAIN_PARTS[name]
    m, f_c, c_v = drivetrain_values
    dt_ = Drivetrain(inertia_mass=m if drum else 0.0,
                     friction_coulomb=f_c if friction else 0.0,
                     friction_viscous=c_v if friction else 0.0)
    a = np.r_[0.0, np.diff(v) / np.diff(t)] if len(t) > 1 else np.zeros_like(v)
    shaft = dt_.motor_force(T, v) * v - dt_.inertia_mass * a * v if drum else dt_.motor_force(T, v) * v
    span = t[-1] - t[0] if len(t) > 1 else np.nan
    return dict(
        t=t - t[0], v=v, T=T, shaft=shaft,
        vtau=df["speed_tangential"].to_numpy()[:cut],
        cut=cut < len(df), nodes=cut, total=len(df), duration=span,
        p_tether=np.trapezoid(T * v, t) / span, p_shaft=np.trapezoid(shaft, t) / span,
        t_max=T.max(),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dir", default="results/LEI-V3-KITE/winch_dynamics")
    parser.add_argument("--system", default="data/LEI-V3-KITE/system.yaml")
    args = parser.parse_args()
    folder = Path(args.dir)
    values = system_drivetrain(args.system)

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": "#e4e4e0", "grid.linewidth": 0.6})
    print(f"{'figure':24s} {'case':6s} {'model':10s} {'nodes':>9s} {'time s':>7s} {'P_tether kW':>11s}"
          f" {'P_shaft kW':>10s} {'T max kN':>9s}")
    for key, (title, models) in COMPARISONS.items():
        fig, axs = plt.subplots(len(CASES), 4, figsize=(16, 7.5), constrained_layout=True,
                                squeeze=False)
        for row, (case, case_title) in enumerate(CASES):
            for name, label, color, ls in models:
                path = folder / f"{case}_{name}.csv"
                if not path.exists():
                    continue
                d = load(path, name, values)
                print(f"{key:24s} {case:6s} {name:10s} {d['nodes']:4d}/{d['total']:<4d} {d['duration']:7.1f}"
                      f" {d['p_tether'] / 1e3:11.2f} {d['p_shaft'] / 1e3:10.2f} {d['t_max'] / 1e3:9.2f}")
                kw = dict(color=color, ls=ls, lw=1.5)
                axs[row, 0].plot(d["t"], d["vtau"], label=label, **kw)
                axs[row, 1].plot(d["t"], d["v"], **kw)
                axs[row, 2].plot(d["t"], d["T"] / 1e3, **kw)
                note = f"{d['p_shaft'] / 1e3:.2f} kW"
                if d["cut"]:
                    note += f" (to t = {d['t'][-1]:.0f} s)"
                    for ax, y in zip(axs[row, :3], (d["vtau"][-1], d["v"][-1], d["T"][-1] / 1e3)):
                        ax.plot(d["t"][-1], y, "x", color=color, ms=8, mew=2)
                    axs[row, 2].annotate("can't follow the path", (d["t"][-1], d["T"][-1] / 1e3),
                                         textcoords="offset points", xytext=(6, -12),
                                         fontsize=8, color="#3d3d3a")
                axs[row, 3].plot(d["t"], d["shaft"] / 1e3, label=note, **kw)
            axs[row, 0].set(ylabel="tangential speed v_τ (m s$^{-1}$)", title=f"{case_title}: tangential speed")
            axs[row, 1].set(ylabel="reel speed v_r (m s$^{-1}$)", title="Reel speed")
            axs[row, 2].set(ylabel="tension at ground T (kN)", title="Tether tension")
            axs[row, 3].set(ylabel="shaft power (kW)", title="Shaft power (legend: mean)")
            axs[row, 2].axhline(8.4, color="#6b6a63", lw=0.9, ls=":")
            axs[row, 2].ticklabel_format(axis="y", useOffset=False)
            lo, hi = axs[row, 2].get_ylim()
            if hi - lo < 0.5:  # flat at the law's limit: show it, not 1e-4 kN wiggles
                axs[row, 2].set_ylim(lo - 0.5, hi + 0.5)
            axs[row, 3].axhline(0, color="#6b6a63", lw=0.8)
            axs[row, 3].legend(fontsize=8, frameon=False)
            for ax in axs[row]:
                ax.set_xlabel("time (s)")
        axs[0, 0].legend(fontsize=8, frameon=False)
        fig.suptitle(f"LEI-V3, 9 m/s, forward simulation (no optimisation): {title}", fontsize=11)
        out = folder / f"{key}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(out)


if __name__ == "__main__":
    main()

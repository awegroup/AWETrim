"""Optimised downloop reel-out loops of the two LEI-V3 ROMs in azimuth-elevation.

Reads the runs written by ``downloop_pattern.py --rom <rom> --system <system>``
(results/LEI-V3-KITE/optimization/downloops/<system>_<rom>/) and overlays the
two ROMs per hardware file, with each run's mean power and loop time. The loop
is evaluated from the OPTIMISED path parameters in the saved config (the
exported timeseries carries no path angles after an optimisation).

Usage (project root)
    python scripts/reduced-order-model/optimization/reelout/plot_downloop_rom_comparison.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import casadi as ca

from awetrim.kinematics.parametrized_patterns import create_pattern_from_dict
from awetrim.plotting.plotting import PALETTE, set_plot_style
from awetrim.utils.utils import load_cycle_config_from_yaml

ROOT = Path("results/LEI-V3-KITE/optimization/downloops")
STEM = "depower_downloop_optimized_config_wind_8_z0_0.03_logarithmic_spline"
SYSTEMS = {"optimisation": "Optimisation system.yaml (KCU 8.4 kg)",
           "flown_2019": "Flown 2019 hardware (KCU 22 kg + turbine)"}
ROMS = {
    "semi_empirical": ("Semi-empirical ROM", dict(color=PALETTE["Vermillion"], lw=2.0, ls="--")),
    "aerostructural": ("Aerostructural ROM", dict(color=PALETTE["Blue"], lw=2.0, ls="-")),
    "aerostructural_flight": ("Aerostructural + flight correction",
                              dict(color=PALETTE["Bluish Green"], lw=2.0, ls="-.")),
}


def loop_angles(reelout, n=400):
    """Azimuth and elevation [deg] of the optimised loop over s."""
    pattern = create_pattern_from_dict(reelout["pattern_type"], reelout["path_parameters"])
    sim = reelout["sim_parameters"]
    s_grid = np.linspace(sim["start_angle"], sim["end_angle"], n)
    r = float(reelout["path_parameters"]["r0"])
    az = np.array([float(ca.evalf(pattern.azimuth(r, s))) for s in s_grid])
    el = np.array([float(ca.evalf(pattern.elevation(r, s))) for s in s_grid])
    return np.degrees(az), np.degrees(el)


def main() -> None:
    set_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.2))
    extent = []
    for ax, (system, title) in zip(axes, SYSTEMS.items()):
        for rom, (label, style) in ROMS.items():
            run = ROOT / f"{system}_{rom}"
            reelout, _ = load_cycle_config_from_yaml(run / f"{STEM}.yaml")
            metrics = json.loads((run / f"{STEM}_metrics.json").read_text(encoding="utf-8"))
            az, el = loop_angles(reelout)
            extent.append((az.min(), az.max(), el.min(), el.max()))
            ax.plot(az, el, label=(f"{label}: {metrics['avg_power'] / 1e3:.1f} kW, "
                                   f"loop {metrics['total_time']:.1f} s"), **style)
            ax.plot(az[0], el[0], marker="o", ms=7, color=style["color"],
                    mfc="white", ls="none")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(r"$\phi$ ($^\circ$)")
        ax.set_ylabel(r"$\beta$ ($^\circ$)")
        ax.legend(frameon=False, fontsize=8, loc="upper center")
    # One frame for all four loops, equal degrees on both axes, headroom for
    # the legend above the loops.
    ext = np.array(extent)
    x_lim = (ext[:, 0].min() - 2.0, ext[:, 1].max() + 2.0)
    y_lim = (ext[:, 2].min() - 2.0, ext[:, 3].max() + 7.0)
    for ax in axes:
        ax.set_xlim(*x_lim)
        ax.set_ylim(*y_lim)
        ax.set_aspect("equal", adjustable="box")
    fig.suptitle("Optimised downloop reel-out, 8 m s$^{-1}$ at 100 m, "
                 r"$l_\mathrm{dp}$ bounded to 1.6-2.2 m (open circle: loop start)",
                 fontsize=10)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(ROOT / f"rom_comparison_loops.{ext}", dpi=200)
    print(f"Wrote {ROOT / 'rom_comparison_loops'}.png/.pdf")


if __name__ == "__main__":
    main()

"""Strut bending and span change across the Billow configurations.

Real inflatable struts do not visibly bend in flight. The model only reproduces
that once two inputs are right: the apparent wind speed, and the canopy
stiffness. This draws the three configurations side by side so the two effects
can be told apart, and marks the collapse curvature the tube law is calibrated
up to -- above it the moment has saturated and the tube is hinging, which is
reported rather than enforced.

Usage (from project root):
    python scripts/aerostructural/plot_billow_strut_bending.py
"""

import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import yaml as _yaml

from awetrim import plotting
from awetrim.aerostructural.billow import structural_billow as sb
from awetrim.aerostructural.coupled import read_struc_geometry_yaml as rd
from awetrim.aerostructural.utils import load_yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_billow_geometry import rebuild  # noqa: E402

from common import DEFAULT_KITE_NAME  # noqa: E402

#: (results folder, label) -- the three converged states.
CASES = [
    ("depower_p0000mm_steer_p0000mm_54p_va38", r"$v_a$ 38, canopy 5 kN m$^{-1}$"),
    ("depower_p0000mm_steer_p0000mm", r"$v_a$ 24, canopy 493 kN m$^{-1}$"),
]


def strut_metrics(positions, strut_sections, law_of):
    """Per-strut spanwise station, sagitta as a fraction of chord, utilisation."""
    station, sagitta, utilisation = [], [], []
    for section in strut_sections[1:-1]:      # the two tip stubs are not struts
        index = list(map(int, section))
        start, end = positions[index[0]], positions[index[-1]]
        axis = end - start
        length = float(np.linalg.norm(axis))
        axis = axis / length
        relative = positions[index] - start
        offset = np.linalg.norm(
            relative - np.outer(relative @ axis, axis), axis=1
        ).max()
        laws = [law_of[key] for key in zip(index[:-1], index[1:]) if key in law_of]
        collapse = float(np.mean([law.curvature_collapse for law in laws]))
        station.append(0.5 * (start[1] + end[1]))
        sagitta.append(100.0 * offset / length)
        # Parabolic bend, so kappa = 8 * sagitta / L^2.
        utilisation.append((8.0 * offset / length**2) / collapse)
    order = np.argsort(station)
    return (
        np.asarray(station)[order],
        np.asarray(sagitta)[order],
        np.asarray(utilisation)[order],
    )


def main():
    project = Path(__file__).resolve().parents[2]
    kite = DEFAULT_KITE_NAME
    geometry = load_yaml(project / "data" / kite / "struc_geometry_FEM_full.yaml")
    with (project / "data" / kite / "system.yaml").open(encoding="utf-8") as handle:
        system_config = _yaml.safe_load(handle)
    strut_sections = rd.main(geometry, config={}, system_config=system_config)[8]

    structure = rebuild(project, kite)
    law_of = {
        (int(a), int(b)): law
        for (a, b), law in zip(
            structure.model.element_set(sb.TUBES).connectivity, structure.tube_laws
        )
    }

    plotting.set_plot_style()
    colours = plotting.PALETTE
    styles = [
        (colours["Vermillion"], "o", "--"),
        (colours["Blue"], "s", "-"),
    ]

    figure, axes = plt.subplots(1, 2, figsize=(10.4, 4.1))
    base = project / "results" / kite / "aerostructural" / "billow"

    for (case, label), (colour, marker, dash) in zip(CASES, styles):
        path = base / case / "sim_output.h5"
        if not path.exists():
            print(f"skipping {case}: not on disk")
            continue
        with h5py.File(path, "r") as handle:
            final = np.asarray(handle["tracking/positions"])[-1]
        station, sagitta, utilisation = strut_metrics(final, strut_sections, law_of)

        axes[0].plot(station, sagitta, color=colour, marker=marker, ms=5,
                     ls=dash, lw=1.7, label=label)
        axes[1].plot(station, utilisation, color=colour, marker=marker, ms=5,
                     ls=dash, lw=1.7, label=label)
        print(f"{label:34s} sagitta {sagitta.min():5.2f}-{sagitta.max():5.2f}%  "
              f"utilisation {utilisation.min():.2f}-{utilisation.max():.2f}x")

    axes[0].set_xlabel("spanwise station $y$ (m)")
    axes[0].set_ylabel("strut sagitta (\\% of chord)")
    axes[0].set_title("(a) How far the struts bend")
    axes[0].legend(loc="upper center", fontsize=8)
    axes[0].set_ylim(bottom=0)

    axes[1].axhspan(0, 1, color=colours["Bluish Green"], alpha=0.10)
    axes[1].axhline(1.0, color=colours["Bluish Green"], lw=1.2)
    axes[1].text(0.02, 0.96, "calibrated range", transform=axes[1].transAxes,
                 fontsize=7.5, va="top", color=colours["Bluish Green"])
    axes[1].set_xlabel("spanwise station $y$ (m)")
    axes[1].set_ylabel("curvature / collapse curvature")
    axes[1].set_title("(b) Whether the tube law still applies")
    axes[1].legend(loc="upper center", fontsize=8)
    axes[1].set_ylim(bottom=0)

    figure.tight_layout()
    output = project / "docs" / "billow" / "figures" / "billow_strut_bending.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    print(f"\nwritten to {output}")


if __name__ == "__main__":
    main()

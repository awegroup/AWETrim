"""Depower chains: how the trimmed shape moves with u_dp and apparent speed.

Reads the CSV ``run_chain_depower_BILLOW.py`` writes and draws the four
quantities a depower sweep is run to see: the span the wing keeps, how deeply it
arcs, what it pulls, and what apparent speed it ends up flying at.

Span is drawn as a fraction of the as-built span, since the absolute value is
less useful than the loss: the wing arcs under load and pulls its tips in, and
how much it does that is the structural signature depower is being swept for.

Each wind is one curve. Within a chain the wind is held, so v_a slides with
u_dp -- panel (d) is therefore not a control variable but a readout, and it is
what any later interpolation onto constant v_a has to work from.

Usage (from project root):
    python scripts/aerostructural/studies/plot_billow_depower_chains.py
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from awetrim import plotting
from awetrim.aerostructural.results import aerostructural_results_root
from awetrim.aerostructural.case import DEFAULT_KITE_NAME

def _parse_cell(key, value):
    """One CSV cell: direction stays text, converged is a bool, the rest are floats."""
    if key == "direction":
        return value
    if key == "converged":
        return value == "True"
    return float(value)


def read_chains(csv_path):
    """Rows grouped by wind speed, each sorted by depower input."""
    with Path(csv_path).open(newline="", encoding="utf-8") as handle:
        rows = [
            {k: _parse_cell(k, v) for k, v in row.items()}
            for row in csv.DictReader(handle)
        ]
    winds = sorted({row["wind_speed"] for row in rows})
    grouped = {}
    for wind in winds:
        chain = [row for row in rows if row["wind_speed"] == wind]
        # The two directions share the built point; keep it once.
        seen, unique = set(), []
        for row in sorted(chain, key=lambda r: r["u_dp"]):
            key = round(row["tape_length"], 6)
            if key not in seen:
                seen.add(key)
                unique.append(row)
        grouped[wind] = unique
    return grouped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kite", default=DEFAULT_KITE_NAME)
    parser.add_argument("--built-span", type=float, default=8.202,
                        help="as-built tip-to-tip span [m], for the span fraction")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parents[3]
    root = aerostructural_results_root(project_dir, args.kite) / "billow_depower_chains"
    grouped = read_chains(root / "chains.csv")

    plotting.set_plot_style()
    colours = plotting.PALETTE
    cycle = [colours["Blue"], colours["Vermillion"], colours["Bluish Green"],
             colours["Orange"], colours["Reddish Purple"]]
    markers = ["o", "s", "^", "D", "v"]

    figure, axes = plt.subplots(2, 2, figsize=(10.6, 7.4))
    panels = [
        (axes[0, 0], "span", "span retained, $b/b_0$", "(a) How much span survives"),
        (axes[0, 1], "arch_rise", "arch rise (m)", "(b) How deeply it arcs"),
        (axes[1, 0], "aero_force", "resultant aerodynamic load (N)", "(c) What it pulls"),
        (axes[1, 1], "speed_apparent", "$v_a$ (m s$^{-1}$)", "(d) Apparent speed reached"),
    ]

    for index, (wind, rows) in enumerate(sorted(grouped.items())):
        colour = cycle[index % len(cycle)]
        marker = markers[index % len(markers)]
        u_dp = np.array([r["u_dp"] for r in rows])
        label = f"$v_w$ = {wind:.1f} m s$^{{-1}}$"
        for axis, key, _ylabel, _title in panels:
            values = np.array([r[key] for r in rows])
            if key == "span":
                values = values / args.built_span
            axis.plot(u_dp, values, color=colour, marker=marker, ms=5, lw=1.7,
                      label=label)

    for axis, key, ylabel, title in panels:
        axis.set_xlabel("depower input $u_\\mathrm{dp}$")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.legend(fontsize=8)
    axes[0, 0].axhline(1.0, color="0.6", lw=0.9, ls=":")

    figure.tight_layout()
    output = Path(args.output) if args.output else root / "depower_chains.png"
    figure.savefig(output, dpi=180)

    print(f"{sum(len(v) for v in grouped.values())} points, "
          f"{len(grouped)} winds")
    for wind, rows in sorted(grouped.items()):
        u_dp = np.array([r["u_dp"] for r in rows])
        span = np.array([r["span"] for r in rows]) / args.built_span
        va = np.array([r["speed_apparent"] for r in rows])
        print(f"  v_w {wind:4.1f}: u_dp {u_dp.min():.3f}-{u_dp.max():.3f}, "
              f"v_a {va.min():5.2f}-{va.max():5.2f}, "
              f"span {100*span.min():5.2f}-{100*span.max():5.2f}% of built")
    print(f"\nwritten to {output}")


if __name__ == "__main__":
    main()

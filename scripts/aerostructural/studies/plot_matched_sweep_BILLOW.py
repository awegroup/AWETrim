"""Figures of the matched-sweep chains (run_matched_sweep_BILLOW.py).

``matched_sweep_shapes``  the solved full-model kite at one apparent speed:
    three depower settings seen ACROSS THE SPAN, where the wing's pitch about
    the bridle point is what depower changes, over three steering settings seen
    FROM THE FRONT, where the asymmetry is. Canopy triangles are coloured by the
    membrane's own regime -- slack, wrinkled, taut -- and drawn back to front.

``matched_sweep_curves``  what the sweeps say against the paper's model: the
    tether force at the two matched depowers against apparent speed, with the
    2019 near-straight flight medians when ``--comparison`` is given, and the
    turn law.

``--comparison`` is a JSON of the paper's layers (wireframe chains, flight
medians and turn cloud); without it only the full-model curves are drawn.

Usage (from project root):
    python scripts/aerostructural/studies/plot_matched_sweep_BILLOW.py --va 19
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection, PolyCollection

from awetrim import plotting
from awetrim.aerostructural.billow import structural_billow as sb
from billow.elements.membrane import membrane_regimes
from awetrim.aerostructural.case import DEFAULT_KITE_NAME

from check_mirror_asymmetry import final_positions
from plot_billow_geometry import rebuild

REGIME_COLOURS = ("#3A9DC5", "#E69F00", "#D55E00")   # slack, wrinkled, taut
UDP_REELOUT, UDP_REELIN = 0.242, 0.323


def load_rows(root):
    rows = []
    for path in sorted(root.glob("*/*/row.json")):
        if path.parent.name.startswith("_"):   # set aside by hand
            continue
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("status") in ("ok", "off_target", "plateau"):
            row["folder"] = path.parent
            rows.append(row)
    return rows


def find(rows, mode, va, udp=None, us=None):
    for row in rows:
        if row["mode"] != mode or row["target_va"] != va:
            continue
        if udp is not None and abs(row["udp"] - udp) > 6e-4:
            continue
        if us is not None and abs(row["us"] - us) > 1e-6:
            continue
        return row
    return None


def draw_view(ax, structure, row, title, view="front", note=True):
    """One solved shape. ``front`` looks along the flight direction (y, z),
    ``side`` across the span (x, z) -- which is where depower shows, as the
    whole wing pitches about the bridle point."""
    canopy = structure.model.element_set(sb.CANOPY)
    triangles = np.asarray(canopy.connectivity, dtype=int)
    positions = final_positions(row["folder"])
    regime = np.asarray(membrane_regimes(positions, canopy)["regime"], dtype=int)
    plane = [1, 2] if view == "front" else [0, 2]
    grid = np.asarray(structure.grid, dtype=int)
    if view == "front":
        # Front: the canopy itself, painter-sorted along the flight direction.
        order = np.argsort(-positions[triangles, 0].mean(axis=1))
        ax.add_collection(PolyCollection(positions[triangles[order]][:, :, plane],
                                         facecolors=[REGIME_COLOURS[r] for r in regime[order]],
                                         edgecolors="white", linewidths=0.15))
    else:
        # Across the span the canopy projects onto itself and reads as a blob.
        # Draw what depower actually moves: every rib's chord, the centre one
        # picked out, with the apparent wind along the horizontal.
        for row_nodes in grid:
            chord = positions[row_nodes][:, plane]
            ax.plot(chord[:, 0], chord[:, 1], "-", color="#C7CCD6", lw=0.7, zorder=1)
        centre = positions[grid[len(grid) // 2]][:, plane]
        ax.plot(centre[:, 0], centre[:, 1], "-", color="#D55E00", lw=2.2, zorder=3,
                solid_capstyle="round")
        span = np.ptp(positions[np.unique(grid)][:, plane[0]])
        y_arrow = positions[np.unique(grid)][:, plane[1]].min()
        ax.annotate("", xy=(centre[:, 0].min() - 0.1 * span, y_arrow),
                    xytext=(centre[:, 0].min() - 0.45 * span, y_arrow),
                    arrowprops=dict(arrowstyle="-|>", color="#5B6478", lw=1.1))
        ax.text(centre[:, 0].min() - 0.27 * span, y_arrow, r"$v_a$", color="#5B6478",
                fontsize=9, ha="center", va="bottom")
    for name, colour, width in ((sb.TUBES, "#23293A", 1.3), (sb.CABLES, "#7F8BA3", 0.4)):
        elements = np.asarray(structure.model.element_set(name).connectivity, dtype=int)
        ax.add_collection(LineCollection(positions[elements][:, :, plane], colors=colour,
                                         linewidths=width, zorder=2))
    pulleys = np.asarray(structure.model.element_set(sb.PULLEYS).connectivity, dtype=int)
    for arm in (0, 1):
        ax.add_collection(LineCollection(positions[pulleys[:, [arm, arm + 1]]][:, :, plane],
                                         colors="#7F8BA3", linewidths=0.4, zorder=2))
    ax.set_title(title, fontsize=10)
    if note:
        parts = []
        if view == "front":
            parts.append(rf"{100 * np.mean(regime == 1):.0f}\% wrinkled")
        parts.append(rf"$\alpha$ {row['aoa_deg']:.1f}$^\circ$")
        parts.append(rf"$F_\mathrm{{t}}$ {row['tether_force'] / 1e3:.2f} kN")
        if row["us"] > 0:
            parts.append(rf"$\dot\chi$ {row['course_rate']:+.2f} rad s$^{{-1}}$")
        ax.text(0.5, -0.04, ", ".join(parts), transform=ax.transAxes, ha="center",
                va="top", fontsize=8.5, color="#5B6478")
    # Frame the WING, not the bridle fan: the canopy is what the figure is about.
    wing = positions[np.unique(structure.grid)]
    lo, hi = wing[:, plane].min(axis=0), wing[:, plane].max(axis=0)
    pad = 0.12 * max(hi - lo)
    ax.set_aspect("equal")
    ax.set_xlim(lo[0] - pad, hi[0] + pad)
    ax.set_ylim(lo[1] - pad, hi[1] + pad)
    ax.axis("off")


def shapes_figure(rows, structure, va, path, udp_steering, paper=False):
    """Depower in side view over steering in front view, at one apparent speed.

    ``paper`` drops the figure title (the caption carries it) and the
    "paper's match" tags on the depower panels."""
    solved = sorted({r["udp"] for r in rows if r["mode"] == "depower" or r["us"] < 1e-9})
    wanted = [u for u in (UDP_REELOUT, 0.283, UDP_REELIN) if any(abs(u - s) < 6e-4 for s in solved)]
    depower = [(u, find(rows, "depower", va, udp=u)) for u in wanted]
    steering = [(s, find(rows, "steering", va, udp=udp_steering, us=s)) for s in (0.0, 0.125, 0.25)]
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.4))
    for ax, (u, row) in zip(axes[0], depower):
        if row is None:
            ax.axis("off")
            continue
        label = "" if paper else {UDP_REELOUT: " (paper's reel-out match)",
                                  UDP_REELIN: " (paper's reel-in match)"}.get(u, "")
        draw_view(ax, structure, row, rf"$u_\mathrm{{dp}}$ = {u:.3f}{label}", view="side")
    for ax, (st, row) in zip(axes[1], steering):
        if row is None:
            ax.axis("off")
            continue
        draw_view(ax, structure, row, rf"$u_\mathrm{{s}}$ = {st:.3f} m", view="front")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in REGIME_COLOURS]
    fig.legend(handles, ["slack", "wrinkled", "taut"], loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.0))
    if not paper:
        fig.suptitle(rf"LEI V3 on Billow's full model at $v_a$ = {va} m s$^{{-1}}$: depower across "
                     rf"the span (top), steering at $u_\mathrm{{dp}}$ = {udp_steering:.3f} from the "
                     r"front (bottom)", fontsize=11.5, y=0.99)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95 if not paper else 1.0), h_pad=4.0)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"written {path}")


def curves_figure(rows, comparison, va, path, udp_steering, paper=False):
    fig, (ax_f, ax_t) = plt.subplots(1, 2, figsize=(10.5, 4.2))
    colours = {"reel-out": plotting.PALETTE["Sky Blue"], "reel-in": plotting.PALETTE["Orange"]}
    for phase, udp in (("reel-out", UDP_REELOUT), ("reel-in", UDP_REELIN)):
        colour = colours[phase]
        full = [r for r in rows if abs(r["udp"] - udp) < 6e-4 and r["us"] < 1e-9]
        full = sorted({r["target_va"]: r for r in full}.values(), key=lambda r: r["va"])
        ax_f.plot([r["va"] for r in full], [r["tether_force"] / 1e3 for r in full], "-o", color=colour,
                  lw=2 if phase == "reel-in" else 1.1, ms=4 if phase == "reel-in" else 2.5,
                  alpha=1 if phase == "reel-in" else 0.55, label=rf"full, $u_\mathrm{{dp}}$ = {udp:.3f}")
        if phase == "reel-out":
            own = [r for r in rows if abs(r["udp"] - udp_steering) < 6e-4 and r["us"] < 1e-9]
            own = sorted({r["target_va"]: r for r in own}.values(), key=lambda r: r["va"])
            ax_f.plot([r["va"] for r in own], [r["tether_force"] / 1e3 for r in own], "-o", color=colour,
                      lw=2, ms=4, label=rf"full, $u_\mathrm{{dp}}$ = {udp_steering:.3f}" + ("" if paper else " (its match)"))
        if comparison:
            flight = comparison["flight"][phase]
            curve = [p for p in flight["wireframe_curve"] if not p["extrapolated"]]
            ax_f.plot([p["v_a"] for p in curve], [p["f_t"] / 1e3 for p in curve], "--", color=colour, lw=1.4,
                      label=rf"wireframe, $u_\mathrm{{dp}}$ = {udp:.3f}")
            ax_f.plot(flight["medians_va"], flight["medians_ft_kN"], "o" if phase == "reel-out" else "^",
                      color=colour, mec="white", ms=7, ls="", label=f"2019 {phase} flight medians")
    ax_f.set_xlabel(r"$v_a$ (m s$^{-1}$)")
    ax_f.set_ylabel(r"$F_\mathrm{t}$ (kN)")
    ax_f.set_ylim(bottom=0)
    ax_f.legend(fontsize=8, frameon=False)
    ax_f.grid(alpha=0.25)

    steer = [r for r in rows if r["mode"] == "steering" and r["us"] > 0 and abs(r["udp"] - udp_steering) < 6e-4]
    x = np.array([r["va"] * r["us"] for r in steer])
    y = np.array([r["course_rate"] for r in steer])
    if comparison:
        turn = comparison["turn"]["reel-out"]
        ax_t.scatter(turn["x"], turn["y"], s=2, color=colours["reel-out"], alpha=0.08, lw=0, rasterized=True)
        med = np.asarray(turn["medians"])
        ax_t.plot(med[:, 0], med[:, 1], "o", color=colours["reel-out"], mec="white", ms=7, ls="",
                  label=rf"2019 reel-out flight, gain {turn['gain']:.3f}")
        wire = [r for r in comparison["wireframe_steering"] if r["us"] > 0]
        xw = np.array([r["v_a"] * r["us"] for r in wire])
        yw = np.array([r["chi_dot_turn"] for r in wire])
        gw = float(np.sum(xw * yw) / np.sum(xw * xw))
        ax_t.plot(xw, yw, "o", mfc="none", mec="#8A8F99", ms=4, ls="", label=rf"wireframe, $u_\mathrm{{dp}}$ = 0.242, gain {gw:.3f}")
    stalled = np.array([(r.get("n_stalled_panels") or 0) > 0 for r in steer], dtype=bool)
    if len(x):
        gain = float(np.sum(x * y) / np.sum(x * x))
        ax_t.plot(x[~stalled], y[~stalled], "o", color="#1D2433", ms=4.5, ls="", label=rf"full, $u_\mathrm{{dp}}$ = {udp_steering:.3f}, gain {gain:.3f}")
        ax_t.plot(x[stalled], y[stalled], "o", mfc="white", mec="#1D2433", ms=4.5, ls="", label="full, part of span stalled")
        grid = np.linspace(0, x.max() * 1.05, 10)
        ax_t.plot(grid, gain * grid, "-", color="#1D2433", lw=1)
    ax_t.set_xlabel(r"$v_a\,u_\mathrm{s}$ (m$^2$ s$^{-1}$)")
    ax_t.set_ylabel(r"$\dot\chi$ (rad s$^{-1}$)")
    ax_t.set_xlim(left=0)
    ax_t.set_ylim(bottom=0)
    ax_t.legend(fontsize=8, frameon=False)
    ax_t.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"written {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default="billow_matched_sweep")
    parser.add_argument("--va", type=float, default=19)
    parser.add_argument("--comparison", default=None)
    parser.add_argument("--udp-steering", type=float, default=0.315,
                        help="depower the steering chains were solved at")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--kite", default=DEFAULT_KITE_NAME)
    parser.add_argument("--paper", action="store_true",
                        help="no figure title, no 'paper's match' tags, and a PDF next to each PNG")
    args = parser.parse_args()

    project = Path(__file__).resolve().parents[3]
    root = project / "results" / args.kite / "aerostructural" / args.root
    out = Path(args.output_dir) if args.output_dir else root
    out.mkdir(parents=True, exist_ok=True)
    plotting.set_plot_style()
    rows = load_rows(root)
    va = int(args.va) if float(args.va).is_integer() else args.va
    structure = rebuild(project, args.kite, 2, {"canopy_pattern": "cross", "canopy_refinement": 1})
    comparison = json.loads(Path(args.comparison).read_text(encoding="utf-8")) if args.comparison else None
    for ext in (("png", "pdf") if args.paper else ("png",)):
        shapes_figure(rows, structure, va, out / f"matched_sweep_shapes.{ext}", args.udp_steering, args.paper)
        curves_figure(rows, comparison, va, out / f"matched_sweep_curves.{ext}", args.udp_steering, args.paper)


if __name__ == "__main__":
    main()

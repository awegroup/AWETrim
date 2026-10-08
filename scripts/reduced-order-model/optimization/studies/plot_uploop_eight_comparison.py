"""Condensed comparison of the three free-v_r pumping cycles of the up-loop study.

Normal cycle (figure eights + separate reel-in, ``--baseline-seed``) vs the
slanted up-loop eight vs the symmetric side-climb eight (``--shape symmetric
--symmetric``), all solved by ``run_uploop_eight_opti.py`` on the same bounds.
Writes three figures next to the results:

- ``three_cycle_trajectories_<tag>.png``: azimuth/elevation, coloured by v_r,
  initial guess dashed;
- ``three_cycle_trajectories_3d_<tag>.png``: the same in metres (solved r);
- ``three_cycle_timeseries_<tag>.png``: P, F_t, v_r, u_s, u_p, t = 0 at the
  start of reel-out.

Run from the repo root after the three driver runs:

    python scripts/reduced-order-model/optimization/studies/plot_uploop_eight_comparison.py
"""
import argparse
from pathlib import Path

import casadi as ca
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from awetrim.kinematics.parametrized_patterns import create_pattern_from_dict
from awetrim.plotting import PALETTE
from awetrim.utils.control_metrics import count_steering_reversals

_parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
_parser.add_argument("--kite", default="LEI-V3-KITE")
_parser.add_argument("--tag", default="wind_10at100m_logarithmic_z0_0.03",
                     help="wind tag of the driver outputs")
_args = _parser.parse_args()

REPO = Path(__file__).resolve().parents[4]
RES = REPO / "results" / _args.kite / "optimization" / "uploop_eight"
TAG = _args.tag
CASES = [  # (label, file stem, colour = categorical slots 1-3)
    ("Normal cycle", "baseline", PALETTE["Black"]),
    ("Slanted up-loop eight", "uploop_eight", PALETTE["Orange"]),
    ("Symmetric eight (mirror)", "symmetric_eight_mirror", PALETTE["Bluish Green"]),
]
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3dc"


def load(stem):
    ro = yaml.safe_load(open(RES / f"{stem}_optimized_{TAG}.yaml"))["reelout"]
    pp = ro["path_parameters"]
    d = np.genfromtxt(RES / f"{stem}_optimized_{TAG}.csv", delimiter=",", names=True)
    pat = create_pattern_from_dict(ro["pattern_type"], pp)
    s = d["s"]
    r = d["distance_radial"]
    az = np.array([float(ca.DM(pat.azimuth(ri, si))) for ri, si in zip(r, s)])
    el = np.array([float(ca.DM(pat.elevation(ri, si))) for ri, si in zip(r, s)])
    # Left-rule time incl. the seam interval (as the NLP integrates it).
    s_next = np.append(s[1:], pp["s_final"])
    dt = (s_next - s) / d["s_dot"]
    t = np.concatenate([[0.0], np.cumsum(dt)])
    vr, T = d["speed_radial"], d["tension_tether_ground"]
    P = T * vr
    p_mean = np.sum(P * dt) / t[-1]

    # Roll so node 0 = first reel-out node after a reel-in (t = 0 at reel-out start).
    up = np.where((vr > 0) & (np.roll(vr, 1) <= 0))[0]
    k0 = int(up[0]) if len(up) else 0
    vr, T, P, dt = (np.roll(x, -k0) for x in (vr, T, P, dt))
    az, el, r = np.roll(az, -k0), np.roll(el, -k0), np.roll(r, -k0)
    us, up_ = np.roll(d["input_steering"], -k0), np.roll(d["input_depower"], -k0)
    t = np.concatenate([[0.0], np.cumsum(dt)])

    def closed(x):  # append node 0 at t = T so lines close the period
        return np.append(x, x[0])

    return dict(
        t=t, az=closed(np.degrees(az)), el=closed(np.degrees(el)), vr=closed(vr),
        T=closed(T) / 1e3, P=closed(P) / 1e3, us=closed(us),
        up=closed(up_), p_mean=p_mean / 1e3, dur=t[-1],
        rev=count_steering_reversals(d["input_steering"], periodic=True),
        r0=d["distance_radial"][0], tmax=T.max() / 1e3, r=closed(r),
    )


SEEDS = {  # the baseline seed is the stored flight-fitted cycle (the driver fairs it by <= 0.76 deg)
    "baseline": REPO / "data" / _args.kite / "cycle_configs" / "full_cycle_periodic_from_exp.yaml",
    "uploop_eight": RES / "uploop_eight_seed.yaml",
    "symmetric_eight_mirror": RES / "symmetric_eight_mirror_seed.yaml",
}


def seed_path(stem, n=600):
    ro = yaml.safe_load(open(SEEDS[stem]))
    ro = ro.get("reelout", ro)
    pp = ro["path_parameters"]
    r0 = ro.get("radial_parameters", {}).get("r0", pp.get("r0"))
    pat = create_pattern_from_dict(ro["pattern_type"], pp)
    s = np.linspace(pp.get("s_init", 0.0), pp.get("s_final", 1.0), n)
    az = np.array([float(ca.DM(pat.azimuth(r0, si))) for si in s])
    el = np.array([float(ca.DM(pat.elevation(r0, si))) for si in s])
    return np.degrees(az), np.degrees(el), r0


data = [(lbl, col, load(stem)) for lbl, stem, col in CASES]
seeds = [seed_path(stem) for _, stem, _ in CASES]

plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False})
SUPTITLE = (r"LEI-V3 pumping cycles at 10 m s$^{-1}$ @ 100 m (log, $z_0$ = 0.03 m), "
            r"free $v_r$, shared bounds (height 40-400 m, $R_{turn}\geq$25 m)")
fig = plt.figure(figsize=(13, 4.6), facecolor="white")
gs = fig.add_gridspec(1, 3, wspace=0.18)

# Each side of zero gets its own range, so reel-out (0..~2.5) uses the full warm ramp.
norm = TwoSlopeNorm(vcenter=0.0, vmin=min(d["vr"].min() for *_, d in data),
                    vmax=max(d["vr"].max() for *_, d in data))
cmap = LinearSegmentedColormap.from_list(  # AWETrim palette, grey (visible) midpoint
    "vr_awetrim", ["#14223F", PALETTE["Blue"], PALETTE["Sky Blue"], "#A8A8A8",
                   PALETTE["Yellow"], PALETTE["Orange"], PALETTE["Vermillion"], "#6E2500"])
az_lim = (min(d["az"].min() for *_, d in data) - 4, max(d["az"].max() for *_, d in data) + 4)
el_lim = (0, max(d["el"].max() for *_, d in data) + 5)

for j, (lbl, col, d) in enumerate(data):
    ax = fig.add_subplot(gs[0, j])
    pts = np.column_stack([d["az"], d["el"]]).reshape(-1, 1, 2)
    lc = LineCollection(np.concatenate([pts[:-1], pts[1:]], axis=1), cmap=cmap, norm=norm,
                        linewidths=2.2)
    lc.set_array(0.5 * (d["vr"][:-1] + d["vr"][1:]))
    ax.plot(*seeds[j][:2], color=MUTED, lw=1.0, ls="--", zorder=1,
            label="initial guess" if j == 0 else None)
    ax.add_collection(lc)
    if j == 0:
        ax.legend(loc="upper right", frameon=False, fontsize=7, handlelength=2.2)
    # direction arrows
    for k in np.linspace(0, len(d["az"]) - 2, 7, dtype=int)[1:-1]:
        ax.annotate("", xy=(d["az"][k + 1], d["el"][k + 1]), xytext=(d["az"][k], d["el"][k]),
                    arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.8, mutation_scale=9))
    ax.plot(d["az"][0], d["el"][0], "o", ms=5, mfc="white", mec=INK, zorder=5)
    el_floor = np.degrees(np.arcsin(40.0 / d["r0"]))
    ax.axhline(el_floor, color=MUTED, lw=0.8, ls=":")
    ax.text(az_lim[1] - 1, el_floor + 1, "z = 40 m", ha="right", color=MUTED, fontsize=7)
    ax.set_xlim(az_lim); ax.set_ylim(el_lim); ax.set_aspect("equal")
    ax.grid(color=GRID, lw=0.6)
    ax.set_xlabel(r"Azimuth ($^\circ$)")
    if j == 0:
        ax.set_ylabel(r"Elevation ($^\circ$)")
    ax.set_title(lbl, color=col, fontweight="bold", fontsize=10, loc="left", pad=16)
    ax.text(0.0, 1.01,
            f"$\\bar P$ = {d['p_mean']:.2f} kW  |  {d['rev']} reversals  |  "
            f"{d['dur']:.0f} s  |  $r_0$ = {d['r0']:.0f} m",
            transform=ax.transAxes, va="bottom", fontsize=8, color=INK)

cax = fig.add_axes([0.92, 0.2, 0.01, 0.6])
cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
cb.set_label(r"$v_r$ (m s$^{-1}$)"); cb.outline.set_visible(False)

fig.suptitle(SUPTITLE, x=0.08, y=1.04, ha="left", fontsize=11, color=INK)
out = RES / f"three_cycle_trajectories_{TAG}.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(out)

fig = plt.figure(figsize=(12, 9), facecolor="white")
gs = fig.add_gridspec(5, 1, hspace=0.15)
rows = [("P", r"$P$ (kW)", None), ("T", r"$F_t$ (kN)", (0.865, 8.4)),
        ("vr", r"$v_r$ (m s$^{-1}$)", (-8, 3.5)), ("us", r"$u_s$", (-0.175, 0.175)),
        ("up", r"$u_p$ (m)", (1.7, 2.1))]
axes = []
for i, (key, ylabel, band) in enumerate(rows):
    ax = fig.add_subplot(gs[i, 0], sharex=axes[0] if axes else None)
    axes.append(ax)
    for lbl, col, d in data:
        ax.plot(d["t"], d[key], color=col, lw=1.6, label=lbl)
        if key == "P":
            ax.axhline(d["p_mean"], color=col, lw=0.9, ls="--")
    if band is not None:
        for b in band:
            ax.axhline(b, color=MUTED, lw=0.7, ls=":")
    if key in ("P", "vr", "us"):
        ax.axhline(0, color=MUTED, lw=0.6)
    ax.set_ylabel(ylabel); ax.grid(color=GRID, lw=0.6)
    if i < len(rows) - 1:
        plt.setp(ax.get_xticklabels(), visible=False)
axes[-1].set_xlabel("Time from start of reel-out (s)")
fig.legend(*axes[0].get_legend_handles_labels(), ncol=3, frameon=False, loc="upper left",
           bbox_to_anchor=(0.08, 0.935), fontsize=9)
axes[0].text(1.0, 1.02, "dashed: cycle mean", transform=axes[0].transAxes, ha="right",
             fontsize=7, color=MUTED)

fig.suptitle(SUPTITLE, x=0.08, y=0.97, ha="left", fontsize=11, color=INK)
out = RES / f"three_cycle_timeseries_{TAG}.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(out)
for lbl, _, d in data:
    print(f"{lbl:28s} P={d['p_mean']:.2f} kW rev={d['rev']} T={d['dur']:.1f} s r0={d['r0']:.0f} Tmax={d['tmax']:.2f} kN")


# --- 3-D view: x downwind, y crosswind, z up; r from the solution (seed at its r0) ---
def xyz(az_deg, el_deg, r):
    a, e = np.radians(az_deg), np.radians(el_deg)
    return r * np.cos(e) * np.cos(a), r * np.cos(e) * np.sin(a), r * np.sin(e)


fig = plt.figure(figsize=(14, 5.2), facecolor="white")
lim_xy = max(np.abs(xyz(d["az"], d["el"], d["r"])[1]).max() for *_, d in data) * 1.1
lim_x = max(xyz(d["az"], d["el"], d["r"])[0].max() for *_, d in data) * 1.05
lim_z = max(xyz(d["az"], d["el"], d["r"])[2].max() for *_, d in data) * 1.05
for j, (lbl, col, d) in enumerate(data):
    ax = fig.add_subplot(1, 3, j + 1, projection="3d")
    x, y, z = xyz(d["az"], d["el"], d["r"])
    segs = np.stack([np.column_stack([x[:-1], y[:-1], z[:-1]]),
                     np.column_stack([x[1:], y[1:], z[1:]])], axis=1)
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    lc = Line3DCollection(segs, cmap=cmap, norm=norm, linewidths=2.2)
    lc.set_array(0.5 * (d["vr"][:-1] + d["vr"][1:]))
    sa, se, sr0 = seeds[j]
    ax.plot(*xyz(sa, se, sr0), color=MUTED, lw=1.0, ls="--", label="initial guess")
    ax.add_collection3d(lc)
    ax.plot(x[0], y[0], z[0], "o", ms=5, mfc="white", mec=INK)
    # tether to the reel-out start and ground footprint for depth cues
    ax.plot([0, x[0]], [0, y[0]], [0, z[0]], color=MUTED, lw=0.6)
    ax.plot(x, y, np.zeros_like(z), color=GRID, lw=0.8)
    ax.plot([0], [0], [0], "s", color=INK, ms=4)
    ax.set_xlim(0, lim_x); ax.set_ylim(-lim_xy, lim_xy); ax.set_zlim(0, lim_z)
    ax.set_box_aspect((lim_x, 2 * lim_xy, lim_z))
    ax.view_init(elev=18, azim=-128)
    ax.set_xlabel("x, downwind (m)", fontsize=8, labelpad=4)
    ax.set_ylabel("y (m)", fontsize=8)
    ax.set_zlabel("z (m)", fontsize=8)
    ax.tick_params(labelsize=7)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_facecolor("white"); pane.set_edgecolor(GRID)
    ax.set_title(f"{lbl}\n$\\bar P$ = {d['p_mean']:.2f} kW  |  {d['rev']} reversals",
                 color=col, fontsize=10, loc="left")
    if j == 0:
        ax.legend(loc="upper right", frameon=False, fontsize=7)
cax = fig.add_axes([0.93, 0.2, 0.01, 0.55])
cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
cb.set_label(r"$v_r$ (m s$^{-1}$)"); cb.outline.set_visible(False)
fig.suptitle(SUPTITLE, x=0.06, y=0.98, ha="left", fontsize=11, color=INK)
out = RES / f"three_cycle_trajectories_3d_{TAG}.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(out)

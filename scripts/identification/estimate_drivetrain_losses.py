"""Bottom-up estimate of the mechanical drivetrain losses of the TU Delft GS1
ground station (2010-2011), component by component, from its hardware data
(not from flight-log identification), evaluated on optimised LEI-V3 pumping
cycles. Feeds docs/winch/drivetrain_physics.tex.

Tether path (outlet -> drum): pulley P1 (wrap = elevation angle), pulley P2
(90 deg), pulley P3 (90 deg, carries the force sensor), straight span, drum,
synchronous belt, gearbox, motor-generator. All pulleys D = 0.095 m; drum
D = 0.323 m; overall ratio n = 6.2; tether 4 mm braided Dyneema.

Every loss is written as a tether-referred force F_i(v, T) >= 0 opposing the
motion, so the power lost is F_i |v|:
  * pulley bearing      F = mu_b * 2 T sin(theta/2) * (d_b / D)
  * tether bending      F = c_b * (d / D) * T              per full bend+unbend
  * drum bending        F = 0.5 * c_b * (d / D_drum) * T   (one half-cycle)
  * drum bearings       F = mu_b * T * (d_b / D_drum)
  * belt                F = (1 - eta_belt) * T             (load-dependent)
  * gear mesh           F = (1 - eta_mesh) * T             (load-dependent)
  * no-load drivetrain  F = F_c + c_v |v|                  (seals, churning,
    belt pretension, motor bearings + fan; the cold-day tow test on GS1,
    Fechner 2016 Tab. 4.1: 122 N + 30.7 N s/m, one value, no range)

The load-dependent parameters have a low/high bound; low/high totals bracket
the estimate.
Also draws the explanatory figures of the report (power chain, scaling
classes, axle force, pulley physics, efficiency map, cycle time series and
waterfall, tow test + oil viscosity) and the GS1 energy balance of Fechner's
AWEC 2011 talk with an illustrative what-if of the cheap improvements.
Run from the project root:
    python scripts/identification/estimate_drivetrain_losses.py
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --- hardware (known) ---------------------------------------------------------
D_PULLEY = 0.095        # m, all three pulleys
D_DRUM = 0.323          # m
D_TETHER = 0.004        # m
GEAR_RATIO = 6.2
ELEVATION_NOMINAL = np.radians(30.0)  # drawing; P1 wrap follows the elevation
WRAPS = {"P1": ELEVATION_NOMINAL, "P2": np.pi / 2, "P3": np.pi / 2}

# --- physics parameters (assumed ranges, low / high) -----------------------------
# Rolling-bearing friction coefficient (SKF catalogue: ~0.0015 deep-groove ball,
# 0.0018 spherical/tapered roller; sealed bearings higher) and bore-to-sheave ratio.
MU_BEARING = (0.0015, 0.003)
BORE_RATIO_PULLEY = 0.2           # d_b / D for a 0.095 m sheave (~20 mm bore)
BORE_RATIO_DRUM = 0.2             # d_b / D_drum (~65 mm journal)
# Bending loss of a braided fibre rope over a sheave, dT = c_b (d/D) T per
# bend + unbend (internal strand friction + hysteresis). c_b 0.05-0.2 puts one
# sheave at d/D = 0.042 at 0.2-0.8 % of the tension.
C_BEND = (0.05, 0.2)
# Gearbox stage, mesh (load-dependent) loss fraction.
MESH_LOSS = (0.01, 0.02)
# Synchronous (toothed) belt, load-dependent loss fraction (efficiency 97-99 %).
BELT_LOSS = (0.01, 0.03)
# No-load drivetrain friction, tether-referred (F_c N, c_v N s/m): the GS1 tow
# test (Fechner 2016 Tab. 4.1), the same for both bounds.
NO_LOAD = (122.0, 30.7)

OUT_FIG = Path("docs/winch/figures")
OUT_TEX = Path("docs/winch/generated")
CYCLE_CSV = "results/LEI-V3-KITE/optimization/uploop_eight/baseline_optimized_wind_*.csv"


def components(T, v, bound: str) -> dict[str, np.ndarray]:
    """Tether-referred loss forces per component (N) at tension T (N), speed v."""
    i = 0 if bound == "low" else 1
    T = np.asarray(T, float); v = np.asarray(v, float)
    out = {}
    for name, theta in WRAPS.items():
        bearing = MU_BEARING[i] * 2 * T * np.sin(theta / 2) * BORE_RATIO_PULLEY
        bending = C_BEND[i] * (D_TETHER / D_PULLEY) * T
        out[name] = bearing + bending
    out["drum"] = (0.5 * C_BEND[i] * (D_TETHER / D_DRUM) * T
                   + MU_BEARING[i] * T * BORE_RATIO_DRUM)
    out["belt"] = BELT_LOSS[i] * T
    out["gear mesh"] = MESH_LOSS[i] * T
    Fc, cv = NO_LOAD
    out["no-load"] = Fc + cv * np.abs(v) + 0 * T
    return out


def split(comp: dict) -> tuple[np.ndarray, np.ndarray]:
    """(upstream of the sensor, downstream). The P3 axle reads the mean of its
    in- and out-going tension, so half of P3's own loss is on each side."""
    up = comp["P1"] + comp["P2"] + 0.5 * comp["P3"]
    down = (0.5 * comp["P3"] + comp["drum"] + comp["belt"] + comp["gear mesh"]
            + comp["no-load"])
    return up, down


OPERATING_POINTS = [
    ("Moderate reel-out", 2800.0, 1.1),
    ("Optimised reel-out", 7400.0, 1.75),
    ("Typical reel-in", 1000.0, -3.5),
    ("Fast reel-in", 1000.0, -8.0),
]
CYCLES_GLOB = "results/LEI-V3-KITE/optimization/uploop_eight/*_optimized_wind_*.csv"
CYCLE_LABELS = {"baseline": "baseline (multi-lobe)", "uploop_eight": "up-loop eight",
                "symmetric_eight": "symmetric eight",
                "symmetric_eight_mirror": "symmetric eight, mirrored"}


def operating_point_table() -> pd.DataFrame:
    rows = []
    for label, T, v in OPERATING_POINTS:
        for bound in ("low", "high"):
            c = components(T, v, bound)
            up, down = split(c)
            rows.append(dict(point=label, bound=bound, T=T, v=v,
                             **{k: float(x) for k, x in c.items()},
                             upstream=float(up), downstream=float(down),
                             total=float(up + down)))
    return pd.DataFrame(rows)


def _cycle(path: str):
    """(dt, v, T) of a saved periodic optimum on the optimiser's left rule,
    seam interval included."""
    d = pd.read_csv(path)
    s, sd = d.s.values, d.s_dot.values
    dt = np.diff(np.r_[s, s[0] + (s[-1] - s[0]) * len(s) / (len(s) - 1)]) / sd
    return dt, d.speed_radial.values, d.tension_tether_ground.values


def cycles_power() -> pd.DataFrame:
    """Cycle-mean mechanical loss power for every saved tether-power optimum."""
    rows = []
    for f in sorted(glob.glob(CYCLES_GLOB)):
        name = Path(f).name.split("_optimized")[0]
        if "stage" in f or name not in CYCLE_LABELS:
            continue
        dt, v, T = _cycle(f)
        D, E = dt.sum(), np.sum(T * v * dt)
        ro = v > 0
        row = dict(cycle=CYCLE_LABELS[name], duration_s=D, P_tether_kW=E / D / 1e3)
        for bound in ("low", "high"):
            c = components(np.abs(T), v, bound)
            up, down = split(c)
            P = (up + down) * np.abs(v)
            row[f"loss_kW_{bound}"] = np.sum(P * dt) / D / 1e3
            row[f"loss_pct_{bound}"] = 100 * np.sum(P * dt) / E
            row[f"shaft_kW_{bound}"] = (E - np.sum(P * dt)) / D / 1e3
            row[f"reelin_share_{bound}"] = 100 * np.sum((P * dt)[~ro]) / np.sum(P * dt)
            row[f"peak_kW_{bound}"] = P.max() / 1e3
        rows.append(row)
    order = list(CYCLE_LABELS.values())
    return pd.DataFrame(rows).sort_values("cycle", key=lambda s: s.map(order.index))


def cycle_energy() -> pd.DataFrame | None:
    hits = [h for h in glob.glob(CYCLE_CSV) if "stage" not in h and "_friction" not in h]
    if not hits:
        return None
    dt, v, T = _cycle(hits[0])
    E = np.sum(T * v * dt)
    rows = []
    for bound in ("low", "high"):
        c = components(np.abs(T), v, bound)
        row = dict(bound=bound, P_tether_kW=E / dt.sum() / 1e3)
        for k, F in c.items():
            row[k] = 100 * np.sum(F * np.abs(v) * dt) / E
            row[k + " W"] = np.sum(F * np.abs(v) * dt) / dt.sum()
        up, down = split(c)
        row["upstream"] = 100 * np.sum(up * np.abs(v) * dt) / E
        row["downstream"] = 100 * np.sum(down * np.abs(v) * dt) / E
        row["total"] = row["upstream"] + row["downstream"]
        for k, F in (("upstream", up), ("downstream", down), ("total", up + down)):
            row[k + " W"] = np.sum(F * np.abs(v) * dt) / dt.sum()
        rows.append(row)
    return pd.DataFrame(rows)


COLORS = {"P1": "#2a78d6", "P2": "#1baf7a", "P3": "#eda100", "drum": "#e87ba4",
          "belt": "#b5651d", "gear mesh": "#4a3aa7", "no-load": "#52514e"}


def plot_breakdown(path: Path) -> None:
    v = np.linspace(-8, 3.5, 231)
    fig, axs = plt.subplots(1, 2, figsize=(9.5, 4.0), sharey=True)
    for ax, (T, title) in zip(axs, ((1000.0, "T = 1 kN (reel-in level)"),
                                    (7400.0, "T = 7.4 kN (optimised reel-out level)"))):
        lo, hi = components(T, v, "low"), components(T, v, "high")
        base = np.zeros_like(v)
        for k in COLORS:
            mid = 0.5 * (lo[k] + hi[k])
            ax.fill_between(v, base, base + mid, color=COLORS[k], lw=0, label=k)
            base = base + mid
        tot_lo = sum(lo.values()); tot_hi = sum(hi.values())
        ax.plot(v, tot_lo, color="#0b0b0b", lw=1, ls="--", label="low / high bound")
        ax.plot(v, tot_hi, color="#0b0b0b", lw=1, ls="--")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("reel speed v (m s$^{-1}$)")
        ax.grid(color="#e4e3df", lw=0.8); ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axs[0].set_ylabel("tether-referred loss force (N)")
    axs[1].legend(fontsize=7.5, frameon=False, loc="upper right", ncol=2)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def _style(ax):
    ax.grid(color="#e4e3df", lw=0.8); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def _mid(T, v):
    lo, hi = components(T, v, "low"), components(T, v, "high")
    return {k: 0.5 * (lo[k] + hi[k]) for k in lo}


def _baseline_cycle():
    hits = [h for h in glob.glob(CYCLE_CSV) if "stage" not in h and "_friction" not in h]
    return _cycle(hits[0]) if hits else None


def plot_power_chain(path: Path, cyc: pd.DataFrame | None) -> None:
    """Block diagram tether -> motor shaft with the cycle-mean loss per element."""
    names = ["P1", "P2", "P3", "drum", "belt", "gear mesh", "no-load"]
    boxes = [("tether\nin", None), ("pulley\nP1", "P1"), ("pulley\nP2", "P2"),
             ("pulley P3\n+ sensor", "P3"), ("drum", "drum"), ("belt", "belt"),
             ("gearbox", "gear mesh"),
             ("motor\nshaft", None), ("electrical", None)]
    fig, ax = plt.subplots(figsize=(7.8, 2.9))
    w, h, gap = 1.0, 0.8, 0.35
    xs = [i * (w + gap) for i in range(len(boxes))]
    for x, (label, key) in zip(xs, boxes):
        out_of_scope = label.startswith("electrical")
        fc = "#f2f1ed" if key is None else "white"
        ax.add_patch(plt.Rectangle((x, 0), w, h, fc=fc, ec="#52514e",
                                   lw=1.0, ls="--" if out_of_scope else "-",
                                   hatch="///" if out_of_scope else None))
        ax.text(x + w / 2, h / 2, label, ha="center", va="center", fontsize=8.5,
                bbox=dict(fc="white", ec="none", pad=1) if out_of_scope else None)
    for x0, x1 in zip(xs[:-1], xs[1:]):
        ax.annotate("", (x1, h / 2), (x0 + w, h / 2),
                    arrowprops=dict(arrowstyle="-|>", color="#0b0b0b", lw=1.2))
    # loss arrows; gearbox carries mesh + no-load (seals, churning, motor bearings, fan)
    loss_at = {"P1": 1, "P2": 2, "P3": 3, "drum": 4, "belt": 5, "gear mesh": 6,
               "no-load": 6}
    for k in names:
        i = loss_at[k]
        dx = {"gear mesh": -0.25, "no-load": 0.25}.get(k, 0.0)
        ylen = 1.1 if k == "no-load" else 0.55
        x = xs[i] + w / 2 + dx
        ax.annotate("", (x, -ylen), (x, 0),
                    arrowprops=dict(arrowstyle="-|>", color=COLORS[k], lw=2.2))
        if cyc is not None:
            lo, hi = cyc.iloc[0][k + " W"], cyc.iloc[1][k + " W"]
            txt = f"{k}\n{_fmt_range(lo, hi, sep='–')} W"
        else:
            txt = k
        ax.text(x, -ylen - 0.07, txt, ha="center", va="top", fontsize=7.2, color=COLORS[k])
    # sensor marker and the upstream / downstream brackets
    xs3 = xs[3] + w / 2
    ax.plot([xs3, xs3], [h + 0.05, h + 0.55], color="#e34948", lw=1.2, ls=":")
    ax.text(xs3, h + 0.6, "force sensor\n(mean tension at P3)", ha="center", va="bottom",
            fontsize=7.5, color="#e34948")
    ax.annotate("", (xs[1], h + 0.25), (xs3 - 0.05, h + 0.25),
                arrowprops=dict(arrowstyle="<->", color="#6c6b67", lw=0.8))
    ax.text((xs[1] + xs3) / 2, h + 0.32, "upstream", ha="center", fontsize=7.5, color="#6c6b67")
    ax.annotate("", (xs3 + 0.05, h + 0.25), (xs[7] + w, h + 0.25),
                arrowprops=dict(arrowstyle="<->", color="#6c6b67", lw=0.8))
    ax.text((xs3 + xs[7] + w) / 2, h + 0.32, "downstream", ha="center", fontsize=7.5,
            color="#6c6b67")
    ax.text(xs[8] + w / 2, -0.25, "outside\nthis note", ha="center", va="top",
            fontsize=7.2, color="#6c6b67")
    ax.set_xlim(-0.1, xs[-1] + w + 0.1); ax.set_ylim(-1.75, h + 1.25)
    ax.set_aspect("equal"); ax.axis("off")
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def plot_scaling_classes(path: Path) -> None:
    """The three loss classes: force vs speed, force vs tension, power vs speed."""
    v = np.linspace(-1, 1, 201); T = np.linspace(0, 1, 2)
    cls = [("constant (Coulomb)", "#52514e", lambda v, T: 0.15 + 0 * v * T),
           ("proportional to tension", "#4a3aa7", lambda v, T: 0.4 * T + 0 * v),
           ("speed-dependent (viscous)", "#eda100", lambda v, T: 0.45 * np.abs(v) + 0 * T)]
    fig, axs = plt.subplots(1, 3, figsize=(7.6, 2.7))
    for name, c, f in cls:
        axs[0].plot(v, f(v, 0.8) * np.ones_like(v), color=c, lw=2, label=name)
        axs[1].plot(T, f(0.4, T) * np.ones_like(T), color=c, lw=2)
        axs[2].plot(v, f(v, 0.8) * np.abs(v), color=c, lw=2)
    titles = ["loss force vs reel speed\n(fixed tension)",
              "loss force vs tension\n(fixed speed)",
              "loss power $F|v|$ vs reel speed\n(fixed tension)"]
    xl = ["reel speed $v$", "tension $T$", "reel speed $v$"]
    for ax, t, x in zip(axs, titles, xl):
        _style(ax); ax.set_title(t, fontsize=8.5); ax.set_xlabel(x, fontsize=8.5)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_ylim(bottom=0)
    for ax in (axs[0], axs[2]):
        ax.axvline(0, color="#b5b4af", lw=0.8)
        ax.text(-0.95, ax.get_ylim()[1] * 0.93, "reel-in", fontsize=7.5, color="#6c6b67")
        ax.text(0.55, ax.get_ylim()[1] * 0.93, "reel-out", fontsize=7.5, color="#6c6b67")
    axs[0].set_ylabel("tether-referred loss")
    fig.legend(*axs[0].get_legend_handles_labels(), loc="lower center", ncol=3,
               fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.1, 1, 1)); fig.savefig(path, dpi=200); plt.close(fig)


def plot_axle_force(path: Path) -> None:
    """P3 free-body sketch and the axle-force factor 2 sin(theta/2)."""
    fig, axs = plt.subplots(1, 2, figsize=(9.0, 3.4), gridspec_kw=dict(width_ratios=[1, 1.3]))
    ax = axs[0]
    R = 0.5
    ax.add_patch(plt.Circle((0, 0), R, fc="white", ec="#52514e", lw=1.2))
    ax.plot(0, 0, "o", color="#e34948", ms=5)
    # rope: comes down vertically on the left tangent, leaves horizontally at the bottom
    ax.plot([-R, -R], [0, 1.6], color="#2a78d6", lw=2.5)
    ax.add_patch(matplotlib.patches.Arc((0, 0), 2 * R, 2 * R, theta1=180, theta2=270,
                                        color="#2a78d6", lw=2.5))
    ax.plot([0, 1.8], [-R, -R], color="#2a78d6", lw=2.5)
    ax.text(-R - 0.08, 1.45, "from P2", ha="right", fontsize=8, color="#2a78d6")
    ax.text(1.75, -R - 0.12, "to drum", ha="right", va="top", fontsize=8, color="#2a78d6")
    kw = dict(arrowstyle="-|>", lw=1.6, mutation_scale=12)
    ax.annotate("", (0, 1.0), (0, 0), arrowprops=dict(color="#1baf7a", **kw))
    ax.text(0.07, 0.9, r"$T_\mathrm{in}$", fontsize=10, color="#1baf7a")
    ax.annotate("", (1.0, 0), (0, 0), arrowprops=dict(color="#1baf7a", **kw))
    ax.text(0.95, 0.07, r"$T_\mathrm{out}$", fontsize=10, color="#1baf7a")
    ax.annotate("", (1.0, 1.0), (0, 0), arrowprops=dict(color="#e34948", **kw))
    ax.text(1.02, 1.02, r"$F_\mathrm{axle}\approx\sqrt{2}\,\bar T$", fontsize=10,
            color="#e34948")
    ax.plot([0, 1, 1], [1, 1, 0], color="#b5b4af", lw=0.8, ls=":")
    ax.set_xlim(-1.2, 2.1); ax.set_ylim(-1.0, 1.7); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("forces on P3 (wrap 90°)", fontsize=9)

    ax = axs[1]
    th = np.linspace(0, 180, 181)
    ax.plot(th, 2 * np.sin(np.radians(th) / 2), color="#0b0b0b", lw=1.8)
    for name, deg, dy in (("P1 (β = 30°)", 30, 0.12), ("P2, P3", 90, 0.12)):
        y = 2 * np.sin(np.radians(deg) / 2)
        ax.plot(deg, y, "o", color="#2a78d6")
        ax.text(deg + 4, y - dy - 0.05, name, fontsize=8, color="#2a78d6")
    _style(ax)
    ax.set_xlabel("wrap angle θ (deg)"); ax.set_ylabel(r"$F_\mathrm{axle}/T = 2\sin(\theta/2)$")
    ax.set_xlim(0, 180); ax.set_ylim(0, 2.1)
    ax.set_title("axle load grows with the wrap angle", fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def plot_pulley_physics(path: Path) -> None:
    """Per-pulley loss: bearing vs bending against wrap angle, and bending vs D/d."""
    fig, axs = plt.subplots(1, 2, figsize=(9.0, 3.3))
    th = np.radians(np.linspace(0, 180, 181))
    ax = axs[0]
    b_lo = 100 * MU_BEARING[0] * 2 * np.sin(th / 2) * BORE_RATIO_PULLEY
    b_hi = 100 * MU_BEARING[1] * 2 * np.sin(th / 2) * BORE_RATIO_PULLEY
    ax.fill_between(np.degrees(th), b_lo, b_hi, color="#2a78d6", alpha=0.35, lw=0,
                    label="bearing friction")
    be = 100 * np.array(C_BEND) * D_TETHER / D_PULLEY
    ax.fill_between(np.degrees(th), be[0], be[1], color="#eda100", alpha=0.35, lw=0,
                    label="tether bending")
    for deg, lab in ((30, "P1"), (90, "P2, P3")):
        ax.axvline(deg, color="#b5b4af", lw=0.8, ls="--")
        ax.text(deg + 2, 0.86, lab, fontsize=8, color="#6c6b67")
    _style(ax); ax.set_xlim(0, 180); ax.set_ylim(0, 0.95)
    ax.set_xlabel("wrap angle θ (deg)"); ax.set_ylabel("loss per pulley (% of T)")
    ax.set_title(f"one sheave, D = {D_PULLEY*1e3:.0f} mm, d = {D_TETHER*1e3:.0f} mm", fontsize=9)
    ax.legend(fontsize=8, frameon=False, loc="center right")

    ax = axs[1]
    Dd = np.linspace(10, 100, 200)
    ax.fill_between(Dd, 100 * C_BEND[0] / Dd, 100 * C_BEND[1] / Dd, color="#eda100",
                    alpha=0.35, lw=0, label="full bend + unbend (pulley)")
    ax.fill_between(Dd, 50 * C_BEND[0] / Dd, 50 * C_BEND[1] / Dd, color="#e87ba4",
                    alpha=0.35, lw=0, label="half cycle (winding onto drum)")
    for x, lab in ((D_PULLEY / D_TETHER, "pulleys"), (D_DRUM / D_TETHER, "drum")):
        ax.axvline(x, color="#b5b4af", lw=0.8, ls="--")
        ax.text(x + 1.5, 1.2, f"{lab}\nD/d = {x:.0f}", fontsize=8, color="#6c6b67")
    _style(ax); ax.set_xlim(10, 100); ax.set_ylim(0, 2.1)
    ax.set_xlabel("sheave-to-rope diameter ratio D/d")
    ax.set_ylabel("bending loss (% of T)")
    ax.set_title("bending loss falls with D/d", fontsize=9)
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def _efficiency(T, v):
    """Mechanical drivetrain efficiency, mid-range: useful over input power
    (shaft/tether in reel-out, tether/shaft in reel-in)."""
    F = sum(_mid(T, v).values())
    return np.where(v > 0, (T - F) / T, T / (T + F))


def plot_efficiency_map(path: Path) -> None:
    v = np.linspace(-8.5, 4.0, 251); T = np.linspace(300, 8400, 200)
    V, TT = np.meshgrid(v, T)
    eta = 100 * _efficiency(TT, V)
    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    lev = [30, 40, 50, 60, 70, 80, 85, 90, 93, 95]
    cf = ax.contourf(V, TT / 1e3, eta, levels=[0] + lev + [100], cmap="viridis", vmin=20, vmax=100)
    cs = ax.contour(V, TT / 1e3, eta, levels=lev, colors="white", linewidths=0.6)
    ax.clabel(cs, fmt="%d %%", fontsize=7.5)
    cyc = _baseline_cycle()
    if cyc is not None:
        dt, vv, Tc = cyc
        ax.plot(np.r_[vv, vv[0]], np.r_[Tc, Tc[0]] / 1e3, color="#e34948", lw=1.4,
                label="optimised baseline cycle")
        ax.legend(fontsize=8, frameon=False, loc="upper left", labelcolor="white")
    ax.axvline(0, color="white", lw=0.8, ls="--")
    ax.set_xlabel("reel speed v (m s$^{-1}$)"); ax.set_ylabel("tether tension T (kN)")
    ax.text(-4.25, 1.02, "reel-in (motor drives)", transform=ax.get_xaxis_transform(),
            ha="center", fontsize=8.5)
    ax.text(2.0, 1.02, "reel-out (generator)", transform=ax.get_xaxis_transform(),
            ha="center", fontsize=8.5)
    cb = fig.colorbar(cf, ax=ax); cb.set_label("mechanical efficiency (%)")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def plot_cycle_timeseries(path: Path) -> None:
    cyc = _baseline_cycle()
    if cyc is None:
        return
    dt, v, T = cyc
    t = np.r_[0, np.cumsum(dt)[:-1]]
    T = np.abs(T)
    fig, axs = plt.subplots(3, 1, figsize=(8.5, 6.0), sharex=True,
                            gridspec_kw=dict(height_ratios=[1, 1, 1.5]))
    ri = v < 0
    for ax in axs:
        _style(ax)
        edges = np.flatnonzero(np.diff(np.r_[0, ri.astype(int), 0]))
        for a, b in zip(edges[::2], edges[1::2]):
            ax.axvspan(t[a], t[min(b, len(t) - 1)], color="#f2f1ed", lw=0)
    axs[0].plot(t, T / 1e3, color="#0b0b0b", lw=1.2)
    axs[0].set_ylabel("T (kN)")
    axs[1].plot(t, v, color="#0b0b0b", lw=1.2); axs[1].axhline(0, color="#b5b4af", lw=0.8)
    axs[1].set_ylabel("v (m s$^{-1}$)")
    mid = _mid(T, v); base = np.zeros_like(t)
    for k in COLORS:
        P = mid[k] * np.abs(v)
        axs[2].fill_between(t, base / 1e3, (base + P) / 1e3, color=COLORS[k], lw=0,
                            label=k, step=None)
        base = base + P
    lo = sum(components(T, v, "low").values()) * np.abs(v)
    hi = sum(components(T, v, "high").values()) * np.abs(v)
    axs[2].plot(t, lo / 1e3, color="#0b0b0b", lw=0.8, ls="--", label="low / high")
    axs[2].plot(t, hi / 1e3, color="#0b0b0b", lw=0.8, ls="--")
    axs[2].set_ylabel("loss power (kW)"); axs[2].set_xlabel("time in cycle (s)")
    axs[2].legend(fontsize=7.5, frameon=False, ncol=4, loc="upper left")
    axs[0].text(0.99, 0.9, "shaded: reel-in", transform=axs[0].transAxes, ha="right",
                fontsize=8, color="#6c6b67")
    axs[2].set_xlim(t[0], t[-1] + dt[-1])
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def _waterfall(ax, steps, total_label, color_total="#0b0b0b", unit="kW", fmt="{:.2f}",
               errs=None):
    """Horizontal waterfall: steps = [(label, delta, color)], first is the start."""
    y = 0; level = 0.0
    labels = []
    for i, (lab, d, c) in enumerate(steps):
        if i == 0:
            ax.barh(y, d, color=c); level = d
            ax.text(d, y, " " + fmt.format(d), va="center", fontsize=8)
        else:
            ax.barh(y, d, left=level, color=c)
            txt = fmt.format(d)
            if errs and errs[i] is not None:
                if fmt.format(errs[i][0]) != fmt.format(errs[i][1]):
                    txt += "  (" + _fmt_range(errs[i][0], errs[i][1], fmt, "–") + ")"
            ax.text(min(level, level + d), y, txt + " ", va="center", ha="right", fontsize=8)
            level += d
        labels.append(lab); y -= 1
    ax.barh(y, level, color=color_total)
    ax.text(level, y, " " + fmt.format(level), va="center", fontsize=8)
    labels.append(total_label)
    ax.set_yticks(-np.arange(len(labels))); ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel(unit); _style(ax); ax.grid(axis="y", visible=False)


def plot_cycle_waterfall(path: Path, cyc: pd.DataFrame | None) -> None:
    if cyc is None:
        return
    lo, hi = cyc.iloc[0], cyc.iloc[1]
    P0 = lo["P_tether_kW"]
    keys = ["P1", "P2", "P3", "drum", "belt", "gear mesh", "no-load"]
    steps = [("tether power", P0, "#2a78d6")]
    errs = [None]
    for k in keys:
        m = 0.5 * (lo[k + " W"] + hi[k + " W"]) / 1e3
        steps.append((k, -m, COLORS[k])); errs.append((lo[k + " W"] / 1e3, hi[k + " W"] / 1e3))
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    _waterfall(ax, steps, "motor shaft", errs=errs)
    ax.set_xlabel("cycle-mean power (kW)")
    ax.set_xlim(0, P0 * 1.15)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


# --- gear oil viscosity (ASTM D341 Walther) --------------------------------------
# Illustrative ISO VG 220 mineral gear oil: 220 mm2/s at 40 C, 19 mm2/s at 100 C.
VG_NU = ((40.0, 220.0), (100.0, 19.0))
CHURN_EXP = (0.2, 0.5)   # c_v ~ nu^n


def oil_viscosity(temp_c):
    (t1, n1), (t2, n2) = VG_NU
    z = lambda n: np.log10(np.log10(n + 0.7))
    B = (z(n1) - z(n2)) / (np.log10(t2 + 273.15) - np.log10(t1 + 273.15))
    A = z(n1) + B * np.log10(t1 + 273.15)
    return 10 ** (10 ** (A - B * np.log10(np.asarray(temp_c) + 273.15))) - 0.7


def plot_tow_test(path: Path) -> None:
    fig, axs = plt.subplots(1, 2, figsize=(9.0, 3.4))
    ax = axs[0]
    v = np.linspace(0, 8, 50)
    Fc, cv = NO_LOAD
    ax.plot(v, Fc + cv * v, color="#0b0b0b", lw=1.8, label="GS1 tow test (cold day)")
    r = [(oil_viscosity(50) / oil_viscosity(10)) ** n for n in CHURN_EXP]
    ax.fill_between(v, Fc + cv * min(r) * v, Fc + cv * max(r) * v, color="#1baf7a",
                    alpha=0.3, lw=0, label="same, $c_v$ warmed 10 → 50 °C")
    ax.annotate("intercept: $F_\\mathrm{c}$ (sliding)", (0, Fc), (1.2, 60), fontsize=8,
                arrowprops=dict(arrowstyle="-", color="#6c6b67", lw=0.7))
    ax.text(5.3, Fc + cv * 5.0 - 70, "slope: $c_v$", fontsize=8)
    _style(ax); ax.set_xlim(0, 8); ax.set_ylim(0, 450)
    ax.set_xlabel("tow speed v (m s$^{-1}$)"); ax.set_ylabel("pulling force (N)")
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    ax.set_title("what a tow test measures", fontsize=9)

    ax = axs[1]
    tc = np.linspace(0, 80, 161)
    nu = oil_viscosity(tc)
    ax.semilogy(tc, nu, color="#0b0b0b", lw=1.8)
    ax.set_ylabel("oil viscosity ν (mm$^2$ s$^{-1}$)")
    ax.set_xlabel("oil temperature (°C)")
    ax2 = ax.twinx()
    for n, ls in zip(CHURN_EXP, ("-", "--")):
        ax2.plot(tc, (nu / oil_viscosity(10)) ** n, color="#1baf7a", lw=1.4, ls=ls,
                 label=f"$c_v/c_v(10\\,°C)$, n = {n}")
    ax2.set_ylim(0, 1.6); ax2.set_ylabel("relative viscous drag", color="#1baf7a")
    ax2.legend(fontsize=7.5, frameon=False, loc="upper right")
    ax.axvspan(0, 15, color="#2a78d6", alpha=0.12, lw=0)
    ax.axvspan(40, 70, color="#e34948", alpha=0.10, lw=0)
    ax.text(1, 45, "cold\ntow", fontsize=7.5, color="#2a78d6")
    ax.text(42, 45, "warm\noperation", fontsize=7.5, color="#e34948")
    _style(ax); ax2.spines["top"].set_visible(False)
    ax.set_title("ISO VG 220 gear oil (illustrative)", fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


# GS1 (TU Delft, 2011): Fechner, AWEC 2011, slide 13, cycle 1, Wh.
GS1_CYCLE1 = [("reel-out tether energy", 200), ("reel-out friction", -14),
              ("generator electrical", -33), ("reel-in tether energy", -39),
              ("reel-in friction", -14), ("motor electrical", -7),
              ("spindle motor", -11), ("brakes, computers", -4)]


def plot_gs1_waterfall(path: Path) -> None:
    col = {"tether": "#2a78d6", "friction": "#52514e", "electrical": "#eda100",
           "spindle": "#e87ba4", "brakes": "#e87ba4"}
    steps = []
    for lab, d in GS1_CYCLE1:
        c = next((v for k, v in col.items() if k in lab), "#2a78d6")
        if "reel-in tether" in lab:
            c = "#7fb0ea"
        steps.append((lab, float(d), c))
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    _waterfall(ax, steps, "net electrical", unit="Wh", fmt="{:.0f}")
    ax.set_xlabel("energy per cycle (Wh)"); ax.set_xlim(0, 230)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def gs1_improvements() -> pd.DataFrame:
    """Illustrative what-if on GS1 cycle 1 (Wh): each lever applied alone and all
    together. Assumptions are stated in the report, not measured."""
    E_out, E_in = 200.0, 39.0                  # tether energies
    f_out, f_in = 14.0, 14.0                   # friction
    el_out, el_in = 33.0, 7.0                  # electrical (machine + controller)
    spindle, brakes = 11.0, 4.0
    # viscous share of the friction from the GS1 law (13.0 / 16.5 Wh model split)
    visc_out, visc_in = 6.0 * 14 / 13.0, 9.6 * 14 / 16.5

    def net(f_o=f_out, f_i=f_in, eta_g=None, eta_m=None, sp=spindle, br=brakes, e_in=E_in):
        shaft_out = E_out - f_o
        term_out = shaft_out - el_out if eta_g is None else shaft_out * eta_g
        shaft_in = e_in + f_i
        term_in = shaft_in + el_in if eta_m is None else shaft_in / eta_m
        return term_out - term_in - sp - br

    warm = 0.7                                  # c_v at 50 C ~ 0.7 x cold (n ~ 0.3)
    rows = [("as measured", net()),
            ("brake holding-current economiser", net(br=1.0)),
            ("efficient spindle motor + gear", net(sp=4.0)),
            ("oil grade/fill, low-friction seals", net(f_o=f_out - (1 - warm) * visc_out - 1.5,
                                                 f_i=f_in - (1 - warm) * visc_in - 1.5)),
            ("reel-in tether energy -25 %", net(e_in=0.75 * E_in)),
            ("PM machine + inverter, 94 % each way", net(eta_g=0.94, eta_m=0.94))]
    allfix = net(f_o=f_out - (1 - warm) * visc_out - 1.5, f_i=f_in - (1 - warm) * visc_in - 1.5,
                 eta_g=0.94, eta_m=0.94, sp=4.0, br=1.0, e_in=0.75 * E_in)
    rows.append(("all of the above", allfix))
    df = pd.DataFrame(rows, columns=["case", "net_Wh"])
    df["gain_Wh"] = df.net_Wh - df.net_Wh.iloc[0]
    df["eta_energy_pct"] = 100 * df.net_Wh / E_out
    return df


def plot_gs1_improvements(path: Path, df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    y = -np.arange(len(df))
    cols = ["#b5b4af"] + ["#1baf7a"] * (len(df) - 2) + ["#2a78d6"]
    ax.barh(y, df.net_Wh, color=cols)
    for yi, (n, g) in zip(y, zip(df.net_Wh, df.gain_Wh)):
        ax.text(n, yi, f" {n:.0f} Wh" + (f"  (+{g:.0f})" if g > 0.5 else ""), va="center",
                fontsize=8)
    ax.set_yticks(y); ax.set_yticklabels(df.case, fontsize=8)
    ax.set_xlabel("net electrical energy per cycle (Wh), GS1 cycle 1")
    ax.set_xlim(0, df.net_Wh.max() * 1.25); _style(ax); ax.grid(axis="y", visible=False)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def _fmt_range(lo, hi, fmt="{:.0f}", sep="--"):
    a, b = fmt.format(lo), fmt.format(hi)
    return a if a == b else a + sep + b


def write_tables(op: pd.DataFrame, cyc: pd.DataFrame | None, cycles: pd.DataFrame,
                 path: Path) -> None:
    keys = ["P1", "P2", "P3", "drum", "belt", "gear mesh", "no-load"]
    cols = keys + ["upstream", "downstream", "total"]
    for fname, scale in (("drivetrain_operating_points.tex", "force"),
                         ("drivetrain_power_points.tex", "power")):
        lines = [r"\begin{tabular}{l" + "r" * len(cols) + "}", r"\toprule",
                 "Operating point & " + " & ".join(cols) + r" \\", r"\midrule"]
        for label, T, v in OPERATING_POINTS:
            lo = op[(op.point == label) & (op.bound == "low")].iloc[0]
            hi = op[(op.point == label) & (op.bound == "high")].iloc[0]
            f = abs(v) if scale == "power" else 1.0  # loss power = F |v| (W)
            cells = [_fmt_range(lo[k] * f, hi[k] * f) for k in cols]
            head = f"{label} ({T/1e3:.1f}~kN, {v:+.2f}~m\\,s$^{{-1}}$)"
            lines.append(head + " & " + " & ".join(cells) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        path.with_name(fname).write_text("\n".join(lines) + "\n")

    lines = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r"Cycle & tether (kW) & loss (kW) & loss (\%) & shaft (kW) & reel-in share of loss (\%) & peak loss (kW) \\",
             r"\midrule"]
    for _, r in cycles.iterrows():
        lines.append(f"{r.cycle} ({r.duration_s:.0f}~s) & {r.P_tether_kW:.2f} & "
                     f"{_fmt_range(r.loss_kW_low, r.loss_kW_high, '{:.2f}')} & "
                     f"{_fmt_range(r.loss_pct_low, r.loss_pct_high, '{:.1f}')} & "
                     f"{_fmt_range(r.shaft_kW_high, r.shaft_kW_low, '{:.2f}')} & "
                     f"{_fmt_range(min(r.reelin_share_low, r.reelin_share_high), max(r.reelin_share_low, r.reelin_share_high), '{:.0f}')} & "
                     f"{_fmt_range(r.peak_kW_low, r.peak_kW_high, '{:.1f}')}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.with_name("drivetrain_cycles.tex").write_text("\n".join(lines) + "\n")
    if cyc is None:
        return
    lo, hi = cyc.iloc[0], cyc.iloc[1]
    lines = [r"\begin{tabular}{l" + "r" * len(cols) + "}", r"\toprule",
             " & " + " & ".join(cols) + r" \\", r"\midrule",
             r"mean loss power (W) & "
             + " & ".join(_fmt_range(lo[k + " W"], hi[k + " W"]) for k in cols) + r" \\",
             r"share of $\int T v\,\mathrm{d}t$ (\%) & "
             + " & ".join(_fmt_range(lo[k], hi[k], "{:.1f}") for k in cols)
             + r" \\", r"\bottomrule", r"\end{tabular}"]
    path.with_name("drivetrain_cycle.tex").write_text("\n".join(lines) + "\n")
    # every number the report text quotes, so the prose follows the model
    pul = lambda r: r["P1"] + r["P2"] + r["P3"]
    base = cycles.iloc[0]
    o = lambda label, b: op[(op.point == label) & (op.bound == b)].iloc[0]
    ro_lo, ro_hi = o("Optimised reel-out", "low"), o("Optimised reel-out", "high")
    ri_lo, ri_hi = o("Typical reel-in", "low"), o("Typical reel-in", "high")
    rf_lo, rf_hi = o("Fast reel-in", "low"), o("Fast reel-in", "high")
    m = {
        "cyclePtether": f"{lo['P_tether_kW']:.2f}",
        "cycleLossLow": f"{lo['total']:.1f}", "cycleLossHigh": f"{hi['total']:.1f}",
        "cycleDownLow": f"{lo['downstream']:.1f}", "cycleDownHigh": f"{hi['downstream']:.1f}",
        "cycleLossKWLow": f"{lo['total W'] / 1e3:.2f}",
        "cycleLossKWHigh": f"{hi['total W'] / 1e3:.2f}",
        "cycleNoLoadLow": f"{lo['no-load']:.1f}", "cycleNoLoadHigh": f"{hi['no-load']:.1f}",
        "cycleNoLoadShareLow": f"{100 * hi['no-load'] / hi['total']:.0f}",
        "cycleNoLoadShareHigh": f"{100 * lo['no-load'] / lo['total']:.0f}",
        "cycleMeshLow": f"{lo['gear mesh']:.1f}", "cycleMeshHigh": f"{hi['gear mesh']:.1f}",
        "cycleBeltLow": f"{lo['belt']:.1f}", "cycleBeltHigh": f"{hi['belt']:.1f}",
        "cyclePulleyLow": f"{pul(lo):.1f}", "cyclePulleyHigh": f"{pul(hi):.1f}",
        "peakLow": f"{base.peak_kW_low:.1f}", "peakHigh": f"{base.peak_kW_high:.1f}",
        "reelinShareLow": f"{min(base.reelin_share_low, base.reelin_share_high):.0f}",
        "reelinShareHigh": f"{max(base.reelin_share_low, base.reelin_share_high):.0f}",
        "allReelinShareLow": f"{cycles[['reelin_share_low', 'reelin_share_high']].min().min():.0f}",
        "allReelinShareHigh": f"{cycles[['reelin_share_low', 'reelin_share_high']].max().max():.0f}",
        "allLossKWLow": f"{cycles.loss_kW_low.min():.2f}",
        "allLossKWHigh": f"{cycles.loss_kW_high.max():.2f}",
        "allPctLow": f"{cycles.loss_pct_low.min():.1f}",
        "allPctHigh": f"{cycles.loss_pct_high.max():.1f}",
        "shaftShareLow": f"{100 - cycles.loss_pct_high.max():.0f}",
        "shaftShareHigh": f"{100 - cycles.loss_pct_low.min():.0f}",
        "roLossLow": f"{ro_lo.total * ro_lo.v / 1e3:.2f}",
        "roLossHigh": f"{ro_hi.total * ro_hi.v / 1e3:.2f}",
        "roPctLow": f"{100 * ro_lo.total / ro_lo['T']:.1f}",
        "roPctHigh": f"{100 * ro_hi.total / ro_hi['T']:.1f}",
        "roMeshLow": f"{ro_lo['gear mesh']:.0f}", "roMeshHigh": f"{ro_hi['gear mesh']:.0f}",
        "roBeltLow": f"{ro_lo['belt']:.0f}", "roBeltHigh": f"{ro_hi['belt']:.0f}",
        "roPulleyLow": f"{pul(ro_lo):.0f}", "roPulleyHigh": f"{pul(ro_hi):.0f}",
        "riLossLow": f"{ri_lo.total * 3.5 / 1e3:.2f}", "riLossHigh": f"{ri_hi.total * 3.5 / 1e3:.2f}",
        "rfLossLow": f"{rf_lo.total * 8.0 / 1e3:.2f}", "rfLossHigh": f"{rf_hi.total * 8.0 / 1e3:.2f}",
    }
    path.with_name("drivetrain_macros.tex").write_text(
        "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in m.items()))


# --- objective comparison: tether power vs shaft power after drive losses ------
OBJ_DIR = "results/LEI-V3-KITE/optimization/uploop_eight"
OBJ_RUNS = [  # (cycle, objective, file prefix)
    ("baseline (multi-lobe)", "tether power $Tv$", "baseline"),
    ("baseline (multi-lobe)", "shaft power, from the $Tv$ optimum", "baseline_drive_warm"),
    ("baseline (multi-lobe)", "shaft power, from the seed", "baseline_drive"),
    ("up-loop eight", "tether power $Tv$", "uploop_eight"),
    ("up-loop eight", "shaft power, from the seed", "uploop_eight_drive"),
]


def _run_metrics(path: str) -> dict:
    """Cycle metrics of a saved optimum on the optimiser's left rule: tether
    and shaft power (mid estimate, low-high) and the reel strategy."""
    dt, v, T = _cycle(path)
    T = np.abs(T)
    D = dt.sum()
    ri = v < 0
    out = dict(duration=D, P_tether=np.sum(T * v * dt) / D / 1e3,
               t_in_share=100 * dt[ri].sum() / D,
               v_out=np.sum((v * dt)[~ri]) / dt[~ri].sum(),
               v_in=np.sum((-v * dt)[ri]) / dt[ri].sum(), v_in_max=-v.min(),
               T_out=np.sum((T * dt)[~ri]) / dt[~ri].sum(),
               T_in=np.sum((T * dt)[ri]) / dt[ri].sum(),
               E_in=-np.sum((T * v * dt)[ri]) / 3.6e3)
    for b in ("low", "high"):
        up, down = split(components(T, v, b))
        out[f"P_shaft_{b}"] = (np.sum(T * v * dt) - np.sum((up + down) * np.abs(v) * dt)) / D / 1e3
    mid = sum(_mid(T, v).values())
    out["P_shaft"] = (np.sum(T * v * dt) - np.sum(mid * np.abs(v) * dt)) / D / 1e3
    out["series"] = (np.r_[0, np.cumsum(dt)[:-1]], v, T)
    return out


def objective_comparison() -> list[tuple[str, str, dict]]:
    rows = []
    for cyc, obj, pre in OBJ_RUNS:
        hits = [h for h in glob.glob(f"{OBJ_DIR}/{pre}_optimized_wind_*.csv") if "stage" not in h]
        if hits:
            rows.append((cyc, obj, _run_metrics(hits[0])))
    return rows


def write_objective_table(rows, path: Path) -> None:
    lines = [r"\begin{tabular}{llrrrrrrrr}", r"\toprule",
             r"Cycle & objective & period & tether & shaft & reel-in & $\bar v_\mathrm{out}$ "
             r"& $\bar T_\mathrm{out}$ & $\bar v_\mathrm{in}$ & $\bar T_\mathrm{in}$ \\",
             r" & & (s) & (kW) & (kW) & time (\%) & (m\,s$^{-1}$) & (kN) & (m\,s$^{-1}$) & (kN) \\",
             r"\midrule"]
    for cyc, obj, m in rows:
        shaft = f"{m['P_shaft']:.2f} ({m['P_shaft_high']:.2f}--{m['P_shaft_low']:.2f})"
        lines.append(f"{cyc} & {obj} & {m['duration']:.0f} & {m['P_tether']:.2f} & {shaft} & "
                     f"{m['t_in_share']:.0f} & {m['v_out']:.2f} & {m['T_out'] / 1e3:.2f} & "
                     f"{m['v_in']:.2f} & {m['T_in'] / 1e3:.2f}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")
    # macros for the prose, one letter per run: A baseline Tv, B baseline shaft
    # from the Tv optimum, C baseline shaft from the seed, E eight Tv, F eight shaft
    letters = dict(zip([pre for _, _, pre in OBJ_RUNS], "ABCEF"))
    m = {}
    for (cyc, obj, r), (_, _, pre) in zip(rows, [x for x in OBJ_RUNS if _has_run(x[2])]):
        L = letters[pre]
        m.update({f"objTether{L}": f"{r['P_tether']:.2f}", f"objShaft{L}": f"{r['P_shaft']:.2f}",
                  f"objPeriod{L}": f"{r['duration']:.0f}", f"objVin{L}": f"{r['v_in']:.2f}",
                  f"objVinMax{L}": f"{r['v_in_max']:.2f}", f"objTin{L}": f"{r['T_in'] / 1e3:.2f}",
                  f"objVout{L}": f"{r['v_out']:.2f}", f"objTout{L}": f"{r['T_out'] / 1e3:.2f}",
                  f"objEin{L}": f"{r['E_in']:.1f}", f"objTinShare{L}": f"{r['t_in_share']:.0f}"})
        m[f"_r{L}"] = r
    for x, y in (("B", "A"), ("C", "A"), ("F", "E")):
        if f"_r{x}" in m and f"_r{y}" in m:
            rx, ry = m[f"_r{x}"], m[f"_r{y}"]
            m[f"objShaftGain{x}"] = f"{100 * (rx['P_shaft'] / ry['P_shaft'] - 1):+.1f}"
            m[f"objTetherChange{x}"] = f"{100 * (rx['P_tether'] / ry['P_tether'] - 1):+.1f}"
            m[f"objEinChange{x}"] = f"{100 * (rx['E_in'] / ry['E_in'] - 1):+.0f}"
    path.with_name("drivetrain_objective_macros.tex").write_text(
        "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in m.items()
                if not k.startswith("_")))


def _has_run(prefix: str) -> bool:
    return any("stage" not in h for h in glob.glob(f"{OBJ_DIR}/{prefix}_optimized_wind_*.csv"))


def plot_objective_comparison(rows, path: Path) -> None:
    styles = {"tether power $Tv$": ("#2a78d6", "-"),
              "shaft power, from the $Tv$ optimum": ("#e34948", "-")}
    base = [(obj, m) for cyc, obj, m in rows if cyc.startswith("baseline") and obj in styles]
    if len(base) < 2:
        return
    fig, axs = plt.subplots(3, 1, figsize=(8.0, 5.6), sharex=True)
    for obj, m in base:
        t, v, T = m["series"]
        c, ls = styles[obj]
        loss = sum(_mid(T, v).values()) * np.abs(v)
        lab = f"objective: {obj}"
        axs[0].plot(t, T / 1e3, color=c, ls=ls, lw=1.3, label=lab)
        axs[1].plot(t, v, color=c, ls=ls, lw=1.3)
        axs[2].plot(t, loss / 1e3, color=c, ls=ls, lw=1.3)
    axs[0].set_ylabel("T (kN)"); axs[1].set_ylabel("v (m s$^{-1}$)")
    axs[2].set_ylabel("drive loss (kW)"); axs[2].set_xlabel("time in cycle (s)")
    axs[1].axhline(0, color="#b5b4af", lw=0.8)
    axs[0].legend(fontsize=8, frameon=False, loc="lower center")
    for ax in axs:
        _style(ax)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--fig-dir", type=Path, default=OUT_FIG)
    ap.add_argument("--tex-dir", type=Path, default=OUT_TEX)
    args = ap.parse_args()
    args.fig_dir.mkdir(parents=True, exist_ok=True); args.tex_dir.mkdir(parents=True, exist_ok=True)
    op = operating_point_table()
    cyc = cycle_energy()
    cycles = cycles_power()
    plot_breakdown(args.fig_dir / "drivetrain_breakdown.png")
    plot_power_chain(args.fig_dir / "drivetrain_chain.png", cyc)
    plot_scaling_classes(args.fig_dir / "drivetrain_scaling_classes.png")
    plot_axle_force(args.fig_dir / "drivetrain_axle_force.png")
    plot_pulley_physics(args.fig_dir / "drivetrain_pulley_physics.png")
    plot_efficiency_map(args.fig_dir / "drivetrain_efficiency_map.png")
    plot_cycle_timeseries(args.fig_dir / "drivetrain_cycle_timeseries.png")
    plot_cycle_waterfall(args.fig_dir / "drivetrain_cycle_waterfall.png", cyc)
    plot_tow_test(args.fig_dir / "drivetrain_tow_test.png")
    plot_gs1_waterfall(args.fig_dir / "drivetrain_gs1_waterfall.png")
    imp = gs1_improvements()
    plot_gs1_improvements(args.fig_dir / "drivetrain_gs1_improvements.png", imp)
    obj_rows = objective_comparison()
    write_objective_table(obj_rows, args.tex_dir / "drivetrain_objectives.tex")
    plot_objective_comparison(obj_rows, args.fig_dir / "drivetrain_objectives.png")
    for cyc_name, obj, m in obj_rows:
        print(f"{cyc_name:22s} {obj:20s} period {m['duration']:.0f} s  tether {m['P_tether']:.3f} kW  "
              f"shaft {m['P_shaft']:.3f} ({m['P_shaft_high']:.3f}-{m['P_shaft_low']:.3f}) kW  "
              f"v_in {m['v_in']:.2f} max {m['v_in_max']:.2f}  T_in {m['T_in']:.0f}  "
              f"v_out {m['v_out']:.2f} T_out {m['T_out']:.0f}  E_in {m['E_in']:.1f} Wh  "
              f"t_in {m['t_in_share']:.0f}%")
    print(imp.round(1).to_string(index=False))
    print("VG220 viscosity 10/40/50/60 C:",
          np.round(oil_viscosity([10, 40, 50, 60]), 0))
    write_tables(op, cyc, cycles, args.tex_dir / "drivetrain_operating_points.tex")
    pd.set_option("display.width", 220)
    print(op.round(1).to_string(index=False))
    if cyc is not None:
        print("\n" + cyc.round(2).to_string(index=False))
    print("\n" + cycles.round(2).to_string(index=False))


if __name__ == "__main__":
    main()

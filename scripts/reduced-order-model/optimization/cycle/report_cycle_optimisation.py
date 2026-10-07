r"""Figures, tables and numbers of the full-cycle optimisation note.

Writes everything ``docs/cycle_optimisation/cycle_optimisation.tex`` includes,
so the document follows the code instead of restating it:

    docs/cycle_optimisation/generated/macros.tex     key numbers as \newcommand
    docs/cycle_optimisation/generated/seeds.tex      per-seed feasibility table
    docs/cycle_optimisation/generated/auto_trace.tex the --auto knob trace
    docs/cycle_optimisation/figures/*                the figures

Three parts:

* METHOD figures are pure geometry, built here from the kite's
  ``cycle_profile.yaml`` and the generator's default knobs (no simulation):
  the reel-in window and depower, the lobe seed before and after the reel-in
  design and fairing, the Dubins reel-in, and the depower-shifted winch law.
* SEEDS: each ``--run LABEL=KITE_DIR`` is a kite folder whose seed
  (``cycle_configs/``) was written by ``fit_periodic_cycle_config.py``. The
  seed is forward-simulated once here (cached as ``seed_series.npz`` under
  ``results/<folder name>/``); ``--logs DIR`` holding ``seed_<LABEL>.log``
  gives the --auto trace and generation time.
* OPTIMUM: ``--archived STEM`` reads an optimum written by
  ``run_full_cycle_opti.py`` (``STEM.csv`` / ``STEM.yaml``) and copies the
  seed-vs-optimum figures that run saved next to it.

Usage (project root)
    python scripts/reduced-order-model/optimization/cycle/report_cycle_optimisation.py \
        --run lobe=PATH/TO/KITE_A --run dubins=PATH/TO/KITE_B --logs PATH/TO/LOGS \
        --archived results/LEI-V3-KITE/optimization/full_cycle/full_cycle_optimized_<tag>
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

import fit_periodic_cycle_config as fpc  # noqa: E402
import run_full_cycle_opti as rfco  # noqa: E402
from cycle_comparison_plots import (  # noqa: E402
    cycle_metrics,
    series_from_optimizer_result,
    series_from_phase,
)
from cycle_kites import SEED_FILENAME, resolve_kite, wind_tag  # noqa: E402

from awetrim.kinematics import parametrized_patterns as pp  # noqa: E402
from awetrim.kinematics.parametrized_patterns import (  # noqa: E402
    _spherical_path_cross_speed,
    design_reelin_spline,
    fair_periodic_spline_to_curvature_limit,
    full_cycle_angles,
    reelin_bump,
)
from awetrim.utils.config_paths import REPO_ROOT  # noqa: E402

DOC = REPO_ROOT / "docs" / "cycle_optimisation"
GEN = DOC / "generated"
FIG = DOC / "figures"

C_SEED = "0.45"
C_OPT = "#0072B2"
C_REELIN = "#D55E00"
C_AUX = "#009E73"
WIDTH = 6.3  # in, text width of the A4 note

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.labelsize": 9,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    }
)

MACROS: dict[str, str] = {}


def macro(name, value, fmt=None):
    MACROS[name] = value if fmt is None else format(value, fmt)


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / name)
    plt.close(fig)
    print(f"  figures/{name}")


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def spline_angles(path, s):
    """(azimuth, elevation) of a path-parameter dict's spline at ``s`` [rad]."""
    M = int(path["M"])
    B0 = pp.periodic_bspline_basis_matrix(np.asarray(s, dtype=float), M)
    return B0 @ np.asarray(path["C_phi"], float), B0 @ np.asarray(path["C_beta"], float)


def spline_curvature(path, s):
    """Physical path curvature [1/m] of the spline at ``s`` (exact derivatives)."""
    M = int(path["M"])
    B0, B1, B2 = pp.periodic_bspline_basis_matrices(np.asarray(s, dtype=float), M)
    cp, cb = np.asarray(path["C_phi"], float), np.asarray(path["C_beta"], float)
    cross2, speed2, _ = _spherical_path_cross_speed(
        B0 @ cp, B0 @ cb, B1 @ cp, B1 @ cb, B2 @ cp, B2 @ cb, np
    )
    return np.sqrt(cross2) / np.maximum(speed2, 1e-30) ** 1.5 / float(path["r0"])


def sampled_turn_radius(phi, beta, r0):
    """Turn radius r0 / k_g [m] along densely sampled angles (geodesic part).

    The 3D curvature of a curve on the unit sphere splits as
    kappa^2 = 1 + k_g^2 (the normal curvature is 1), so the geodesic
    curvature -- what the kite turns at -- is sqrt(kappa^2 - 1).
    """
    d1 = (np.gradient(phi), np.gradient(beta))
    d2 = (np.gradient(d1[0]), np.gradient(d1[1]))
    cross2, speed2, _ = _spherical_path_cross_speed(
        phi, beta, d1[0], d1[1], d2[0], d2[1], np
    )
    kappa2 = cross2 / np.maximum(speed2, 1e-30) ** 3
    k_g = np.sqrt(np.maximum(kappa2 - 1.0, 0.0))
    with np.errstate(divide="ignore"):
        return float(r0) / k_g


def window_mask(s, art):
    return reelin_bump(
        s,
        reelout_fraction=art["reelout_fraction"],
        ramp_fraction=art["ramp_fraction"],
        reelin_center=art.get("reelin_center", 0.5),
    ) > 1e-6


def lobe_knobs():
    """The generator's default lobe knobs for the configured kite (as ``main``)."""
    art = dict(fpc.ARTIFICIAL)
    s_exit = 1.0 if np.sin(float(art["psi_exit"])) >= 0.0 else -1.0
    art["az_reelin_amp"] = s_exit * abs(float(art["az_reelin_amp"]))
    art["az_reelin_through"] = 0.0  # --reelin smooth (default)
    return art


def dubins_knobs(n_halves):
    """Knobs of ``--reelin dubins --loops n_halves`` (as ``main``)."""
    art = dict(fpc.ARTIFICIAL)
    art["bow_shape"] = "dubins"
    art["n_halves"] = int(n_halves)
    art["n_loops"] = (0.5 * int(n_halves) + 1.0) / float(art["reelout_fraction"])
    if fpc.MIN_TURN_RADIUS:
        art["reelin_min_turn_radius"] = float(fpc.MIN_TURN_RADIUS)
    return art


def target_angles(art, s):
    return full_cycle_angles(
        s,
        n_loops=art["n_loops"],
        reelout_fraction=art["reelout_fraction"],
        beta0=art["beta0"],
        beta_amp0=art["beta_amp0"],
        az_amp0=art["az_amp0"],
        beta_reelin_peak=art["beta_reelin_peak"],
        az_reelin_amp=art["az_reelin_amp"],
        az_reelin_through=art.get("az_reelin_through", 0.0),
        reelin_cross_pos=art.get("reelin_cross_pos", 0.7),
        ramp_fraction=art["ramp_fraction"],
        reelin_center=art.get("reelin_center", 0.5),
        psi0=art.get("psi0", 0.0),
        psi_entry=art.get("psi_entry"),
        psi_exit=art.get("psi_exit"),
        bow_shape=art.get("bow_shape", "sym"),
        lobe_handover_phase=art.get("lobe_handover_phase", pp.LOBE_HANDOVER_PHASE),
        n_halves=art.get("n_halves"),
        r0=fpc.R0,
        reelin_turn_radius=art.get("reelin_turn_radius", pp.DUBINS_TURN_RADIUS_M),
        reelin_min_turn_radius=art.get("reelin_min_turn_radius"),
        downloops=True,
    )


def shade_window(ax, s, mask, **kw):
    idx = np.flatnonzero(mask)
    if not idx.size:
        return
    for chunk in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1):
        ax.axvspan(s[chunk[0]], s[chunk[-1]], color=C_REELIN, alpha=0.10, lw=0, **kw)


def plot_path(ax, phi, beta, mask=None, color=C_OPT, lw=1.5, label=None, **kw):
    """Wind-window path, reel-in part (mask) drawn in the reel-in colour."""
    ax.plot(np.degrees(phi), np.degrees(beta), color=color, lw=lw, label=label, **kw)
    if mask is not None and mask.any():
        idx = np.flatnonzero(mask)
        for chunk in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1):
            ax.plot(
                np.degrees(phi[chunk]), np.degrees(beta[chunk]),
                color=C_REELIN, lw=lw, **kw,
            )


def window_axes(ax):
    ax.set_xlabel(r"azimuth $\phi$ (deg)")
    ax.set_ylabel(r"elevation $\beta$ (deg)")
    ax.set_aspect("equal", adjustable="datalim")


# ---------------------------------------------------------------------------
# Method figures
# ---------------------------------------------------------------------------
def fig_window(art):
    """Reel-in window, synthetic depower and the lobe path angles over s."""
    s = np.linspace(0.0, 1.0, 1500)
    phi, beta = target_angles(art, s)
    bump = reelin_bump(
        s, art["reelout_fraction"], art["ramp_fraction"], art.get("reelin_center", 0.5)
    )
    u_p = fpc._synthetic_depower(
        s, art["reelout_fraction"], art["ramp_fraction"], 1.0,
        art.get("reelin_center", 0.5),
    )
    mask = bump > 1e-6
    fig, axs = plt.subplots(3, 1, figsize=(WIDTH, 3.6), sharex=True)
    axs[0].plot(s, np.degrees(phi), color=C_OPT, label=r"$\phi$")
    axs[0].plot(s, np.degrees(beta), color=C_AUX, label=r"$\beta$")
    axs[0].set_ylabel("angle (deg)")
    axs[0].legend(loc="upper left", ncol=2, frameon=False)
    axs[1].plot(s, bump, color=C_REELIN)
    axs[1].set_ylabel(r"$w(s)$")
    axs[2].plot(s, u_p, color="k")
    axs[2].set_ylabel(r"$u_p$ (m)")
    axs[2].set_xlabel(r"path parameter $s$")
    for ax in axs:
        shade_window(ax, s, mask)
    axs[2].set_xlim(0, 1)
    fig.align_ylabels(axs)
    save(fig, "seed_window.pdf")


def fig_lobe_seed(art):
    """Lobe seed: sketch fit -> designed reel-in -> faired spline, curvature."""
    limit = fpc.CURVATURE_LIMIT_1PM
    raw = fpc._artificial_path_parameters(art)
    designed, des = design_reelin_spline(
        raw, limit,
        reelin_center=art.get("reelin_center", 0.5),
        reelout_fraction=art["reelout_fraction"],
        peak_elevation=art.get("beta_reelin_peak"),
    )
    faired, fair = fair_periodic_spline_to_curvature_limit(designed, limit)
    s = np.linspace(0.0, 1.0, 4000, endpoint=False)
    mask = window_mask(s, art)
    phi_t, beta_t = target_angles(art, s)

    macro("lobeM", int(raw["M"]), "d")
    macro("lobeKappaSketch", des["max_before"], ".3f")
    macro("lobeKappaDesigned", des["max_after"], ".3f")
    macro("lobeKappaFaired", fair["max_after"], ".3f")
    macro("lobeDesignFreed", len(des["freed"]), "d")
    macro("lobeDesignMoveDeg", np.degrees(des["max_path_move"]), ".1f")
    macro("lobePeakTargetDeg", np.degrees(des["peak_target"]), ".1f")
    macro("lobePeakAchievedDeg", np.degrees(des["peak_achieved"]), ".1f")
    macro("lobeFairTouched", len(fair["touched"]), "d")
    macro("lobeFairMoveDeg", np.degrees(fair["max_path_move"]), ".2f")

    fig = plt.figure(figsize=(WIDTH, 3.5))
    gs = fig.add_gridspec(1, 2, width_ratios=(1.15, 1.0), wspace=0.3)
    ax = fig.add_subplot(gs[0])
    ax.plot(np.degrees(phi_t), np.degrees(beta_t), color="0.7", lw=2.6,
            label="analytic sketch", solid_capstyle="round")
    phi_r, beta_r = spline_angles(raw, s)
    ax.plot(np.degrees(phi_r), np.degrees(beta_r), color="k", lw=0.8, ls="--",
            label="spline fit of the sketch")
    phi_f, beta_f = spline_angles(faired, s)
    plot_path(ax, phi_f, beta_f, mask, lw=1.4, label="designed + faired")
    ax.plot(np.degrees(faired["C_phi"]), np.degrees(faired["C_beta"]), "o",
            ms=2.0, color="0.35", label=f"control points ($M={raw['M']}$)")
    ax.plot(np.degrees(phi_f[0]), np.degrees(beta_f[0]), "s", ms=4, color="k")
    ax.annotate(r"$s=0$", (np.degrees(phi_f[0]), np.degrees(beta_f[0])),
                xytext=(4, -9), textcoords="offset points", fontsize=7.5)
    window_axes(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2,
              frameon=False, fontsize=6.8)

    ax2 = fig.add_subplot(gs[1])
    for path, kw in (
        (raw, dict(color="k", lw=0.8, ls="--", label="sketch fit")),
        (designed, dict(color=C_AUX, lw=1.0, label="designed reel-in")),
        (faired, dict(color=C_OPT, lw=1.3, label="+ fairing")),
    ):
        ax2.plot(s, spline_curvature(path, s), **kw)
    ax2.axhline(limit, color=C_REELIN, lw=0.9, ls=":", label=r"limit $\kappa_{\max}$")
    shade_window(ax2, s, mask)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1.15 * max(des["max_before"], limit))
    ax2.set_xlabel(r"path parameter $s$")
    ax2.set_ylabel(r"curvature $\kappa$ (1/m)")
    ax2.legend(loc="upper right", frameon=False)
    save(fig, "seed_lobe.pdf")
    return faired


def fig_dubins_seed(art):
    """Dubins reel-in: path in the wind window and turn radius along it."""
    s = np.linspace(0.0, 1.0, 20000, endpoint=False)
    pp._DUBINS_REELIN_CACHE.clear()
    phi, beta = target_angles(art, s)
    # The radius/apex scan's choice: (unit-sphere geodesic curvature, length)
    # pieces of the shortest reel-in, cached by dubins_full_cycle_angles.
    (pieces,) = pp._DUBINS_REELIN_CACHE.values()
    r0 = float(fpc.R0)
    arcs = [(k, ln) for k, ln in pieces if abs(k) > 1e-9]
    radius = r0 / abs(arcs[0][0])
    macro("dubinsHalves", int(art["n_halves"]), "d")
    macro("dubinsRadius", radius, ".0f")
    macro("dubinsRadiusDefault", pp.DUBINS_TURN_RADIUS_M, ".0f")
    macro("dubinsLength", r0 * sum(ln for _, ln in pieces), ".0f")
    macro("dubinsTurningDeg", np.degrees(sum(abs(k) * ln for k, ln in pieces)), ".0f")
    macro("dubinsPieces", len(pieces), "d")
    macro("dubinsApexDeg", np.degrees(art["beta_reelin_peak"]), ".1f")

    f = float(art["reelout_fraction"])
    c = float(art.get("reelin_center", 0.5))
    u = (s - c - 0.5 * (1.0 - f)) % 1.0
    reel_in = u > f
    R = sampled_turn_radius(phi, beta, r0)

    fig = plt.figure(figsize=(WIDTH, 3.3))
    gs = fig.add_gridspec(1, 2, width_ratios=(1.15, 1.0), wspace=0.3)
    ax = fig.add_subplot(gs[0])
    order = np.argsort(u)
    ph, be, ri = phi[order], beta[order], reel_in[order]
    ax.plot(np.degrees(ph[~ri]), np.degrees(be[~ri]), color=C_OPT, lw=1.3,
            label="reel-out figures")
    # Reel-in coloured by piece: arcs vs great-circle straights.
    Rr = R[order][ri]
    straight = ~(Rr < 3.0 * radius)
    phr, ber = np.degrees(ph[ri]), np.degrees(be[ri])
    seg_arc = np.where(straight, np.nan, ber)
    seg_str = np.where(straight, ber, np.nan)
    ax.plot(phr, seg_arc, color=C_REELIN, lw=2.0, label=f"arc, $R={radius:.0f}$ m")
    ax.plot(phr, seg_str, color=C_AUX, lw=2.0, label="great circle")
    i_apex = int(np.argmax(ber))
    for idx, name, off in ((0, "peel-off", (-34, -12)), (-1, "landing", (6, -10)),
                           (i_apex, "apex", (-10, 5))):
        ax.plot(phr[idx], ber[idx], "o", ms=3.5, color="k")
        ax.annotate(name, (phr[idx], ber[idx]), xytext=off,
                    textcoords="offset points", fontsize=7.5)
    window_axes(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3,
              frameon=False, fontsize=6.8)

    ax2 = fig.add_subplot(gs[1])
    xi =(u[order][ri] - f) / (1.0 - f)
    ax2.plot(xi, 1.0 / Rr, color=C_REELIN, lw=1.2)
    ax2.axhline(1.0 / radius, color="k", lw=0.7, ls=":")
    ax2.annotate(f"$1/R$, $R={radius:.0f}$ m", (0.02, 1.0 / radius),
                 xytext=(0, 4), textcoords="offset points", fontsize=7.5)
    ax2.set_ylim(0, 1.35 / radius)
    ax2.set_xlim(0, 1)
    ax2.set_xlabel(r"reel-in progress $\xi$")
    ax2.set_ylabel(r"geodesic curvature $k_g/r_0$ (1/m)")
    save(fig, "seed_dubins.pdf")


def fig_winch_law():
    """The depower-shifted linear winch law with its soft clamps."""
    import casadi as ca

    from awetrim.system.winch import Winch

    law = fpc._winch_law_with_hardware_rating(fpc.WINCH_LAW)
    cfg = dict(law, force_model="linear", softplus=True, softplus_beta=1e-3,
               softminus=True, softminus_beta=1e-3)
    winch = Winch(pattern_config=cfg)
    v = ca.MX.sym("v")
    u = ca.MX.sym("u")
    T = ca.Function("T", [v, u], [winch.tension_curve(v, input_depower=u)])
    lo, hi = fpc.DEPOWER_BAND
    vr = np.linspace(-8.0, 4.0, 400)
    fig, ax = plt.subplots(figsize=(0.62 * WIDTH, 2.5))
    cmap = plt.get_cmap("viridis")
    for j, up in enumerate(np.linspace(lo, hi, 5)):
        ax.plot(vr, np.asarray(T(vr, up)).ravel() / 1e3, color=cmap(j / 4),
                label=f"$u_p={up:.1f}$ m")
    for lim in (law["min_tether_force"], law["max_tether_force"]):
        ax.axhline(lim / 1e3, color="0.5", lw=0.6, ls=":")
    ax.axvline(0.0, color="0.5", lw=0.6)
    ax.set_xlabel(r"reel speed $v_r$ (m/s)")
    ax.set_ylabel(r"tension $T$ (kN)")
    ax.legend(frameon=False, loc="upper left")
    save(fig, "winch_law.pdf")
    macro("winchSlope", law["slope_winch_ro"], ".0f")
    macro("winchOffset", law["offset_winch_ro"], ".3f")
    macro("winchGain", law["winch_offset_depower_gain"], ".2f")
    macro("winchRef", law["winch_depower_ref"], ".1f")
    macro("winchTmin", law["min_tether_force"], ".0f")
    macro("winchTmax", law["max_tether_force"], ".0f")
    macro("upPowered", lo, ".1f")
    macro("upDepowered", hi, ".1f")


def method_macros(kite):
    art = lobe_knobs()
    macro("kiteName", kite["name"])
    macro("seedRzero", fpc.R0, ".1f")
    macro("seedReeloutFraction", art["reelout_fraction"], ".2f")
    macro("seedBetaZeroDeg", np.degrees(art["beta0"]), ".0f")
    macro("seedBetaAmpDeg", np.degrees(art["beta_amp0"]), ".1f")
    macro("seedAzAmpDeg", np.degrees(art["az_amp0"]), ".1f")
    macro("seedBetaPeakDeg", np.degrees(art["beta_reelin_peak"]), ".1f")
    macro("seedRampFraction", art["ramp_fraction"], ".2f")
    macro("seedAzReelinDeg", np.degrees(abs(art["az_reelin_amp"])), ".1f")
    macro("seedHandoverPhase", art["lobe_handover_phase"], ".1f")
    macro("curvatureLimit", fpc.CURVATURE_LIMIT_1PM, ".2f")
    macro("curvatureRadius", 1.0 / fpc.CURVATURE_LIMIT_1PM, ".1f")
    macro("closureTol", fpc.CLOSURE_TOL_M, "d")
    macro("steerSlack", fpc.AUTO_STEER_SLACK, ".2f")
    macro("autoMaxIter", fpc.AUTO_MAX_ITER, "d")
    macro("nodesPerSecond", fpc.NPOINTS_PER_SECOND, ".0f")
    macro("mPerFigure", fpc.M_PER_FIGURE, "d")
    macro("mReelinWindow", fpc.M_REELIN_WINDOW, "d")
    macro("trustMain", rfco.STAGES[0]["trust_region_weight"], "g")
    macro("trustPolish", rfco.STAGES[1]["trust_region_weight"], "g")
    macro("boxSpline", rfco.STAGES[0]["step_bounds"]["C_phi"], "g")
    macro("boxRzero", rfco.STAGES[0]["step_bounds"]["r0"], "g")
    macro("maxIter", rfco.MAX_ITER, "d")
    lim = nlp_limits(kite)
    macro("limSteer", lim["input_steering"][1], "g")
    macro("limSteerRate", lim["steering_rate"][1], "g")
    macro("limDepowerRate", lim["depower_rate"][1], "g")
    macro("limVrMin", lim["speed_radial"][0], "g")
    macro("limVrMax", lim["speed_radial"][1], "g")
    macro("limWinchAcc", lim["winch_acceleration"][1], "g")
    macro("limAoaMaxDeg", np.degrees(lim["angle_of_attack"][1]), ".0f")
    macro("limHeightMin", lim["height"][0], "g")
    macro("limRzeroMin", lim["r0"][0], "g")
    macro("limRzeroMax", lim["r0"][1], "g")
    macro("limBetaCoeffMaxDeg", np.degrees(lim["C_beta"][1]), ".0f")
    w = kite["wind"]
    macro("windRef", w["speed_wind_ref"], "g")
    macro("windHeight", w["height_ref"], "g")
    macro("windLaw", w["model_type"])
    return art


def nlp_limits(kite):
    """The bounds the NLP applies: defaults < system.yaml hardware < the
    seed's opti_limits_override (same precedence as ``opti_phase``)."""
    from awetrim.system.factory import create_system_model_from_yaml
    from awetrim.utils.defaults import DEFAULT_OPTI_LIMITS

    sm = create_system_model_from_yaml(yaml_path=kite["system_config"])
    lim = dict(DEFAULT_OPTI_LIMITS)
    lim.update(sm.hardware_limits or {})
    lim.update(fpc.OPTI_LIMITS_OVERRIDE)
    return lim


# ---------------------------------------------------------------------------
# Result figures
# ---------------------------------------------------------------------------
def load_seed_run(label, kite_dir, logs):
    """A generated seed: config, forward simulation (cached) and its log."""
    kite = resolve_kite(kite_dir)
    fpc.configure_kite(kite_dir)
    seed_path = Path(kite["cycle_config_dir"]) / SEED_FILENAME
    seed_cfg = yaml.safe_load(seed_path.read_text())
    results = REPO_ROOT / kite["results_dir"]
    run = {"label": label, "kite": kite, "seed_cfg": seed_cfg}
    cache = results / "seed_series.npz"
    if cache.is_file() and cache.stat().st_mtime > seed_path.stat().st_mtime:
        run["seed"] = dict(np.load(cache))
    else:
        phase, _ = fpc._simulate_cycle(seed_cfg)
        run["seed"] = series_from_phase(phase)
        results.mkdir(parents=True, exist_ok=True)
        np.savez(cache, **run["seed"])
    run["seed_log"] = _read(Path(logs) / f"seed_{label}.log") if logs else ""
    return run


def load_archived(stem):
    """An optimum written by run_full_cycle_opti.py (``<stem>.csv/.yaml``)."""
    import pandas as pd

    stem = Path(stem)
    df = pd.read_csv(f"{stem}.csv")  # the wind tag holds dots (z0_0.0002)
    cfg = yaml.safe_load(Path(f"{stem}.yaml").read_text())
    cfg = cfg.get("reelout", cfg)
    traj = {k: df[k].to_numpy(float) for k in df.columns}

    class _R:  # the series helper reads ``optimized_trajectory`` only
        optimized_trajectory = traj

    return {
        "stem": stem,
        "cfg": cfg,
        "opt": series_from_optimizer_result(_R, cfg["path_parameters"]),
        "turn_radius": traj.get("turn_radius"),
    }


def _read(path):
    return path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""


def _reelin_mask_from_depower(cfg, s):
    u_p = np.asarray(cfg["sim_parameters"]["input_depower_profile"], float)
    return np.interp(s, np.linspace(0, 1, u_p.size), u_p) > np.mean(fpc.DEPOWER_BAND)


def auto_trace(run):
    """--auto iterations: (iteration, symptom, knob move) from the seed log."""
    rows = []
    for line in run["seed_log"].splitlines():
        m = re.match(r"\[auto\] it (\d+): (.*) -> (\S+ \S+ -> \S+)$", line.strip())
        if m:
            rows.append((int(m.group(1)), m.group(2), m.group(3)))
    stuck = re.search(r"\[auto\] stuck: (.*?);", run["seed_log"])
    conv = re.search(r"\[auto\] converged after (\d+) iteration", run["seed_log"])
    return rows, (stuck.group(1) if stuck else None), (int(conv.group(1)) if conv else None)


def _wall_minutes(log):
    real = re.search(r"real\s+(\d+)m([\d.]+)s", log)
    return None if real is None else (60 * int(real.group(1)) + float(real.group(2))) / 60


def fig_seeds(runs):
    """The generated seeds in the wind window."""
    fig, axs = plt.subplots(1, len(runs), figsize=(WIDTH, 2.9), squeeze=False)
    for ax, run in zip(axs[0], runs):
        reel = run["seed_cfg"]["reelout"]
        s = np.linspace(0, 1, 3000, endpoint=False)
        phi, beta = spline_angles(reel["path_parameters"], s)
        plot_path(ax, phi, beta, _reelin_mask_from_depower(reel, s), lw=1.3)
        ax.set_title(f"{run['label']} seed", fontsize=9)
        window_axes(ax)
    save(fig, "results_seeds.pdf")


def fig_seed_series(runs, limits):
    """Forward simulations of the seeds (the NLP warm starts) over time."""
    keys = (
        ("distance_radial", r"$r$ (m)", 1.0, None),
        ("speed_radial", r"$v_r$ (m/s)", 1.0, None),
        ("tension_tether_ground", r"$T$ (kN)", 1e-3, None),
        ("input_steering", r"$u_s$ (-)", 1.0, limits["input_steering"]),
        ("input_depower", r"$u_p$ (m)", 1.0, None),
        ("angle_of_attack", r"$\alpha$ (deg)", 180 / np.pi,
         tuple(np.degrees(limits["angle_of_attack"]))),
    )
    colors = (C_OPT, C_REELIN, C_AUX)
    fig, axs = plt.subplots(len(keys), 1, figsize=(WIDTH, 5.8), sharex=True)
    for run, col in zip(runs, colors):
        ser = run["seed"]
        t = ser["t"] - ser["t"][0]
        for ax, (k, lab, sc, _) in zip(axs, keys):
            if k in ser:
                n = min(len(t), len(ser[k]))
                ax.plot(t[:n], ser[k][:n] * sc, color=col, lw=1.0, label=run["label"])
            ax.set_ylabel(lab)
    for ax, (_, _, sc, bounds) in zip(axs, keys):
        if bounds is not None:
            for b in bounds:
                ax.axhline(b * (1.0 if sc == 180 / np.pi else sc), color="0.4",
                           lw=0.6, ls=":")
    axs[-1].set_xlabel("time (s)")
    axs[0].legend(loc="lower right", ncol=len(runs), frameon=False)
    fig.align_ylabels(axs)
    save(fig, "results_seed_series.pdf")


def seeds_table(runs, limits):
    """Per-seed numbers: resolution, feasibility and cycle metrics."""
    cols = []
    for run in runs:
        reel = run["seed_cfg"]["reelout"]
        ser = run["seed"]
        m = cycle_metrics(ser)
        aoa = ser.get("angle_of_attack")
        rows, stuck, conv = auto_trace(run)
        n_nodes = int(reel["sim_parameters"]["n_points"])
        kappa = fpc.path_curvature_metrics(reel["path_parameters"])["max_physical"]
        cols.append({
            "$M$ / $N$": f"${int(reel['path_parameters']['M'])}$ / ${n_nodes}$",
            "nodes trimmed": f"${len(ser['distance_radial'])}/{n_nodes}$",
            "max curvature (1/m)": f"${kappa:.3f}$",
            "cycle duration (s)": f"${m['duration']:.1f}$",
            "average power (kW)": f"${m['avg_power'] / 1e3:.2f}$",
            "closure $r_\\mathrm{end}-r_0$ (m)": f"${m['closure']:+.1f}$",
            "$u_s$ range (-)": f"${ser['input_steering'].min():+.2f}$ / "
                               f"${ser['input_steering'].max():+.2f}$",
            "max $\\alpha$ (deg)": "--" if aoa is None
                                   else f"${np.degrees(np.max(aoa)):.1f}$",
            "max tension (kN)": f"${m['tension_max'] / 1e3:.2f}$",
            "tuner iterations": "--" if not run["seed_log"] or "[auto]" not in run["seed_log"]
                                else str(len(rows) + 1),
            "generation time (min)": "--" if _wall_minutes(run["seed_log"]) is None
                                     else f"${_wall_minutes(run['seed_log']):.0f}$",
        })
        tag = re.sub(r"[^A-Za-z]", "", run["label"]).capitalize()
        macro(f"seed{tag}Power", m["avg_power"] / 1e3, ".2f")
        macro(f"seed{tag}Duration", m["duration"], ".1f")
        macro(f"seed{tag}Closure", m["closure"], "+.1f")
        macro(f"seed{tag}Nodes", n_nodes, "d")
        macro(f"seed{tag}M", int(reel["path_parameters"]["M"]), "d")
        macro(f"seed{tag}SteerMin", ser["input_steering"].min(), "+.2f")
        macro(f"seed{tag}SteerMax", ser["input_steering"].max(), "+.2f")
        if aoa is not None:
            macro(f"seed{tag}AoaMaxDeg", np.degrees(np.max(aoa)), ".1f")
        if _wall_minutes(run["seed_log"]) is not None:
            macro(f"seed{tag}WallMin", _wall_minutes(run["seed_log"]), ".0f")
        if run["seed_log"] and "[auto]" in run["seed_log"]:
            macro(f"auto{tag}Moves", len(rows), "d")
            macro(f"auto{tag}Stuck", (stuck or "").replace("_", r"\_"))
    lines = [
        "% GENERATED by report_cycle_optimisation.py",
        "\\begin{tabular}{l" + "r" * len(runs) + "}",
        "\\toprule",
        "quantity & " + " & ".join(f"{r['label']} seed" for r in runs) + " \\\\",
        "\\midrule",
    ]
    for key in cols[0]:
        lines.append(f"{key} & " + " & ".join(c[key] for c in cols) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (GEN / "seeds.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("  generated/seeds.tex")


def auto_trace_table(run):
    rows, stuck, conv = auto_trace(run)
    lines = [
        "% GENERATED by report_cycle_optimisation.py",
        "\\begin{tabular}{rp{0.5\\textwidth}l}",
        "\\toprule",
        "it. & dominant symptom & knob move \\\\",
        "\\midrule",
    ]

    def tex(text):
        text = text.replace("_", "\\_").replace("%", "\\%")
        return re.sub(r"(-?\d+\.\d+)\s*->\s*(-?\d+\.\d+)", r"$\1\\to\2$", text)

    for it, why, move in rows:
        lines.append(f"{it} & {tex(why)} & {tex(move)} \\\\")
    if stuck:
        lines.append(f"{len(rows) + 1} & {tex(stuck)} & stop, best iterate \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (GEN / "auto_trace.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("  generated/auto_trace.tex")


def fig_archived(arch, limits):
    """The archived optimum: wind window and the per-node decisions over time."""
    o = arch["opt"]
    t = o["t"] - o["t"][0]
    reel = o["speed_radial"] < 0.0
    fig = plt.figure(figsize=(WIDTH, 4.6))
    gs = fig.add_gridspec(4, 2, width_ratios=(1.0, 1.25), wspace=0.35, hspace=0.15)
    ax = fig.add_subplot(gs[:, 0])
    s = np.linspace(0, 1, 3000, endpoint=False)
    phi, beta = spline_angles(arch["cfg"]["path_parameters"], s)
    ax.plot(np.degrees(phi), np.degrees(beta), color="0.75", lw=2.5,
            label="optimised spline")
    plot_path(ax, o["angle_azimuth"], o["angle_elevation"], reel, lw=0.0,
              marker=".", ms=2.5, label="nodes")
    window_axes(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, frameon=False)
    keys = (
        ("speed_radial", r"$v_r$ (m/s)", 1.0),
        ("tension_tether_ground", r"$T$ (kN)", 1e-3),
        ("input_depower", r"$u_p$ (m)", 1.0),
        ("s_dot", r"$\dot s$ (1/s)", 1.0),
    )
    axs = [fig.add_subplot(gs[j, 1]) for j in range(4)]
    for a, (k, lab, sc) in zip(axs, keys):
        a.plot(t, o[k] * sc, color=C_OPT, lw=1.1)
        a.set_ylabel(lab)
        shade_window(a, t, reel)
        if a is not axs[-1]:
            a.tick_params(labelbottom=False)
    axs[-1].axhline(0.01, color=C_REELIN, lw=0.8, ls=":")
    axs[-1].set_yscale("log")
    axs[-1].set_xlabel("time (s)")
    fig.align_ylabels(axs)
    save(fig, "results_archived.pdf")

    m = cycle_metrics(o)
    floor = int(np.sum(o["s_dot"] < 0.0101))
    macro("archPower", m["avg_power"] / 1e3, ".2f")
    macro("archEnergy", m["energy"] / 1e6, ".2f")
    macro("archDuration", m["duration"], ".1f")
    macro("archReeloutFraction", m["reelout_time_fraction"], ".2f")
    macro("archTensionMax", m["tension_max"] / 1e3, ".2f")
    macro("archTensionMean", m["tension_mean"] / 1e3, ".2f")
    macro("archRmin", m["r_min"], ".0f")
    macro("archRmax", m["r_max"], ".0f")
    macro("archRzero", float(arch["cfg"]["path_parameters"]["r0"]), ".0f")
    macro("archClosure", abs(m["closure"]), ".2f")
    macro("archSteerMax", m["steer_absmax"], ".2f")
    macro("archDepMin", m["depower_min"], ".2f")
    macro("archDepMax", m["depower_max"], ".2f")
    macro("archVrMin", o["speed_radial"].min(), ".1f")
    macro("archVrMax", o["speed_radial"].max(), ".1f")
    macro("archM", int(arch["cfg"]["path_parameters"]["M"]), "d")
    macro("archNodes", len(o["s"]), "d")
    macro("archFloorNodes", floor, "d")
    macro("archFloorReelin", int(np.sum((o["s_dot"] < 0.0101) & reel)), "d")
    macro("archSdotMax", float(np.max(o["s_dot"])), ".2f")
    macro("archSdotMedian", float(np.median(o["s_dot"])), ".4f")
    macro("archSdotMaxS", float(o["s"][int(np.argmax(o["s_dot"]))]), ".3f")
    if arch.get("turn_radius") is not None:
        macro("archTurnRadiusMin", float(np.min(arch["turn_radius"])), ".1f")


def copy_archived_figures(arch):
    """The comparison figures the archived run wrote itself (seed vs optimum)."""
    import shutil

    tag = arch["stem"].name.replace("full_cycle_optimized_", "")
    for kind in ("power",):
        src = arch["stem"].parent / f"full_cycle_compare_{tag}_{kind}.png"
        if src.is_file():
            FIG.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, FIG / f"archived_{kind}.png")
            print(f"  figures/archived_{kind}.png")


# ---------------------------------------------------------------------------
def write_macros():
    GEN.mkdir(parents=True, exist_ok=True)
    lines = ["% GENERATED by scripts/reduced-order-model/optimization/cycle/"
             "report_cycle_optimisation.py"]
    for name, text in MACROS.items():
        lines.append(f"\\newcommand{{\\{name}}}{{{text}}}")
    (GEN / "macros.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("  generated/macros.tex")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kite", default="LEI-V3-KITE",
                    help="kite whose profile drives the method figures")
    ap.add_argument("--run", action="append", default=[], metavar="LABEL=KITE_DIR",
                    help="a generated seed (kite folder holding it); repeatable")
    ap.add_argument("--logs", default=None,
                    help="folder with the seed_<LABEL>.log of each --run")
    ap.add_argument("--archived", default=None, metavar="STEM",
                    help="an optimum written by run_full_cycle_opti.py, as the "
                    "path stem of its .csv/.yaml (results/<kite>/optimization/"
                    "full_cycle/full_cycle_optimized_<wind tag>)")
    ap.add_argument("--dubins-loops", type=int, default=7,
                    help="visible half-figures of the Dubins method figure")
    args = ap.parse_args()

    GEN.mkdir(parents=True, exist_ok=True)
    kite = fpc.configure_kite(args.kite)
    print(f"Method figures ({kite['name']}):")
    art = method_macros(kite)
    fig_window(art)
    fig_lobe_seed(art)
    fig_dubins_seed(dubins_knobs(args.dubins_loops))
    fig_winch_law()
    limits = nlp_limits(kite)

    runs = []
    for spec in args.run:
        label, _, path = spec.partition("=")
        print(f"Seed {label}: {path}")
        runs.append(load_seed_run(label, path, args.logs))
    if runs:
        print("Seed figures:")
        fig_seeds(runs)
        fig_seed_series(runs, limits)
        seeds_table(runs, limits)
        auto_trace_table(runs[0])
    if args.archived:
        print(f"Archived optimum: {args.archived}")
        arch = load_archived(args.archived)
        fig_archived(arch, limits)
        copy_archived_figures(arch)
    write_macros()


if __name__ == "__main__":
    main()

"""Generate a *feasible* initial cycle config for the full-cycle optimization.

Produces everything ``run_full_cycle_opti.py`` needs to optimize a whole
pumping cycle as ONE periodic ``spline_periodic`` phase:

  * a **periodic** cubic B-spline ``(azimuth, elevation)`` path — one period ==
    one full pumping cycle (reel-out figure-eights + a reel-in arc). By default
    the reel-in is flown as ONE giant lobe of the figure, the shape the
    full-cycle optimiser converges to (``bow_shape="lobe"`` in
    ``full_cycle_angles``): the kite peels off a climbing centre crossing a
    bit before azimuth 0, climbs straight up the middle of the wind window,
    crosses the top and comes down on the OTHER side -- re-entering the
    figures on the opposite side from where it went out -- to land
    tangentially on the lobe extreme, flying the lower half of that lobe back
    into the figures -- no figure wiggle during the handover (the figure
    phase is frozen through the reel-in);
  * a synthetic depower profile aligned with the reel-in window; and
  * a linear winch law with a **depower-dependent offset** ``offset(l_dp) =
    offset0 + gain * (l_dp - l_dp_ref)`` so the *same* force law reels out when
    powered and reels in when depowered. This is the key that lets the whole
    cycle live in one phase (see ``Winch.tension_curve``).

By default (``SHAPE_SOURCE = "artificial"``) the trajectory is fully synthetic
— smooth, exactly periodic, tuned for trim feasibility — and NO flight data is
read; the winch law and start radius are data-derived constants (``WINCH_LAW``,
``R0``). ``SHAPE_SOURCE = "experimental"`` instead fits spline + winch law to
one good EKF-reconstructed cycle (kept for reference; the raw fit is usually
too rough for the QS trim to follow).

The result is written to
``data/LEI-V3-KITE/cycle_configs/full_cycle_periodic_from_exp.yaml`` and
checked with the same validator ``run_full_cycle_opti.py`` applies on load.

The max sampled path curvature must stay under ``CURVATURE_LIMIT_1PM``
(0.08 1/m). The parametric reel-in is only a SKETCH: when its fitted spline
violates the limit, the reel-in is DESIGNED between the frozen exit/entry
handover states (``design_reelin_spline``) -- the window control points
solve a spline-in-tension NLP that hits ``beta_reelin_peak`` as a target
with curvature bounded as a hard constraint; the figures, the handover
states and every knob you set stay exactly as configured. A residual
fairing pass (``fair_periodic_spline_to_curvature_limit``) then catches
anything outside the designed window. Purely geometric, no forward sims.

Resolution is derived, not hand-set: ``M`` keeps >= 10 control points per
figure-eight AND >= 15 across the reel-in window (measured convergence knees,
see ``_resolve_artificial_M``), and whenever a forward sim runs (``--check``/
``--auto``/``--close``) ``n_points`` is recalibrated to NPOINTS_PER_SECOND
nodes per second of the MEASURED cycle duration -- the ``CYCLE_DURATION_S``
prior can be over 2x off at other winds/shapes.

Usage:
    python .../fit_periodic_cycle_config.py [--auto] [--loops N]
                                            [--reelin smooth|cross]
                                            [--cross-at P]
                                            [--reentry opposite|same]
                                            [--check] [--close] [--plot]
                                            [--curvature-limit K]

    --auto   auto-tune the shape/depower knobs (one forward sim per
             iteration) until the seed is trim-feasible and radially closed
             at the wind in run_full_cycle_opti's WIND_CONFIG -- use this to
             generate a seed for ANY wind speed / loop count hands-free
    --loops  number of half figure-eights (lobes) VISIBLE during reel-out
             -- what you count on the sphere, honoured EXACTLY; the internal
             full-period n_loops and the parity-matching psi_entry are
             derived from it (full_cycle_n_loops_for_half_figures), and the
             spline fairing absorbs the sharper handover of a new count
    --reelin  how the reel-in gets from the peel-off to the tangential
             landing. The sides are NOT a choice: the landing is always
             tangential at a lobe extreme going down, so --loops parity
             ties the exit side to the entry side (odd = opposite, even =
             same), and flipping both would only mirror the whole cycle.
             "smooth" (default): top loop on the exit side -- one clean
             arc, no crossing. "cross": up the middle, out to the OPPOSITE
             side near the top, descend parked on that side, then cross
             az = 0 low to reach the exit azimuth
    --cross-at  "cross" only: fraction of the exit ramp where the az = 0
             crossing sits (0 = right after the top, 1 = at figure height;
             default 0.7)
    --reentry  requested exit side RELATIVE to the entry lobe ("opposite"
             or "same"). Tangential exits tie this to the --loops parity
             (odd = opposite, even = same), so the flag only validates and
             errors on a mismatch
    --check  forward-simulate the generated config and print a feasibility
             report (steering/AoA vs NLP bounds, radial closure, trim health);
             also recalibrates n_points to the measured cycle duration
    --close  only fit the reel-in depower depth (secant on forward sims)
             until the simulated cycle closes radially
    --plot   show the seed-path figure (wind-window view + az/el/depower vs
             s; it is ALWAYS saved as a PNG next to the optimizer results, see
             SEED_PLOT_PATH) and the overview plots of the simulated guess
    --curvature-limit  max path curvature in 1/m (default 0.08; 0 disables
             both the spline fairing and the write-time guard)
    --reelin-center  centre of the reel-in window in s; any value, mod 1
             (the window wraps the periodic seam). 0.5 (default) keeps s=0
             in steady reel-out; (1-f)/2 starts the reel-in at s=0,
             1-(1-f)/2 ends it there (s=0 = reel-out start), 0 puts s=0 at
             the reel-in top
"""

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import yaml

# Reuse the validated EKF/cycle/winch helpers rather than duplicating them.
# Locate them relative to the repo's top-level ``scripts`` dir so this works
# regardless of how deep under scripts/ (e.g. scripts/personal/...) this lives.
_here = Path(__file__).resolve()
_scripts_dir = next(p for p in _here.parents if p.name == "scripts")
_repo_root = _scripts_dir.parent
_SRC_DIR = _repo_root / "src"
_VALIDATION_DIR = _scripts_dir / "reduced-order-model" / "validation"
for _path in (_SRC_DIR, _VALIDATION_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
from validate_spline_v3 import (  # noqa: E402
    cycles_from_phases,
    read_results,
)

from awetrim.identification.controls import (  # noqa: E402
    ROM_DEPOWERED_INPUT_DEPOWER,
    ROM_POWERED_INPUT_DEPOWER,
    flight_dataframe_depower_to_power_tape_length,
)
from awetrim.kinematics.parametrized_patterns import (  # noqa: E402
    LOBE_HANDOVER_PHASE,
    design_reelin_spline,
    fair_periodic_spline_to_curvature_limit,
    fit_bspline_pattern_to_trajectory,
    full_cycle_angles,
    full_cycle_n_loops_for_half_figures,
    full_cycle_visible_half_figures,
    make_full_cycle_bspline_path_parameters,
)
from awetrim.utils.config_paths import LEI_V3_CYCLE_CONFIG_DIR  # noqa: E402
from awetrim.utils.defaults import DEFAULT_OPTI_LIMITS  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration -- edit for a different flight / cycle / resolution
# ---------------------------------------------------------------------------
FLIGHT = {"year": "2019", "month": "10", "day": "08", "kite_model": "LEI-V3-Kite"}
PATH_TO_MAIN = "./data/LEI-V3-KITE"
CYCLE_ID = 62  # a representative good cycle in the 2019-10-08 flight

# Periodic-spline control points and sim grid. M_PER_SECOND/M_MAX only apply
# to the experimental fit (the artificial M comes from _resolve_artificial_M).
# n_points is the optimizer grid over the whole cycle, at NPOINTS_PER_SECOND
# of SIMULATED cycle time: any forward sim (--check/--auto/--close) measures
# the real duration and recalibrates n_points (the CYCLE_DURATION_S prior can
# be far off -- the n_loops=5 seed assumed 213 s but flies in ~97 s at
# 13 m/s). Convergence sweep vs a 400-node reference (n=5, M=50): 3 nodes/s
# keeps mean cycle power within 0.2% with the steering peaks resolved;
# 2 nodes/s is the floor (-0.6% power, and the handover steering peaks start
# falling BETWEEN nodes -- the NLP only enforces bounds at nodes, so a too
# coarse grid hides violations instead of removing them); 1 node/s breaks the
# trim (AoA spikes). Capped for NLP tractability (nodes dominate NLP size).
M_PER_SECOND = 0.2
NPOINTS_PER_SECOND = 3.0
M_MAX = 90
N_POINTS_MAX = 2000

# Shape source for the periodic path spline:
#   "experimental" -- fit the spline to the flown (azimuth, elevation) of CYCLE_ID
#   "artificial"   -- a clean synthetic full cycle (figure-eights + reel-in arc).
# The data-fit winch law, r0 and the depower split are reused in both cases; only
# the path SHAPE (C_phi/C_beta) and the depower profile differ. The experimental
# fit can be too rough for the QS trim to follow; the artificial curve is smooth
# and exactly periodic, a far more feasible initial guess.
SHAPE_SOURCE = "artificial"

# Synthetic full-cycle parameters (used when SHAPE_SOURCE == "artificial").
ARTIFICIAL = {
    "M": 40,  # control-point FLOOR; resolved as max(M, per-figure floor,
    # reel-in-window floor) -- see _resolve_artificial_M. A fixed M
    # under-resolves high loop counts and the underfit RINGS (spline
    # curvature far above the raw curve's).
    "n_loops": 4,  # INTERNAL count: continuous figure-eights over the WHOLE
    # period (~reelout_fraction * n_loops visible; the pinned handover phases
    # snap the visible count -- 4 here shows 2.5 figures = 5 half-eights
    # during reel-out, the opposite-side handover parity). The --loops CLI
    # knob is the WYSIWYG unit (visible reel-out HALF-figures) and overrides
    # this via full_cycle_n_loops_for_half_figures,
    # which may leave a non-integer value here. (2 loops dwell too long at
    # the azimuth extremes -> steering blows up; 4 loops turn too fast)
    "reelout_fraction": 0.65,  # fraction of the period spent reeling out
    # (lower closes the cycle better but deepens the reel-in-edge steering)
    "beta0": 0.35,  # reel-out base elevation (rad ~ 20 deg)
    # Steering demand of a figure-eight ~ 4*pi / T_loop (course angle sweeps
    # +-2pi per lobe), so at the kite's natural speed the lever is arc length:
    # BIG figures turn gentler. Scale both amps together, az ~ 2.5*beta;
    # a.36/b.14 is the sweet spot (larger starts spiking at the window edges).
    "beta_amp0": 0.14,  # figure-eight elevation amplitude (rad)
    "az_amp0": 0.36,  # figure-eight azimuth amplitude (rad)
    "beta_reelin_peak": 1.4,  # reel-in peak elevation (rad). What you set is
    # what the raw curve does -- if the top U-turn is then too sharp for the
    # curvature limit, the SPLINE is locally faired around it (the knob
    # itself is never lowered). Too high starves the kite of apparent wind
    # at the top -> AoA blows past its limit
    "az_reelin_amp": -0.36,  # reel-in bow (rad). With bow_shape "lobe" this
    # is the azimuth the descent comes down on -- ALWAYS the psi_exit lobe's
    # side (negative = left), so the reel-in exits tangentially at a side of
    # the figure going down; main() normalizes the sign. |.| = az_amp0 lands
    # it vertically on the lobe extreme.
    "az_reelin_through": 0.36,  # azimuth (rad) the --reelin "cross" shape
    # descends along: up the middle, out to this side near the top, DOWN
    # parked on this azimuth, then cross az = 0 low (reelin_cross_pos) to
    # the az_reelin_amp side. Sign = side; 0 = "smooth" (top loop directly
    # on the landing side, nothing crosses). main() resolves the sign
    # (opposite the exit) or zeroes it; the magnitude is this knob.
    "reelin_cross_pos": 0.7,  # "cross" only: where on the descent the az = 0
    # crossing sits, as a fraction of the exit ramp (0 = right after the
    # top, 1 = at figure height). CLI: --cross-at.
    "ramp_fraction": 0.49,  # reel-in window edge width (0.5 = one smooth hump,
    # but then full depower is never held and the cycle under-reels)
    "reelin_center": 0.0,  # reel-in window centre in s (any value, mod 1: the
    # window wraps across the periodic seam). 0.5 keeps s = 0 (the periodic
    # seam AND the forward-trim start state) in steady reel-out; with
    # h = (1 - reelout_fraction)/2, h starts the reel-in exactly at s = 0,
    # 1 - h ends it there so s = 0 is the reel-out start, and 0 puts s = 0 at
    # the reel-in top. Moving the seam off mid-reel-out -> harder trim start.
    "psi0": 0.0,  # figure-phase offset (rad): sets WHERE in a figure-eight the
    # reel-in window fades the oscillation (the phase at the window edges is
    # what makes the handover sharp or smooth, and it scales with n_loops).
    # IGNORED while psi_entry/psi_exit below are pinned.
    # Designed handover (see full_cycle_angles, bow_shape="lobe"): the reel-in
    # is ONE giant lobe, the shape the full-cycle optimiser converges to, and
    # it re-enters the figures on the OPPOSITE side from where it went out.
    # The kite peels off the climbing centre crossing coming OUT of the right
    # lobe, a bit before azimuth 0 heading up-left (psi_entry = pi - 0.3 --
    # for "lobe" the phases are the FROZEN figure phases, held through the
    # whole reel-in so nothing wiggles on the way up), climbs to the top of
    # the zero meridian, crosses and comes down on the az_reelin_amp side to
    # land tangentially on the LEFT-lobe extreme heading down (psi_exit =
    # 3*pi/2), then flies the lower half of that lobe into the next crossing.
    # The reel-in thus replaces the top half of the left lobe itself (the
    # phase freezes at pi - 0.3 and resumes at 3*pi/2 of the SAME lobe), so
    # the giant lobe slots into the figure alternation instead of doubling a
    # side. The reel-in ALWAYS exits tangentially at a side of the figure
    # going down, so parity ties the exit side to the visible half-lobe
    # count (odd = opposite the peel-off lobe, even = same -- an even
    # --loops mirrors psi_entry back to 2*pi - 0.3), and flipping the
    # absolute orientation would only mirror the whole cycle (fixed
    # convention: exit left). The ONE real choice per count is --reelin:
    # "smooth" puts the top loop on the exit side (plain lobe,
    # az_reelin_through = 0, no crossing); "cross" puts it on the opposite
    # side, and the descent crosses the ascending leg at lower elevation on
    # its way to the exit azimuth. --reentry validates the requested
    # relative side against the parity.
    # Legacy shapes: bow_shape "sym" / "descent" read psi_entry/psi_exit as
    # the window EDGE phases (0 = climbing crossing heading up; 2*pi / pi =
    # same- / other-side re-entry); set BOTH phases to None for free psi0.
    "psi_entry": float(np.pi - 0.3),
    "psi_exit": float(1.5 * np.pi),
    "bow_shape": "lobe",
    # Phase budget (rad) the figure advances while the reel-in window fades it
    # out / back in. The freeze spans 2*budget/rate of the entry ramp and the
    # figure RATE grows with the loop count, so a fixed budget completes ever
    # earlier in the ramp (where the climb speed is ~0) and the peel-off bend
    # sharpens ~ linearly with n_loops (bigger budget = freeze stretched back
    # over the ramp; the visible lobe count is budget-independent -- the
    # spline fairing rounds whatever bend remains). Seed at the module
    # default.
    "lobe_handover_phase": float(LOBE_HANDOVER_PHASE),
}

# Data-derived constants used when SHAPE_SOURCE == "artificial", so the
# generator runs without any flight data. Regressed once from the 2019-10-08
# flight, cycle 62 (SHAPE_SOURCE = "experimental" re-derives them from the
# flight logs via _fit_winch_with_depower_offset).
R0 = 236.7  # start radius (m)
CYCLE_DURATION_S = 128.0  # cycle-duration PRIOR (s, the 3-loop flown
# reference); only sets n_points until a forward sim measures the actual
# duration (--check/--auto/--close recalibrate n_points from it -- at 13 m/s
# with 5 loops the prior said 213 s but the cycle flies in ~97 s)
WINCH_LAW = {
    "slope_winch_ro": 5380.8,
    "offset_winch_ro": 0.412,
    "winch_offset_depower_gain": -10.931,
    "winch_depower_ref": 1.7,
    "max_tether_force": 8400.4,
    "min_tether_force": 865.4,
}

# Reel-in depower depth: fraction of the powered->depowered span the synthetic
# bump reaches (1.0 = fully depowered, l_dp = 2.1, at the top). This is the
# single radial-closure knob: shallower depth -> slower reel-in -> larger end
# radius. ``--close`` fits it so the simulated cycle closes on itself; pin the
# printed value here to make the closed config reproducible without re-fitting.
DEPOWER_DEPTH = 1.0

# |r_end - r0| below which the simulated cycle counts as closed ENOUGH for a
# warm start (m). ~8 m on r0 ~ 237 m is 0.03 scaled infeasibility on the NLP's
# closure row -- trivially absorbed by stage 0. Chasing the last metres via
# reelout_fraction destabilizes the steering at the window edges, so don't.
CLOSURE_TOL_M = 10

# NLP winch mode emitted into the config:
#   "force_law"  -- the optimizer keeps the regressed tension curve as a hard
#                   per-node equality (the flown controller's form).
#   "free_speed" -- the reel speed becomes a direct, rate-limited control and
#                   the winch only bounds the holdable tension
#                   [min_tether_force, max_tether_force]. Better conditioned
#                   (no soft-clamp plateau killing the Jacobian at high wind)
#                   and radial closure becomes ~linear in v_r; the optimal
#                   T(v_r, l_dp) law is regressed from the RESULT afterwards.
# The forward seed simulation marches with the force law in both modes.
WINCH_MODE = "free_speed"
WINCH_ACCELERATION = [-2.0, 2.0]  # winch drive acceleration capability (m/s^2)

# Optimizer bound overrides emitted into the generated config. A full cycle
# reaches higher reel-in elevation than the default C_beta range, and
# winch_offset_depower_gain has no global default. run_full_cycle_opti.py reads
# these from the YAML -- this script is the single source of truth, so tune
# them here and regenerate instead of hand-editing the output file.
OPTI_LIMITS_OVERRIDE = {
    "C_phi": [-1.0, 1.0],
    "C_beta": [0.01, 1.4],  # up to ~80 deg elevation for the reel-in arc
    "winch_offset_depower_gain": [-20.0, 0.0],
}

# Output config (loaded verbatim by run_full_cycle_opti.py).
OUTPUT_PATH = LEI_V3_CYCLE_CONFIG_DIR / "full_cycle_periodic_from_exp.yaml"
# Seed-path figure (wind-window view + az/el/depower vs s), written on every
# successful generation next to the optimizer's own outputs (results/ is
# git-ignored; the YAML above is tracked). See plot_seed_path.
SEED_PLOT_PATH = (
    Path("results") / "LEI-V3-KITE" / "optimization" / "full_cycle" / "seed_path.png"
)

# Geometric creation-time guard (1/m): refuse to write a path sharper than
# this sampled max curvature. Every build enforces it ON THE FITTED SPLINE
# (fair_periodic_spline_to_curvature_limit: control points around a violation
# move minimally, the parametric knobs are never touched), so the limit holds
# for ANY --loops value -- the sharpest point of the synthetic cycle is the
# reel-out/reel-in handover, and how sharp it is depends strongly on n_loops.
# Override with --curvature-limit; 0 disables both the fairing and the guard.
CURVATURE_LIMIT_1PM = 0.08
CURVATURE_SAMPLES = 1000

# Magnitude bounds for the reel-in azimuth bow, used by the --auto trim lever
# (``bow_up`` in ``_auto_action``). The seed bow sits inside it; only the
# magnitude is bounded (the seed's sign/direction is kept).
AZ_REELIN_AMP_MAG_BOUNDS = (0.15, 0.80)


def _synthetic_depower(
    s_grid, reelout_fraction, ramp_fraction, depth=1.0, reelin_center=0.5
):
    """Powered during reel-out, depowered through the reel-in window.

    Uses the SAME reel-in window as the path shape (``reelin_bump`` with the
    SAME ``reelin_center`` and ``ramp_fraction``), so the kite is depowered
    exactly where it is flown up to high elevation. A mismatch here (depowering
    at a different s, or
    over a different width, than the elevation arc) leaves the kite powered while
    parked high -> the QS trim goes infeasible at the transition.

    ``depth`` scales the bump: the reel-in speed follows from the winch's
    depower-shifted offset, so the depth directly sets how much tether the
    cycle reels back in (the radial-closure knob, see ``--close``).
    """
    from awetrim.kinematics.parametrized_patterns import reelin_bump

    bump = reelin_bump(
        s_grid,
        reelout_fraction=reelout_fraction,
        ramp_fraction=ramp_fraction,
        reelin_center=reelin_center,
    )
    return ROM_POWERED_INPUT_DEPOWER + depth * bump * (
        ROM_DEPOWERED_INPUT_DEPOWER - ROM_POWERED_INPUT_DEPOWER
    )


def curvature_from_angles(phi, beta, r0, s_span=1.0):
    """Return curvature metrics for the spherical path sampled over one period.

    Curvature is computed on the 3D unit direction q(phi, beta), then divided by
    ``r0`` to report physical curvature in 1/m. Samples are treated as periodic.
    """
    phi = np.asarray(phi, dtype=float).ravel()
    beta = np.asarray(beta, dtype=float).ravel()
    if phi.size != beta.size or phi.size < 5:
        raise ValueError("phi and beta must have the same length >= 5")

    q = np.column_stack(
        (
            np.cos(phi) * np.cos(beta),
            np.sin(phi) * np.cos(beta),
            np.sin(beta),
        )
    )
    ds = float(s_span) / float(phi.size)
    q_s = (np.roll(q, -1, axis=0) - np.roll(q, 1, axis=0)) / (2.0 * ds)
    q_ss = (np.roll(q, -1, axis=0) - 2.0 * q + np.roll(q, 1, axis=0)) / (ds * ds)
    speed = np.linalg.norm(q_s, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        kappa_unit = np.linalg.norm(np.cross(q_s, q_ss), axis=1) / speed**3
    kappa_unit[~np.isfinite(kappa_unit)] = np.nan
    kappa_physical = kappa_unit / float(r0)
    i_max = int(np.nanargmax(kappa_physical))
    return {
        "max_unit": float(kappa_unit[i_max]),
        "max_physical": float(kappa_physical[i_max]),
        "s_at_max": float(i_max / phi.size),
    }


def path_curvature_metrics(path_parameters, n_samples=CURVATURE_SAMPLES):
    """Sample a generated path config and return max curvature diagnostics."""
    from awetrim.kinematics.parametrized_patterns import PeriodicBSpline

    M = int(path_parameters["M"])
    r0 = float(path_parameters["r0"])
    pattern = PeriodicBSpline(
        M=M,
        C_phi=np.asarray(path_parameters["C_phi"], dtype=float).reshape((M, 1)),
        C_beta=np.asarray(path_parameters["C_beta"], dtype=float).reshape((M, 1)),
        s_init=float(path_parameters.get("s_init", 0.0)),
        s_final=float(path_parameters.get("s_final", 1.0)),
        downloops=bool(path_parameters.get("downloops", True)),
    )
    s = np.linspace(0.0, 1.0, int(n_samples), endpoint=False)
    phi = np.asarray([float(pattern.azimuth(r0, si)) for si in s])
    beta = np.asarray([float(pattern.elevation(r0, si)) for si in s])
    return curvature_from_angles(phi, beta, r0, s_span=1.0)


def enforce_curvature_limit(
    path_parameters, curvature_limit, n_samples=CURVATURE_SAMPLES
):
    """Raise when the sampled physical curvature exceeds ``curvature_limit``.

    A falsy limit (``None`` or 0) disables the guard.
    """
    metrics = path_curvature_metrics(path_parameters, n_samples=n_samples)
    print(
        "Path curvature: max {max_physical:.6g} 1/m "
        "({max_unit:.3g} on unit sphere) at s={s_at_max:.3f}".format(**metrics)
    )
    if curvature_limit and metrics["max_physical"] > float(curvature_limit):
        raise ValueError(
            "generated path exceeds curvature limit: "
            f"{metrics['max_physical']:.6g} > {float(curvature_limit):.6g} 1/m "
            f"at s={metrics['s_at_max']:.3f}"
        )
    return metrics


# Spline-density floors for the artificial shape, from a measured convergence
# study (geometric knees over n_loops 3-10 / ramp_fraction / reelout_fraction,
# confirmed by forward sims at 13 m/s): shape fidelity (<= 1 m path deviation
# at r0) needs ~8 control points per figure-eight and ~12-17 across the
# reel-in window, the transition ramps binding first; below that the underfit
# rings and the sampled curvature blows the write-time guard (n_loops=5:
# 0.082 1/m at M=42, 0.098 at M=30). 10/figure + 15/window hold the guard
# with margin while keeping the NLP small (C_phi/C_beta are 2M variables).
M_PER_FIGURE = 10
M_REELIN_WINDOW = 15


def _resolve_artificial_M(art):
    """Spline control points for the knob set: per-figure AND window floors.

    The periodic B-spline knots are uniform in s, so per-phase density can
    only be set through the total M: the figure oscillation runs n_loops
    times per period (M / n_loops points each, everywhere -- it is only
    amplitude-faded inside the reel-in window) and the reel-in window spans
    ``1 - reelout_fraction`` of the period (``M * (1 - f)`` points).
    ARTIFICIAL["M"] stays as a manual floor. A fixed M under-resolves higher
    loop counts and the underfit RINGS -- the sampled spline curvature blows
    far past the raw curve's (n_loops=8: 0.12 1/m raw -> 0.45 at M=40).
    """
    window = int(
        np.ceil(M_REELIN_WINDOW / max(1.0 - float(art["reelout_fraction"]), 1e-9))
    )
    return max(
        int(art["M"]),
        int(np.ceil(M_PER_FIGURE * float(art["n_loops"]))),
        window,
    )


def _artificial_path_parameters(art, r0=R0):
    """YAML-ready spline path parameters for the synthetic knob set ``art``."""
    return make_full_cycle_bspline_path_parameters(
        M=_resolve_artificial_M(art),
        r0=float(r0),
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
        lobe_handover_phase=art.get("lobe_handover_phase", LOBE_HANDOVER_PHASE),
        downloops=True,
    )


def _cycle_arrays(ekf_df, flight_df, cycle_id):
    """Whole-cycle (all four phases, time order) arrays needed for the fit."""
    mask = flight_df["cycle_by_phase"] == cycle_id
    flight = flight_df[mask].reset_index(drop=True)
    ekf = ekf_df[mask].reset_index(drop=True)
    if flight.empty:
        raise RuntimeError(f"Cycle {cycle_id}: no rows found")

    pos = np.column_stack(
        (
            ekf["kite_position_x"].to_numpy(),
            ekf["kite_position_y"].to_numpy(),
            ekf["kite_position_z"].to_numpy(),
        )
    )
    return {
        "azimuth": flight["kite_azimuth"].to_numpy(dtype=float),
        "elevation": flight["kite_elevation"].to_numpy(dtype=float),
        "distance_radial": np.linalg.norm(pos, axis=1),
        "tether_force": flight["ground_tether_force"].to_numpy(dtype=float),
        "reelout_speed": flight["tether_reelout_speed"].to_numpy(dtype=float),
        "time": flight["time"].to_numpy(dtype=float),
        "l_dp": flight_dataframe_depower_to_power_tape_length(flight),
        "wind": ekf.get(
            "wind_speed_horizontal", flight.get("wind_speed_horizontal")
        ).to_numpy(dtype=float),
    }


def _fit_winch_with_depower_offset(arr):
    """Fit ``T = slope * (v_r - offset(l_dp))`` with ``offset`` linear in l_dp.

    Returns slope, base offset (at ``l_dp_ref``), the depower-offset gain, the
    reference depower, and the force clamps. The gain is the data-driven shift of
    the winch's zero-force reeling speed with depower: it is what makes a single
    force law span both reel-out and reel-in (see ``Winch.tension_curve``).
    """
    v_r = arr["reelout_speed"]
    T = arr["tether_force"]
    l_dp = arr["l_dp"]
    finite = np.isfinite(v_r) & np.isfinite(T) & np.isfinite(l_dp) & (T > 500)

    # Slope from the reel-out (production) rows only -- positive reeling at high
    # force -- so the depowered reel-in does not flatten the regression.
    ro = finite & (v_r > 0.2) & (T > np.nanquantile(T[finite], 0.5))
    if ro.sum() < 2:
        ro = finite & (v_r > 0.0)
    slope = float(np.polyfit(v_r[ro], T[ro], 1)[0])
    slope = max(slope, 1.0)  # guard against a degenerate/negative fit

    # Implied per-row zero-force offset, then regress on depower:
    #   offset_row = v_r - T/slope ;  offset(l_dp) = a + gain * l_dp
    offset_row = v_r[finite] - T[finite] / slope
    gain, a = np.polyfit(l_dp[finite], offset_row, 1)
    gain = float(gain)
    l_dp_ref = float(ROM_POWERED_INPUT_DEPOWER)
    offset0 = float(a + gain * l_dp_ref)  # offset at the powered reference

    max_tf = float(np.nanquantile(T[finite], 0.97))
    min_tf = (
        float(np.nanquantile(T[finite & (v_r < 0)], 0.2))
        if (finite & (v_r < 0)).any()
        else float(np.nanquantile(T[finite], 0.05))
    )

    return {
        "slope_winch_ro": slope,
        "offset_winch_ro": offset0,
        "winch_offset_depower_gain": gain,
        "winch_depower_ref": l_dp_ref,
        "max_tether_force": max_tf,
        "min_tether_force": max(min_tf, 100.0),
    }


def build_config(
    arr=None,
    depower_depth=1.0,
    artificial=None,
    duration_s=None,
    curvature_limit=CURVATURE_LIMIT_1PM,
):
    """Assemble the reelout config dict.

    ``arr`` (the experimental cycle arrays) is only required when
    ``SHAPE_SOURCE == "experimental"``; the artificial path is fully synthetic
    and uses the data-derived constants ``WINCH_LAW`` / ``R0`` /
    ``CYCLE_DURATION_S``. ``depower_depth`` scales the synthetic reel-in
    depower bump (the radial-closure knob, see ``--close``). ``artificial``
    optionally overrides the module-level ARTIFICIAL dict (used by ``--auto``).
    ``duration_s`` overrides the cycle-duration estimate that sets
    ``n_points`` -- pass the MEASURED duration of a forward sim (see
    ``_measured_cycle_duration``) so the grid tracks the actual flight time
    instead of the CYCLE_DURATION_S prior. ``curvature_limit`` (1/m, falsy
    disables) is enforced ON THE FITTED SPLINE: control points around any
    violation are moved minimally (see
    ``fair_periodic_spline_to_curvature_limit``) -- the parametric knobs are
    never changed to satisfy it.
    """
    art = artificial if artificial is not None else ARTIFICIAL
    if SHAPE_SOURCE == "experimental":
        if arr is None:
            raise ValueError("experimental SHAPE_SOURCE needs the cycle arrays")
        duration = float(arr["time"][-1] - arr["time"][0])
        winch = _fit_winch_with_depower_offset(arr)
        r0 = float(arr["distance_radial"][0])
    else:
        # Physical loop time is set by the figure geometry and the kite's trim
        # speed, so cycle duration grows ~linearly with loop count (the
        # reel-out/reel-in split stays roughly constant -- it is fixed by the
        # reel speed ratio). This is only a PRIOR: it knows nothing about the
        # target wind or the tuned shape, so any flow that forward-simulates
        # replaces it with the measured duration via ``duration_s``.
        duration = float(CYCLE_DURATION_S) * float(art["n_loops"]) / 3.0
        winch = dict(WINCH_LAW)
        r0 = float(R0)
    if duration_s is not None:
        duration = float(duration_s)

    n_points = int(np.clip(np.ceil(duration * NPOINTS_PER_SECOND), 50, N_POINTS_MAX))
    print(
        "Winch law: T = {slope:.1f} * (v_r - offset), "
        "offset(l_dp) = {off:.3f} + {gain:.3f}*(l_dp - {ref:.2f}); "
        "clamps [{lo:.0f}, {hi:.0f}] N".format(
            slope=winch["slope_winch_ro"],
            off=winch["offset_winch_ro"],
            gain=winch["winch_offset_depower_gain"],
            ref=winch["winch_depower_ref"],
            lo=winch["min_tether_force"],
            hi=winch["max_tether_force"],
        )
    )
    s_grid = np.linspace(0.0, 1.0, n_points + 1, endpoint=True)

    if SHAPE_SOURCE == "artificial":
        # Clean, exactly-periodic synthetic cycle: figure-eights + reel-in arc.
        M = _resolve_artificial_M(art)
        print(
            f"Artificial shape, M={M}, n_loops={float(art['n_loops']):g}, "
            f"n_points={n_points}, depower depth={depower_depth:.3f}"
        )
        path_parameters = _artificial_path_parameters(art, r0=r0)
        if curvature_limit:
            # The parametric reel-in is only a SKETCH: when its fitted spline
            # violates the curvature limit, the reel-in is DESIGNED between
            # the frozen exit/entry handover states -- the window control
            # points solve a bending-energy NLP that hits beta_reelin_peak
            # with curvature bounded as a hard constraint. Knobs and the
            # figures are never changed.
            path_parameters, des = design_reelin_spline(
                path_parameters,
                float(curvature_limit),
                reelin_center=art.get("reelin_center", 0.5),
                reelout_fraction=art["reelout_fraction"],
                peak_elevation=art.get("beta_reelin_peak"),
            )
            if des["changed"]:
                print(
                    f"[design] reel-in designed between the handover states: "
                    f"curvature {des['max_before']:.4g} -> "
                    f"{des['max_after']:.4g} 1/m (limit "
                    f"{float(curvature_limit):g}), peak "
                    f"{np.degrees(des['peak_achieved']):.1f} deg (target "
                    f"{np.degrees(des['peak_target']):.1f}), "
                    f"{len(des['freed'])}/{M} window points designed, path "
                    f"moved <= {np.degrees(des['max_path_move']):.2f} deg"
                )
            # Safety net for anything the design could not (or was not
            # allowed to) touch -- figures, window edges, a failed design.
            path_parameters, fair = fair_periodic_spline_to_curvature_limit(
                path_parameters, float(curvature_limit)
            )
            if fair["changed"]:
                print(
                    f"[fair] residual spline curvature {fair['max_before']:.4g}"
                    f" -> {fair['max_after']:.4g} 1/m (sharpest at "
                    f"s={fair['s_at_max_before']:.3f}); moved "
                    f"{len(fair['touched'])}/{M} control points, path moved "
                    f"<= {np.degrees(fair['max_path_move']):.2f} deg"
                )
            if not fair["converged"]:
                print(
                    "[fair] WARN: curvature limit not reachable -- the "
                    "write-time guard will refuse this config; enlarge the "
                    "figures (az_amp0/beta_amp0) or change --loops"
                )
        # Synthetic depower aligned with the figures-stop / reel-in split
        # (same window AND ramp width as the shape).
        u_dep_profile = _synthetic_depower(
            s_grid,
            art["reelout_fraction"],
            art["ramp_fraction"],
            depth=depower_depth,
            reelin_center=art.get("reelin_center", 0.5),
        )
    elif SHAPE_SOURCE == "experimental":
        M = int(np.clip(np.ceil(duration * M_PER_SECOND), 8, M_MAX))
        print(f"Cycle {CYCLE_ID}: experimental fit, M={M}, n_points={n_points}")
        # Periodic fit over one full cycle. s in [0, 1); the loop closes back on
        # itself, so the periodic basis is the natural representation.
        s_samples = np.linspace(0.0, 1.0, len(arr["azimuth"]), endpoint=False)
        _, C_phi, C_beta = fit_bspline_pattern_to_trajectory(
            spline_type="periodic",
            M=M,
            s_init=0.0,
            s_final=1.0,
            az_target=arr["azimuth"],
            el_target=arr["elevation"],
            s_samples=s_samples,
            downloops=True,
        )
        path_parameters = {
            "r0": r0,
            "M": int(M),
            "C_phi": C_phi.full().flatten().round(6).tolist(),
            "C_beta": C_beta.full().flatten().round(6).tolist(),
            "s_init": 0.0,
            "s_final": 1.0,
            "downloops": True,
        }
        # Resample the measured depower onto the sim s-grid (n_points + 1) as the
        # warm-start for the optimized per-node profile.
        s_meas = np.linspace(0.0, 1.0, len(arr["l_dp"]), endpoint=False)
        u_dep_profile = np.interp(s_grid, s_meas, np.asarray(arr["l_dp"], dtype=float))
    else:
        raise ValueError(f"Unknown SHAPE_SOURCE {SHAPE_SOURCE!r}")

    config = {
        "reelout": {
            "pattern_type": "spline_periodic",
            "path_parameters": path_parameters,
            "radial_parameters": {
                "reeling_strategy": "force",
                "force_model": "linear",
                "reeling_speed": 0.0,
                "max_tether_force": winch["max_tether_force"],
                "min_tether_force": winch["min_tether_force"],
                "softplus": True,
                "softplus_beta": 1.0e-3,
                "softminus": True,
                "softminus_beta": 1.0e-3,
                "slope_winch_ro": winch["slope_winch_ro"],
                "offset_winch_ro": winch["offset_winch_ro"],
                # Depower-dependent offset (the key term for one-phase cycles).
                "winch_offset_depower_gain": winch["winch_offset_depower_gain"],
                "winch_depower_ref": winch["winch_depower_ref"],
            },
            "sim_parameters": {
                "start_angle": 0.0,
                "end_angle": 1.0,
                "n_points": int(n_points),
                "input_depower": float(u_dep_profile[0]),
                # Optimize l_dp as one per-node profile over the whole cycle.
                "optimize_depower_profile": True,
                "input_depower_profile": u_dep_profile.round(6).tolist(),
                "depower_rate": list(DEFAULT_OPTI_LIMITS["depower_rate"]),
                # Loosen per-node accept tolerance so the trim advances past
                # marginal nodes (matches the validation script).
                "solver_accept_residual_norm": 1.0e-3,
                # Solver keys consumed by run_full_cycle_opti.py, emitted here
                # so the generated YAML is complete and never hand-edited.
                "opti_limits_override": OPTI_LIMITS_OVERRIDE,
                # Whole-graph SX expansion: with the local-support node
                # functions (see Phase.run_opti) it builds in ~15 s at ~400
                # nodes (~3.7 MB/node) and iterates ~2x faster than the MX
                # outer graph; set False when memory-bound (~40% less).
                "expand_nlp": True,
                "ipopt_robust": False,
                # Abort -- instead of silently truncating and padding -- when a
                # node of the forward (warm-start) simulation fails to trim; a
                # padded seed breaks the NLP in hard-to-trace ways.
                "require_full_trajectory": True,
                # NLP winch handling (see WINCH_MODE above): free_speed makes
                # v_r a rate-limited control and drops the tension-curve
                # equality; the sim always marches with the force law.
                "winch_mode": WINCH_MODE,
                "winch_acceleration": WINCH_ACCELERATION,
            },
        }
    }
    return config


def _reelin_window_mask(s, art):
    """True inside the reel-in window (bump > 0) for the knob set ``art``."""
    f = float(art["reelout_fraction"])
    c = float(art.get("reelin_center", 0.5))
    h = 0.5 * (1.0 - f)
    d = (np.asarray(s, dtype=float) - c + 0.5) % 1.0 - 0.5
    return np.abs(d) <= h


def plot_seed_path(config, art=None, arr=None, save_path=None, show=False):
    """Plot the generated seed trajectory (pure geometry, no forward sim).

    Left: wind-window view (azimuth vs elevation) of the fitted periodic
    B-spline the optimizer warm-starts from, with the analytic synthetic
    target it was fitted to (artificial shape) or the flight samples
    (experimental fit), flight-direction arrows, the ``s = 0`` seam, the
    control polygon, and the reel-in window highlighted along the path with
    its entry / exit points (where the figure fade begins / ends -- with the
    "lobe" shape the visible peel-off and landing sit LOBE_HANDOVER_PHASE
    of figure phase inside those). Right: azimuth + elevation vs ``s`` and the
    synthetic depower profile, both with the reel-in window shaded. The title
    reports the resolution and the sampled max curvature. Saves a PNG to
    ``save_path`` (parents created) and/or shows the figure; returns it.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    from awetrim.kinematics.parametrized_patterns import PeriodicBSpline

    reelout = config["reelout"]
    path = reelout["path_parameters"]
    sim = reelout["sim_parameters"]
    M = int(path["M"])
    r0 = float(path["r0"])
    pattern = PeriodicBSpline(
        M=M,
        C_phi=np.asarray(path["C_phi"], dtype=float).reshape((M, 1)),
        C_beta=np.asarray(path["C_beta"], dtype=float).reshape((M, 1)),
        s_init=0.0,
        s_final=1.0,
        downloops=bool(path.get("downloops", True)),
    )
    s = np.linspace(0.0, 1.0, 800, endpoint=False)
    phi = np.array([float(pattern.azimuth(r0, si)) for si in s])
    beta = np.array([float(pattern.elevation(r0, si)) for si in s])
    kappa = path_curvature_metrics(path)["max_physical"]

    fig = plt.figure(figsize=(13.0, 6.2))
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=(1.25, 1.0),
        hspace=0.32,
        wspace=0.22,
        left=0.06,
        right=0.98,
        top=0.88,
        bottom=0.09,
    )
    ax_path = fig.add_subplot(grid[:, 0])
    ax_ang = fig.add_subplot(grid[0, 1])
    ax_dep = fig.add_subplot(grid[1, 1], sharex=ax_ang)

    in_window = None
    if art is not None:
        az_t, el_t = full_cycle_angles(
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
            lobe_handover_phase=art.get("lobe_handover_phase", LOBE_HANDOVER_PHASE),
            downloops=True,
        )
        ax_path.plot(
            np.degrees(az_t),
            np.degrees(el_t),
            color="0.6",
            lw=0.9,
            label="analytic target",
        )
        in_window = _reelin_window_mask(s, art)
    if arr is not None:
        ax_path.plot(
            np.degrees(arr["azimuth"]),
            np.degrees(arr["elevation"]),
            ".",
            ms=2,
            alpha=0.35,
            color="0.5",
            label="flight",
        )

    # Control polygon (light) and the fitted spline, reel-in part on top.
    ax_path.plot(
        np.degrees(path["C_phi"]),
        np.degrees(path["C_beta"]),
        "o",
        ms=2.5,
        color="0.4",
        alpha=0.6,
        label=f"control points (M={M})",
    )
    ax_path.plot(
        np.degrees(phi),
        np.degrees(beta),
        color="tab:blue",
        lw=1.8,
        label="periodic B-spline",
    )
    if in_window is not None and in_window.any():
        # Draw the reel-in part as contiguous segments (it may wrap the seam).
        idx = np.flatnonzero(in_window)
        breaks = np.flatnonzero(np.diff(idx) > 1)
        chunks = np.split(idx, breaks + 1)
        if len(chunks) > 1 and idx[0] == 0 and idx[-1] == s.size - 1:
            chunks = [np.r_[chunks[-1], chunks[0]]] + chunks[1:-1]
        for k, ch in enumerate(chunks):
            ax_path.plot(
                np.degrees(phi[ch]),
                np.degrees(beta[ch]),
                color="tab:orange",
                lw=2.6,
                label="reel-in window (spline)" if k == 0 else None,
            )
        # Peel-off (window entry) and landing (window exit) points.
        i_in, i_out = chunks[0][0], chunks[0][-1]
        ax_path.plot(
            np.degrees(phi[i_in]),
            np.degrees(beta[i_in]),
            "^",
            ms=8,
            color="tab:green",
            mec="k",
            label="window entry (fade begins)",
            zorder=6,
        )
        ax_path.plot(
            np.degrees(phi[i_out]),
            np.degrees(beta[i_out]),
            "v",
            ms=8,
            color="tab:red",
            mec="k",
            label="window exit (figures back)",
            zorder=6,
        )

    # Flight direction: one arrow every ~1/32 of the period.
    step = max(1, s.size // 32)
    ax_path.quiver(
        np.degrees(phi[::step]),
        np.degrees(beta[::step]),
        np.gradient(np.degrees(phi))[::step],
        np.gradient(np.degrees(beta))[::step],
        color="k",
        angles="xy",
        scale_units="xy",
        scale=0.08,
        width=0.004,
        headwidth=5,
        alpha=0.8,
        zorder=5,
    )
    ax_path.plot(
        np.degrees(phi[0]),
        np.degrees(beta[0]),
        "o",
        ms=8,
        color="w",
        mec="k",
        mew=1.5,
        label="s = 0",
        zorder=7,
    )
    ax_path.axvline(0.0, color="0.75", lw=0.8, ls=":")
    ax_path.set_xlabel(r"azimuth $\phi$ (deg)")
    ax_path.set_ylabel(r"elevation $\beta$ (deg)")
    ax_path.grid(True, alpha=0.4)
    ax_path.legend(loc="upper right", fontsize=8)
    ax_path.set_title("Wind-window view (seed path)", fontsize=10)

    # Right column: angles and depower vs s, reel-in window shaded.
    ax_ang.plot(s, np.degrees(phi), color="tab:blue", lw=1.3, label="azimuth")
    ax_ang.plot(s, np.degrees(beta), color="tab:orange", lw=1.3, label="elevation")
    ax_ang.set_ylabel("angle (deg)")
    ax_ang.grid(True, alpha=0.4)
    s_dep = np.linspace(0.0, 1.0, len(sim["input_depower_profile"]))
    ax_dep.plot(
        s_dep,
        sim["input_depower_profile"],
        color="tab:green",
        lw=1.5,
        label="depower l_dp (seed)",
    )
    ax_dep.set_xlabel("s (one period = one cycle)")
    ax_dep.set_ylabel("input_depower (m)")
    ax_dep.grid(True, alpha=0.4)
    if in_window is not None:
        for ax in (ax_ang, ax_dep):
            ax.fill_between(
                s,
                0,
                1,
                where=in_window,
                transform=ax.get_xaxis_transform(),
                color="tab:orange",
                alpha=0.12,
                lw=0,
            )
    handles, labels = ax_ang.get_legend_handles_labels()
    if in_window is not None:
        handles.append(Patch(color="tab:orange", alpha=0.25))
        labels.append("reel-in window")
    ax_ang.legend(handles, labels, loc="upper center", fontsize=8, ncol=3)
    ax_dep.legend(loc="upper center", fontsize=8)

    bits = [f"M={M}", f"n_points={int(sim['n_points'])}", f"r0={r0:.0f} m"]
    if art is not None:
        bits = [
            f"n_loops={float(art['n_loops']):g}",
            f"f_ro={float(art['reelout_fraction']):.2f}",
            f"bow={art.get('bow_shape', 'sym')}",
            f"peak={np.degrees(float(art['beta_reelin_peak'])):.0f} deg",
            f"A={float(art['az_reelin_amp']):+.2f} rad",
            f"ramp={float(art['ramp_fraction']):.2f}",
        ] + bits
    fig.suptitle(
        "Full-cycle seed -- " + ", ".join(bits) + f"  |  max curvature {kappa:.3f} 1/m",
        fontsize=10,
    )
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=130)
        print(f"Saved seed-path figure to {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def _simulate_cycle(config, run_plots=False):
    """Forward-simulate the generated cycle config; returns (phase, system_model).

    Uses the same constant-log wind the optimizer defaults to, so the simulated
    trajectory is the actual initial guess the NLP will warm-start from.
    """
    from awetrim.system.factory import create_system_model_from_yaml
    from awetrim.timeseries.phase import Phase
    from awetrim.utils.config_paths import LEI_V3_SYSTEM_CONFIG

    # Evaluate the seed at the SAME wind the optimizer will use: a shape that
    # trims/closes at one wind speed can be infeasible at another (higher wind
    # rides the max-force clamp, lower wind starves the reel-in top), so the
    # feasibility report and --close are only meaningful at the target wind.
    from run_full_cycle_opti import WIND_CONFIG, build_wind_model

    # Work on a copy with strict trimming OFF: this diagnostic run should show
    # how far the trim gets even when a node fails (the optimizer runs strict).
    reelout = copy.deepcopy(config["reelout"])
    reelout["sim_parameters"]["require_full_trajectory"] = False
    sim = reelout["sim_parameters"]

    system_model = create_system_model_from_yaml(yaml_path=LEI_V3_SYSTEM_CONFIG)
    system_model.wind = build_wind_model(**WIND_CONFIG)
    print(
        "Simulating seed at WIND_CONFIG from run_full_cycle_opti: "
        f"{WIND_CONFIG['speed_wind_at_100']:g} m/s @ 100 m, z0={WIND_CONFIG['z0']:g}"
    )

    start_state = {
        "t": 0,
        "s": 0,
        "s_dot": 2,
        "input_steering": 0,
        "tension_tether_ground": reelout["radial_parameters"]["max_tether_force"],
        "speed_radial": 1.0,
        "distance_radial": reelout["path_parameters"]["r0"],
        "input_depower": float(sim["input_depower_profile"][0]),
    }
    phase_obj = Phase(
        system_model=system_model, pattern_config=reelout, start_state=start_state
    )
    phase, _ = phase_obj.run_simulation(run_plots=run_plots, phase_sim=True)
    try:
        print("Initial-guess metrics:", phase.energy_metrics())
    except Exception as exc:
        print(f"Initial-guess metrics unavailable ({exc})")
    return phase, system_model


def _measured_cycle_duration(phase, n_points):
    """Simulated duration (s) of the full cycle, or None when the trim truncated.

    This is what recalibrates ``n_points``: the CYCLE_DURATION_S prior can be
    far off at other winds/shapes (2.2x at 13 m/s with 5 loops), and the grid
    should track the actual flight time, not the prior.
    """
    r = np.asarray(phase.return_variable("distance_radial"), dtype=float)
    if r.size < int(n_points):
        return None
    t = np.asarray(phase.return_variable("t"), dtype=float)
    return float(t[-1] - t[0])


def _dist_to_reelin_center(s_v, center):
    """Circular |s_v - center| on the unit period (the window wraps the seam)."""
    return abs((float(s_v) - float(center) + 0.5) % 1.0 - 0.5)


def _violation_ranges(s, mask):
    """Contiguous s-ranges where ``mask`` is True, as (s_lo, s_hi) tuples."""
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    ranges = []
    start = prev = idx[0]
    for j in idx[1:]:
        if j == prev + 1:
            prev = j
            continue
        ranges.append((float(s[start]), float(s[prev])))
        start = prev = j
    ranges.append((float(s[start]), float(s[prev])))
    return ranges


def _feasibility_metrics(phase, system_model, config, artificial=None):
    """Numeric feasibility summary of a simulated cycle.

    Shared by the printed report and the ``--auto`` tuner: trim truncation,
    radial closure gap, and steering / AoA excursions vs the bounds the NLP
    will impose, with the worst-violation location in ``s`` for the
    knob<->symptom rules.
    """
    art = artificial if artificial is not None else ARTIFICIAL
    sim = config["reelout"]["sim_parameters"]
    hw = getattr(system_model, "hardware_limits", None) or {}

    n_points = int(sim["n_points"])
    s = np.asarray(phase.return_variable("s"), dtype=float)
    r = np.asarray(phase.return_variable("distance_radial"), dtype=float)
    u_s = np.asarray(phase.return_variable("input_steering"), dtype=float)

    f = float(art["reelout_fraction"])
    ri_c = float(art.get("reelin_center", 0.5))
    truncated = r.size < n_points
    m = {
        "n_points": n_points,
        "ri_lo": ri_c - 0.5 * (1.0 - f),
        "ri_hi": ri_c + 0.5 * (1.0 - f),
        "truncated_at": int(r.size) if truncated else None,
        "trunc_s": float(s[-1]) if truncated else None,
        "gap": None if truncated else float(r[-1] - r[0]),
        "r0": float(r[0]),
    }

    lb, ub = hw.get("input_steering") or DEFAULT_OPTI_LIMITS["input_steering"]
    over = np.maximum(u_s - ub, lb - u_s)  # >0 where out of bounds
    viol = over > 0.0
    m.update(
        steer_lb=float(lb),
        steer_ub=float(ub),
        steer_min=float(u_s.min()),
        steer_max=float(u_s.max()),
        steer_n_viol=int(viol.sum()),
        steer_ranges=_violation_ranges(s, viol),
        steer_excess=float(max(over.max(), 0.0)),
        steer_worst_s=float(s[int(np.argmax(over))]) if viol.any() else None,
    )

    try:
        aoa = np.asarray(phase.return_variable("angle_of_attack"), dtype=float)
        alb, aub = hw.get("angle_of_attack") or DEFAULT_OPTI_LIMITS["angle_of_attack"]
        aover = np.maximum(aoa - aub, alb - aoa)
        aviol = aover > 0.0
        m.update(
            aoa_available=True,
            aoa_lb=float(alb),
            aoa_ub=float(aub),
            aoa_min=float(aoa.min()),
            aoa_max=float(aoa.max()),
            aoa_n_viol=int(aviol.sum()),
            aoa_ranges=_violation_ranges(s[: aoa.size], aviol),
            aoa_worst_s=(
                float(s[: aoa.size][int(np.argmax(aover))]) if aviol.any() else None
            ),
        )
    except Exception:
        m["aoa_available"] = False
    return m


def _feasibility_report(phase, system_model, config, artificial=None):
    """Print the simulated cycle's feasibility vs the NLP bounds.

    A "feasible initial trajectory" means: every node trimmed, the controls
    the NLP bounds (steering, AoA) stay inside their limits along the whole
    cycle, and the cycle roughly closes radially. Violations are localized in
    ``s`` and mapped to the generator knob that relieves them (which is what
    ``--auto`` iterates automatically). Returns True for a clean warm start.
    """
    m = _feasibility_metrics(phase, system_model, config, artificial=artificial)

    ri_c = 0.5 * (m["ri_lo"] + m["ri_hi"])
    ri_h = 0.5 * (m["ri_hi"] - m["ri_lo"])

    def _describe(ranges):
        for s_lo, s_hi in ranges:
            in_reelin = any(
                _dist_to_reelin_center(v, ri_c) <= ri_h
                for v in (s_lo, 0.5 * (s_lo + s_hi), s_hi)
            )
            where = "reel-in window" if in_reelin else "figure eights"
            print(f"    s in [{s_lo:.2f}, {s_hi:.2f}]  ({where})")

    print("\n--- Initial-guess feasibility report ---")
    ok = True

    if m["truncated_at"] is not None:
        ok = False
        print(
            f"[FAIL] trim truncated at node {m['truncated_at']}/{m['n_points']} "
            f"(s ~ {m['trunc_s']:.2f}): the shape is infeasible from there on"
        )
    else:
        print(f"[ok]   trim converged at all {m['n_points']} nodes")

    if m["gap"] is not None:
        closed = abs(m["gap"]) <= CLOSURE_TOL_M
        if not closed:
            ok = False
        print(
            f"[{'ok  ' if closed else 'WARN'}] radial closure: r_end - r0 = "
            f"{m['gap']:+.1f} m (r0 = {m['r0']:.1f} m)"
            + ("" if closed else " -> --close/--auto fit the depower depth")
        )

    if m["steer_n_viol"] and m["steer_excess"] <= AUTO_STEER_SLACK:
        # Within the warm-start slack: the flown KCU saturates at exactly this
        # bound (commanding well past it), so exact tracking of any production
        # figure needs a little more than the actuator gives -- the NLP clips
        # these nodes. Do not fail the verdict on them.
        print(
            f"[WARN] steering grazes [{m['steer_lb']}, {m['steer_ub']}] at "
            f"{m['steer_n_viol']} node(s) "
            f"(min {m['steer_min']:.2f}, max {m['steer_max']:.2f}) -- within "
            f"the +-{AUTO_STEER_SLACK:.2f} warm-start slack, NLP clips these"
        )
    elif m["steer_n_viol"]:
        ok = False
        print(
            f"[FAIL] steering outside [{m['steer_lb']}, {m['steer_ub']}] at "
            f"{m['steer_n_viol']} node(s) "
            f"(min {m['steer_min']:.2f}, max {m['steer_max']:.2f}):"
        )
        _describe(m["steer_ranges"])
        print(
            "    -> reel-in window: widen az_reelin_amp / lower beta_reelin_peak\n"
            "    -> figure eights:  bigger figures (az_amp0/beta_amp0 up) or fewer loops"
        )
    else:
        print(
            f"[ok]   steering within [{m['steer_lb']}, {m['steer_ub']}] "
            f"(min {m['steer_min']:.2f}, max {m['steer_max']:.2f})"
        )

    if not m.get("aoa_available"):
        print("[--]   angle of attack not recorded by the sim; skipped")
    elif m["aoa_n_viol"]:
        ok = False
        print(
            f"[FAIL] angle of attack outside [{m['aoa_lb']}, {m['aoa_ub']}] rad "
            f"at {m['aoa_n_viol']} node(s) "
            f"(min {m['aoa_min']:.3f}, max {m['aoa_max']:.3f}):"
        )
        _describe(m["aoa_ranges"])
    else:
        print(
            f"[ok]   angle of attack within [{m['aoa_lb']}, {m['aoa_ub']}] rad "
            f"(min {m['aoa_min']:.3f}, max {m['aoa_max']:.3f})"
        )

    print(
        "verdict: "
        + (
            "FEASIBLE initial trajectory (clean NLP warm start)"
            if ok
            else "NOT feasible as a warm start -- run --auto or tune the knobs"
        )
    )
    return ok


# --auto tuning rules. One knob moves per iteration; the mapping encodes what
# the manual sweeps established (see the ARTIFICIAL comments). Steering gets a
# small slack over the hardware bound: the flown KCU saturates at the bound and
# exact tracking of ANY production figure needs ~4*pi/T_loop of turn rate, so
# a few-percent overshoot at a handful of nodes is the physical floor -- the
# NLP clips it; demanding strict compliance would make the tuner chase forever.
AUTO_STEER_SLACK = 0.05
AUTO_MAX_ITER = 12


def _symptom_signature(m):
    """(kind, location) of the dominant symptom, for stall detection."""
    if m["truncated_at"] is not None:
        return ("trunc", round(m["trunc_s"], 1))
    if m.get("aoa_available") and m.get("aoa_n_viol"):
        return ("aoa", round(m["aoa_worst_s"] or 0.0, 1))
    if m["steer_excess"] > AUTO_STEER_SLACK:
        return ("steer", round(m["steer_worst_s"], 1))
    if m["gap"] is not None and abs(m["gap"]) > CLOSURE_TOL_M:
        return ("gap", 0.0)
    return None


def _feasibility_score(m):
    """Scalar badness of a simulated seed (0 = feasible + closed); lower wins."""
    score = 0.0
    if m["truncated_at"] is not None:
        score += 100.0 + 100.0 * (1.0 - m["truncated_at"] / m["n_points"])
    if m.get("aoa_available") and m.get("aoa_n_viol"):
        score += 50.0 * max(m["aoa_max"] - m["aoa_ub"], m["aoa_lb"] - m["aoa_min"])
    score += 10.0 * max(m["steer_excess"] - AUTO_STEER_SLACK, 0.0)
    if m["gap"] is not None:
        score += max(abs(m["gap"]) - CLOSURE_TOL_M, 0.0) / 10.0
    return score


def _auto_action(m, art, depth, escalate=0):
    """One knob adjustment from the feasibility metrics.

    Returns ``(description, new_art, new_depth)`` with exactly one knob
    changed, ``None`` when the seed is feasible and closed, or raises
    ``RuntimeError`` when no knob can fix the remaining symptom. ``escalate``
    counts how often the SAME symptom survived a knob move: it advances
    through the region's lever chain instead of hammering a lever that has
    proven ineffective (e.g. the reel-in EXIT handover does not care about
    the peak elevation at all -- see the n_loops=4 tuning trace).
    """
    art = dict(art)
    h = 0.5 * (m["ri_hi"] - m["ri_lo"])
    ri_c = 0.5 * (m["ri_hi"] + m["ri_lo"])

    def region(s_v):
        # Circular distance: the reel-in window may wrap the periodic seam.
        d = _dist_to_reelin_center(s_v, ri_c)
        if d <= h:
            return "top" if d <= 0.5 * h else "edge"
        return "figures"

    # Individual levers: (description, art, depth), or None when at their cap.
    def bow_up(why):
        # Wider bow = wider arc through the low-speed top. Widen the MAGNITUDE
        # up to the cap (the seed's bow direction/sign is preserved, so a
        # NEGATIVE seed bow widens -- more negative -- instead of shrinking
        # toward zero, and the exit side survives).
        _, hi = AZ_REELIN_AMP_MAG_BOUNDS
        sign = -1.0 if np.signbit(float(art["az_reelin_amp"])) else 1.0
        mag = abs(art["az_reelin_amp"])
        if mag >= hi - 0.01:
            return None
        old = art["az_reelin_amp"]
        art["az_reelin_amp"] = round(sign * min(mag + 0.06, hi), 3)
        return (
            f"{why} -> az_reelin_amp {old:.2f} -> {art['az_reelin_amp']:.2f}",
            art,
            depth,
        )

    def peak_down(why):
        # Lower peak = more apparent wind at the top.
        if art["beta_reelin_peak"] <= 0.62:
            return None
        old = art["beta_reelin_peak"]
        art["beta_reelin_peak"] = round(old - 0.1, 3)
        return (
            f"{why} -> beta_reelin_peak {old:.2f} -> {art['beta_reelin_peak']:.2f}",
            art,
            depth,
        )

    def ramp_up(why):
        # Gentler window edges = smoother reel-out/reel-in handover.
        if art["ramp_fraction"] >= 0.47:
            return None
        old = art["ramp_fraction"]
        art["ramp_fraction"] = round(min(old + 0.03, 0.48), 3)
        return (
            f"{why} -> ramp_fraction {old:.2f} -> {art['ramp_fraction']:.2f}",
            art,
            depth,
        )

    def figures_up(why):
        # Bigger figures = longer arc per loop = lower turn rate at the kite's
        # natural speed. Scale both amps at the ~2.5 ratio.
        if art["az_amp0"] >= 0.44:
            return None
        old_a, old_b = art["az_amp0"], art["beta_amp0"]
        art["az_amp0"] = round(min(old_a * 1.12, 0.45), 3)
        art["beta_amp0"] = round(art["az_amp0"] / 2.5, 3)
        return (
            f"{why} -> figures a{old_a:.2f}/b{old_b:.2f} -> "
            f"a{art['az_amp0']:.2f}/b{art['beta_amp0']:.2f}",
            art,
            depth,
        )

    # Lever chains per symptom region. "edge" is the reel-out/reel-in
    # handover, where the figures re-emerge while still elevated/depowered --
    # its levers are the handover smoothness and the figure turn rate, NOT
    # the top geometry.
    chains = {
        "top": (bow_up, peak_down),
        "edge": (ramp_up, figures_up, peak_down),
        "figures": (figures_up,),
    }

    def run_chain(reg, why):
        for lever in chains[reg][escalate:]:
            action = lever(why)
            if action is not None:
                return action
        raise RuntimeError(
            f"{why}: no remaining knob helps in the '{reg}' region"
            + ("" if reg == "top" else " -- try fewer loops (--loops)")
        )

    # 1) A node that fails to trim is the hardest symptom -- fix it first.
    if m["truncated_at"] is not None:
        return run_chain(
            region(m["trunc_s"]), f"trim truncated at s={m['trunc_s']:.2f}"
        )

    # 2) AoA violations (in practice always the slow reel-in top).
    if m.get("aoa_available") and m.get("aoa_n_viol"):
        why = f"AoA {m['aoa_max']:.2f} rad > {m['aoa_ub']:.2f} at s={m['aoa_worst_s']:.2f}"
        action = peak_down(why)
        if action is None:
            raise RuntimeError(f"{why}, but beta_reelin_peak is at its floor")
        return action

    # 3) Steering beyond the slack.
    if m["steer_excess"] > AUTO_STEER_SLACK:
        s_w = m["steer_worst_s"]
        why = (
            f"steering [{m['steer_min']:+.2f}, {m['steer_max']:+.2f}] at "
            f"{m['steer_n_viol']} node(s), worst at s={s_w:.2f}"
        )
        return run_chain(region(s_w), why)

    # 4) Radial closure, via the depower depth; window split as fallback.
    # (Exempt from escalation: repeated proportional depth steps ARE progress.)
    if m["gap"] is not None and abs(m["gap"]) > CLOSURE_TOL_M:
        gap = m["gap"]
        # dgap/ddepth ~ -40 m per unit depth (measured at 8-10 m/s).
        new_depth = float(np.clip(depth + gap / 40.0, 0.25, 1.0))
        if abs(new_depth - depth) > 1e-3:
            return (
                f"closure gap {gap:+.1f} m -> depower depth "
                f"{depth:.2f} -> {new_depth:.2f}",
                art,
                new_depth,
            )
        if gap > 0 and art["reelout_fraction"] > 0.53:
            old = art["reelout_fraction"]
            art["reelout_fraction"] = round(old - min(max(gap / 75.0, 0.01), 0.04), 3)
            return (
                f"under-reel {gap:+.1f} m at full depower -> reelout_fraction "
                f"{old:.2f} -> {art['reelout_fraction']:.2f}",
                art,
                depth,
            )
        raise RuntimeError(
            f"cannot close the cycle (gap {gap:+.1f} m) within knob ranges"
        )

    return None


def _auto_tune(arr=None, max_iter=AUTO_MAX_ITER, curvature_limit=None):
    """Automatically tune the synthetic shape until the seed is feasible.

    The simulate -> feasibility-metrics -> one-knob-adjustment loop that was
    previously done by hand, so ANY wind speed (from run_full_cycle_opti's
    WIND_CONFIG) and ANY --loops value gets a trim-feasible, radially closed
    seed without editing ARTIFICIAL. Each iteration is one forward sim
    (~1 min). A symptom that survives a knob move escalates to the next lever
    in its chain (stall detection), and the BEST simulated iterate -- not the
    last -- is returned and written; the knobs are printed for pinning.

    ``curvature_limit`` (1/m) is enforced on the spline inside every
    ``build_config`` (local fairing) and additionally biases the best-iterate
    selection: an iterate whose faired spline still violates the limit is
    penalized -- otherwise --auto could hand the write-time guard a config it
    must refuse.
    """
    art = dict(ARTIFICIAL)
    depth = float(np.clip(DEPOWER_DEPTH, 0.25, 1.0))
    best = None  # (score, config, phase, system_model, art, depth)
    prev_sig = None
    prev_score = float("inf")
    escalate = 0
    duration_s = None  # measured sim duration; recalibrates n_points as we go
    for it in range(1, max_iter + 1):
        config = build_config(
            arr,
            depower_depth=depth,
            artificial=art,
            duration_s=duration_s,
            curvature_limit=curvature_limit,
        )
        phase, system_model = _simulate_cycle(config)
        measured = _measured_cycle_duration(
            phase, config["reelout"]["sim_parameters"]["n_points"]
        )
        if measured is not None:
            if duration_s is None or abs(measured - duration_s) > 0.05 * measured:
                print(
                    f"[auto] measured cycle duration {measured:.1f} s -> "
                    "n_points recalibrated from the next iterate on"
                )
            duration_s = measured
        metrics = _feasibility_metrics(phase, system_model, config, artificial=art)

        score = _feasibility_score(metrics)
        if curvature_limit:
            kappa = path_curvature_metrics(config["reelout"]["path_parameters"])[
                "max_physical"
            ]
            score += 25.0 * max(kappa / float(curvature_limit) - 1.0, 0.0)
        if best is None or score < best[0]:
            best = (score, config, phase, system_model, dict(art), depth)

        # Escalate to the next lever only when the SAME symptom is not even
        # improving: a lever that shrinks the violation (e.g. peak_down on the
        # top steering) should keep stepping, not be abandoned after one try.
        sig = _symptom_signature(metrics)
        stalled = (
            sig is not None
            and sig[0] != "gap"
            and sig == prev_sig
            and score >= 0.9 * prev_score
        )
        escalate = escalate + 1 if stalled else 0
        prev_sig, prev_score = sig, score

        try:
            action = _auto_action(metrics, art, depth, escalate=escalate)
        except RuntimeError as exc:
            print(f"[auto] stuck: {exc}; keeping the best iterate")
            break
        if action is None:
            print(f"[auto] converged after {it} iteration(s)")
            break
        if it == max_iter:
            print(
                f"[auto] iteration budget ({max_iter}) exhausted; keeping best iterate"
            )
            break
        desc, art, depth = action
        print(f"[auto] it {it}: {desc}")

    _, config, phase, system_model, best_art, best_depth = best
    print("[auto] knobs of the written config (pin to reproduce without --auto):")
    for key in (
        "n_loops",
        "reelout_fraction",
        "beta0",
        "beta_amp0",
        "az_amp0",
        "beta_reelin_peak",
        "az_reelin_amp",
        "az_reelin_through",
        "ramp_fraction",
        "reelin_center",
        "psi0",
        "psi_entry",
        "psi_exit",
        "bow_shape",
        "lobe_handover_phase",
    ):
        print(f"    {key}: {best_art.get(key)}")
    print(f"    DEPOWER_DEPTH: {best_depth:.3f}")
    if duration_s is not None:
        print(
            f"    (measured cycle duration {duration_s:.1f} s; set "
            f"CYCLE_DURATION_S = {duration_s * 3.0 / best_art['n_loops']:.0f} "
            "to reproduce n_points without a sim)"
        )
    return config, phase, system_model, best_art


def _close_cycle_depth(arr=None, max_iter=6, curvature_limit=CURVATURE_LIMIT_1PM):
    """Fit the reel-in depower depth so the simulated cycle closes radially.

    Secant iteration on ``depth``: the winch's depower-shifted offset makes the
    reel-in speed (and hence the tether reeled back in) monotone in the bump
    depth, so 2-4 forward sims usually land |r_end - r0| under CLOSURE_TOL_M.
    Returns (config, phase, system_model) at the fitted depth.
    """
    lo_d, hi_d = 0.2, 1.0

    def _closure_gap(phase, n_points):
        r = np.asarray(phase.return_variable("distance_radial"), dtype=float)
        if r.size < n_points:
            return None  # truncated: gap is meaningless
        return float(r[-1] - r[0])

    depth = float(np.clip(DEPOWER_DEPTH, lo_d, hi_d))
    config = build_config(arr, depower_depth=depth, curvature_limit=curvature_limit)
    n_points = int(config["reelout"]["sim_parameters"]["n_points"])
    phase, system_model = _simulate_cycle(config)
    gap = _closure_gap(phase, n_points)
    print(
        f"[close] depth={depth:.3f} -> gap={'trim failed' if gap is None else f'{gap:+.1f} m'}"
    )
    # Calibrate n_points to the measured duration ONCE (held fixed through the
    # secant so the gap is always evaluated on the same grid).
    duration_s = _measured_cycle_duration(phase, n_points)
    if duration_s is not None:
        print(f"[close] measured cycle duration {duration_s:.1f} s")

    if gap is None:
        print("[close] trim truncated; fix shape feasibility first (see report)")
        return config, phase, system_model
    if gap > CLOSURE_TOL_M and depth >= hi_d:
        print(
            "[close] cycle under-reels even fully depowered; lower "
            "reelout_fraction (longer reel-in window) instead of the depth"
        )
        return config, phase, system_model

    depth_prev, gap_prev = depth, gap
    fitted_depth = depth  # last depth actually simulated (== the written config)
    # First step: nudge in the closing direction (gap decreases with depth).
    depth = float(np.clip(depth + (0.15 if gap > 0 else -0.15), lo_d, hi_d))
    for _ in range(max_iter):
        if abs(gap) <= CLOSURE_TOL_M:
            break
        config = build_config(
            arr,
            depower_depth=depth,
            duration_s=duration_s,
            curvature_limit=curvature_limit,
        )
        n_points = int(config["reelout"]["sim_parameters"]["n_points"])
        phase, system_model = _simulate_cycle(config)
        gap = _closure_gap(phase, n_points)
        fitted_depth = depth
        print(
            f"[close] depth={depth:.3f} -> gap="
            f"{'trim failed' if gap is None else f'{gap:+.1f} m'}"
        )
        if gap is None:
            break
        denom = gap - gap_prev
        if abs(gap) <= CLOSURE_TOL_M or abs(denom) < 1e-9:
            break
        step = gap * (depth - depth_prev) / denom
        depth_prev, gap_prev = depth, gap
        depth = float(np.clip(depth - step, lo_d, hi_d))

    print(
        f"[close] done at depth={fitted_depth:.3f}; pin this value in "
        "DEPOWER_DEPTH to reproduce without re-fitting"
    )
    return config, phase, system_model


def _load_experimental_cycle():
    """EKF flight load + cycle isolation (SHAPE_SOURCE == 'experimental' only)."""
    ekf_df, flight_df, _ = read_results(
        FLIGHT["year"],
        FLIGHT["month"],
        FLIGHT["day"],
        FLIGHT["kite_model"],
        addition="",
        path_to_main=PATH_TO_MAIN,
    )
    if "flight_phase_index" not in flight_df:
        raise RuntimeError("flight_phase_index column is required to derive cycles")
    flight_df["cycle_by_phase"] = cycles_from_phases(flight_df["flight_phase_index"])
    available = [int(c) for c in np.unique(flight_df["cycle_by_phase"]) if c >= 0]
    if CYCLE_ID not in available:
        raise RuntimeError(f"Cycle {CYCLE_ID} not found. Available cycles: {available}")
    return _cycle_arrays(ekf_df, flight_df, CYCLE_ID)


def main(
    run_plots: bool = False,
    check: bool = False,
    close: bool = False,
    auto: bool = False,
    loops: int = None,
    reentry: str = None,
    reelin_shape: str = None,
    cross_at: float = None,
    curvature_limit: float = CURVATURE_LIMIT_1PM,
    reelin_center: float = None,
) -> int:
    if cross_at is not None:
        ARTIFICIAL["reelin_cross_pos"] = float(cross_at)
    if loops:
        # CLI override in the WYSIWYG unit: half figure-eights (lobes) VISIBLE
        # during reel-out. Derive the internal continuous full-period count
        # (and the parity-matching psi_entry -- the count is always honoured
        # EXACTLY; the peel-off crossing mirrors by pi as its parity needs);
        # --auto adapts the remaining knobs to the loop count.
        ARTIFICIAL["n_loops"], ARTIFICIAL["psi_entry"] = (
            full_cycle_n_loops_for_half_figures(
                int(loops),
                reelout_fraction=ARTIFICIAL["reelout_fraction"],
                psi_entry=ARTIFICIAL.get("psi_entry"),
                psi_exit=ARTIFICIAL.get("psi_exit"),
                bow_shape=ARTIFICIAL.get("bow_shape", "sym"),
                lobe_handover_phase=ARTIFICIAL.get(
                    "lobe_handover_phase", LOBE_HANDOVER_PHASE
                ),
            )
        )
    if ARTIFICIAL.get("bow_shape") == "lobe":
        # The reel-in ALWAYS exits tangentially at a side of the figure going
        # down, so the exit-vs-entry sameness is fixed by the half-lobe
        # parity; --reentry only VALIDATES the request against it. The free
        # geometric choice per peel-off is which side the top goes THROUGH.
        halves = round(
            full_cycle_visible_half_figures(
                ARTIFICIAL["n_loops"],
                reelout_fraction=ARTIFICIAL["reelout_fraction"],
                psi_entry=ARTIFICIAL.get("psi_entry"),
                psi_exit=ARTIFICIAL.get("psi_exit"),
                bow_shape="lobe",
                lobe_handover_phase=ARTIFICIAL.get(
                    "lobe_handover_phase", LOBE_HANDOVER_PHASE
                ),
            )
        )
        natural = "opposite" if halves % 2 else "same"
        if reentry and reentry != natural:
            other = loops + 1 if loops else "an odd" if reentry == "opposite" else "an even"
            print(
                f"[reentry] a tangential {reentry}-side exit is impossible "
                f"with {halves} visible half-lobes (the giant reel-in lobe "
                f"is one lobe of the figure, so {halves} halves re-enter "
                f"{natural}-side); use "
                + (f"--loops {other} " if loops else f"{other} --loops ")
                + f"or --reentry {natural}"
            )
            return 1
        # Sides in the downloops sense (azimuth ~ sin(psi), right = az > 0).
        # The absolute orientation is a fixed convention (psi_exit = 3*pi/2,
        # exit left): flipping it would only mirror the whole cycle, so
        # there is no flag for it. The ONE real choice per count is HOW the
        # reel-in gets from the peel-off to the tangential landing:
        # "smooth" loops over the top on the exit side (plain lobe, no
        # crossing); "cross" loops on the OPPOSITE side, then the descent
        # crosses the ascending leg at lower elevation on its way to the
        # exit azimuth.
        s_exit = 1.0 if np.sin(float(ARTIFICIAL["psi_exit"])) >= 0.0 else -1.0
        s_entry = 1.0 if np.sin(float(ARTIFICIAL["psi_entry"])) >= 0.0 else -1.0
        ARTIFICIAL["az_reelin_amp"] = s_exit * abs(
            float(ARTIFICIAL["az_reelin_amp"])
        )
        shape = reelin_shape or "smooth"
        s_thr = s_exit if shape == "smooth" else -s_exit
        thr_mag = abs(float(ARTIFICIAL.get("az_reelin_through", 0.0)))
        ARTIFICIAL["az_reelin_through"] = (
            0.0 if shape == "smooth" else s_thr * thr_mag
        )

        def _side(sgn):
            return "az>0" if sgn > 0 else "az<0"

        print(
            f"[shape] reel-in: {shape} -- entry {_side(s_entry)}, top loop "
            f"{_side(s_thr)}, exit {_side(s_exit)} "
            f"({halves} visible half-lobes, {natural}-side re-entry)"
        )
    if reelin_center is not None:
        # CLI override of the window centre (reelin_bump validates the range).
        ARTIFICIAL["reelin_center"] = float(reelin_center)
    # The curvature limit is enforced ON THE FITTED SPLINE inside every
    # build_config call (fair_periodic_spline_to_curvature_limit): control
    # points around a violation move minimally, the parametric knobs are
    # never changed for curvature.
    arr = _load_experimental_cycle() if SHAPE_SOURCE == "experimental" else None

    phase = system_model = None
    art_used = None
    if auto:
        config, phase, system_model, art_used = _auto_tune(
            arr, curvature_limit=curvature_limit
        )
    elif close:
        config, phase, system_model = _close_cycle_depth(
            arr, curvature_limit=curvature_limit
        )
    else:
        config = build_config(
            arr, depower_depth=DEPOWER_DEPTH, curvature_limit=curvature_limit
        )
        if check:
            # Bootstrap sim: calibrate n_points to the MEASURED cycle duration
            # before the config is validated and written (one rebuild + re-sim
            # when the CYCLE_DURATION_S prior was materially off).
            phase, system_model = _simulate_cycle(config)
            n_old = int(config["reelout"]["sim_parameters"]["n_points"])
            measured = _measured_cycle_duration(phase, n_old)
            if measured is not None:
                n_new = int(
                    np.clip(np.ceil(measured * NPOINTS_PER_SECOND), 50, N_POINTS_MAX)
                )
                if abs(n_new - n_old) > 0.1 * n_old:
                    print(
                        f"[calib] measured cycle duration {measured:.1f} s -> "
                        f"n_points {n_old} -> {n_new}; rebuilding and re-simulating"
                    )
                    config = build_config(
                        arr,
                        depower_depth=DEPOWER_DEPTH,
                        duration_s=measured,
                        curvature_limit=curvature_limit,
                    )
                    phase, system_model = _simulate_cycle(config)
                else:
                    print(
                        f"[calib] measured cycle duration {measured:.1f} s; "
                        f"n_points {n_old} already consistent"
                    )

    try:
        enforce_curvature_limit(config["reelout"]["path_parameters"], curvature_limit)
    except ValueError as exc:
        print(f"\nGenerated config failed curvature check -- NOT written:\n{exc}")
        return 1

    # Self-check with the same validator run_full_cycle_opti.py applies on
    # load, so an inconsistent fit fails here instead of at the start of a
    # long optimization run.
    from run_full_cycle_opti import validate_full_cycle_config

    try:
        validate_full_cycle_config(config["reelout"])
    except ValueError as exc:
        print(f"\nGenerated config failed validation -- NOT written:\n{exc}")
        return 1

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, default_flow_style=False)
    print(f"Wrote {OUTPUT_PATH}")

    # Seed-path figure (geometry only): always saved, shown with --plot.
    art_plot = None
    if SHAPE_SOURCE == "artificial":
        art_plot = art_used if art_used is not None else ARTIFICIAL
    plot_seed_path(config, art=art_plot, arr=arr, save_path=SEED_PLOT_PATH, show=False)

    # Feasibility: simulate (or reuse the --auto/--close sim) and check the
    # trajectory against the bounds the NLP will impose on it as a warm start.
    if run_plots or (check and phase is None):
        phase, system_model = _simulate_cycle(config, run_plots=run_plots)
    if phase is not None:
        _feasibility_report(phase, system_model, config, artificial=art_used)

    if run_plots:
        plot_seed_path(config, art=art_plot, arr=arr, save_path=None, show=True)

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Show the seed-path figure (always saved to SEED_PLOT_PATH) and "
        "the overview plots of the simulated initial guess",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Forward-simulate the config and print the feasibility report; "
        "also recalibrates n_points to the measured cycle duration",
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Auto-tune the synthetic shape/depower until the seed is "
        "trim-feasible and radially closed at the optimizer's WIND_CONFIG "
        "(one forward sim per iteration)",
    )
    parser.add_argument(
        "--loops",
        type=int,
        default=None,
        help="Number of half figure-eights (lobes) flown during reel-out "
        "-- the count you see on the sphere, honoured exactly; the "
        "internal full-period n_loops and the parity-matching psi_entry "
        "are derived from it (combine with --auto to adapt the other "
        "knobs)",
    )
    parser.add_argument(
        "--reentry",
        choices=("opposite", "same"),
        default=None,
        help="Which side the reel-in exits on, relative to the lobe the "
        "kite left from. The exit is ALWAYS tangential at a side of the "
        "figure going down, so this is fixed by the --loops parity (odd = "
        "opposite, even = same): the flag validates the request and "
        "errors on a mismatch instead of flying a different shape",
    )
    parser.add_argument(
        "--reelin",
        dest="reelin_shape",
        choices=("smooth", "cross"),
        default=None,
        help="How the reel-in gets from the peel-off to the tangential "
        "landing (the sides themselves are not a choice: --loops parity "
        "ties exit to entry, and flipping both would only mirror the "
        "whole cycle). 'smooth' (default): top loop on the exit side, one "
        "clean arc, no crossing. 'cross': up the middle, out to the "
        "opposite side near the top, DOWN parked on that side, then cross "
        "az = 0 low (see --cross-at) to the exit azimuth",
    )
    parser.add_argument(
        "--cross-at",
        type=float,
        default=None,
        help="--reelin cross only: where on the descent the az = 0 "
        "crossing sits, as a fraction of the exit ramp (0 = right after "
        "the top, 1 = at figure height; default 0.7)",
    )
    parser.add_argument(
        "--close",
        action="store_true",
        help="Fit the reel-in depower depth until the simulated cycle closes",
    )
    parser.add_argument(
        "--curvature-limit",
        type=float,
        default=CURVATURE_LIMIT_1PM,
        help="Maximum sampled physical path curvature in 1/m (default "
        f"{CURVATURE_LIMIT_1PM}). Enforced on the fitted spline: control "
        "points around a violation are moved minimally (local fairing), "
        "the parametric knobs are never changed; the script refuses to "
        "write a config that still exceeds it. 0 disables both.",
    )
    parser.add_argument(
        "--reelin-center",
        type=float,
        default=None,
        help="Centre of the reel-in window in s; any value, mod 1 (the window "
        "wraps the periodic seam). Default 0.5 keeps s=0, the periodic seam "
        "and trim start, in steady reel-out. With f = reelout_fraction: "
        "(1-f)/2 starts the reel-in at s=0, 1-(1-f)/2 ends it there (s=0 = "
        "reel-out start), 0 puts s=0 at the reel-in top. Moving the seam "
        "off mid-reel-out -> harder trim start.",
    )
    args = parser.parse_args()
    raise SystemExit(
        main(
            run_plots=args.plot,
            check=args.check,
            close=args.close,
            auto=args.auto,
            loops=args.loops,
            reentry=args.reentry,
            reelin_shape=args.reelin_shape,
            cross_at=args.cross_at,
            curvature_limit=args.curvature_limit,
            reelin_center=args.reelin_center,
        )
    )

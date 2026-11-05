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

The trajectory is fully synthetic
— smooth, exactly periodic, tuned for trim feasibility — and NO flight data is
read; the winch law, start radius and pattern size come from the kite's
``cycle_profile.yaml`` (``--kite``, see ``cycle_kites.py``).

The result is written to
``data/<kite>/cycle_configs/full_cycle_periodic_from_exp.yaml`` and
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
``--size``) ``n_points`` is recalibrated to NPOINTS_PER_SECOND
nodes per second of the MEASURED cycle duration -- the ``CYCLE_DURATION_S``
prior can be over 2x off at other winds/shapes.

Usage:
    python .../fit_periodic_cycle_config.py [--size] [--loops N]
                                            [--reelin smooth|cross]
                                            [--cross-at P]
                                            [--reentry opposite|same]
                                            [--check] [--plot]
                                            [--curvature-limit K]

    --size   refit the path to the turns the kite holds at the target wind:
             the first margin on the steering-limited turn radius whose
             forward sim is complete and inside the steering limit at the
             kite's own roll gain. Writes nothing if none is
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
# Locate them relative to the repo's top-level ``examples`` dir so this works
# regardless of how deep under examples/ this lives.
_here = Path(__file__).resolve()
_examples_dir = next(p for p in _here.parents if p.name == "examples")
_repo_root = _examples_dir.parent
_SRC_DIR = _repo_root / "src"
for _path in (_SRC_DIR,):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
from cycle_kites import (  # noqa: E402
    DEFAULT_KITE,
    REQUIRED_SEED,
    SEED_FILENAME,
    override_wind,
    resolve_kite,
)

from awetrim.kinematics.parametrized_patterns import (  # noqa: E402
    DUBINS_TURN_RADIUS_M,
    LOBE_HANDOVER_PHASE,
    design_reelin_spline,
    fair_periodic_spline_to_curvature_limit,
    full_cycle_angles,
    full_cycle_n_loops_for_half_figures,
    full_cycle_visible_half_figures,
    make_full_cycle_bspline_path_parameters,
    periodic_bspline_basis_matrices,
)
from awetrim.kinematics.steering_limited_eight import (  # noqa: E402
    BlendedTurnLimit,
    CourseTurnLimit,
    fit_spline_to_turn_limit,
)
from awetrim.system.turn_radius import (  # noqa: E402
    steering_limit,
    steering_limited_turn_radius,
)
from awetrim.system.winch import Winch  # noqa: E402
from awetrim.utils.defaults import DEFAULT_OPTI_LIMITS  # noqa: E402
from awetrim.utils.system_config import get_kite  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration -- edit for a different resolution; everything that belongs to
# a KITE (system config, depower band, wind, data-derived winch law / r0 /
# duration, synthetic-shape knobs, reference flight) is a profile in
# ``cycle_kites.py`` and is loaded by ``configure_kite`` (``--kite``).
# ---------------------------------------------------------------------------
# Sim grid (the spline's M comes from _resolve_artificial_M).
# n_points is the optimizer grid over the whole cycle, at NPOINTS_PER_SECOND
# of SIMULATED cycle time: any forward sim (--check/--size) measures
# the real duration and recalibrates n_points (the CYCLE_DURATION_S prior can
# be far off -- the n_loops=5 seed assumed 213 s but flies in ~97 s at
# 13 m/s). Convergence sweep vs a 400-node reference (n=5, M=50): 3 nodes/s
# keeps mean cycle power within 0.2% with the steering peaks resolved;
# 2 nodes/s is the floor (-0.6% power, and the handover steering peaks start
# falling BETWEEN nodes -- the NLP only enforces bounds at nodes, so a too
# coarse grid hides violations instead of removing them); 1 node/s breaks the
# trim (AoA spikes). Capped for NLP tractability (nodes dominate NLP size).
NPOINTS_PER_SECOND = 3.0
N_POINTS_MAX = 2000

# Synthetic full-cycle parameters.
# The SIZE of the pattern (reelout_fraction, beta0, beta_amp0, az_amp0,
# beta_reelin_peak) has no default here: it is a property of the kite and is
# REQUIRED in the ``seed`` block of its cycle_profile.yaml (the values quoted
# below are the LEI-V3's). The profile may override any other knob too.
ARTIFICIAL = {
    "M": 40,  # control-point FLOOR; resolved as max(M, per-figure floor,
    # reel-in-window floor) -- see _resolve_artificial_M. A fixed M
    # under-resolves high loop counts and the underfit RINGS (spline
    # curvature far above the raw curve's).
    "n_loops": 6,  # INTERNAL count: continuous figure-eights over the WHOLE
    # period (~reelout_fraction * n_loops visible; the pinned handover phases
    # snap the visible count -- 4 here shows 2.5 figures = 5 half-eights
    # during reel-out, the opposite-side handover parity). The --loops CLI
    # knob is the WYSIWYG unit (visible reel-out HALF-figures) and overrides
    # this via full_cycle_n_loops_for_half_figures,
    # which may leave a non-integer value here. (2 loops dwell too long at
    # the azimuth extremes -> steering blows up; 4 loops turn too fast)
    # -- kite profile (REQUIRED in cycle_profile.yaml seed:) --
    # reelout_fraction: fraction of the period spent reeling out (V3 0.65;
    #   lower closes the cycle better but deepens the reel-in-edge steering)
    # beta0: reel-out base elevation (rad; V3 0.35 ~ 20 deg)
    # beta_amp0 / az_amp0: figure-eight elevation / azimuth amplitude (rad;
    #   V3 0.1 / 0.3). Steering demand of a figure-eight ~ 4*pi / T_loop
    #   (course angle sweeps +-2pi per lobe), so at the kite's natural speed
    #   the lever is arc length: BIG figures turn gentler. Scale both amps
    #   together, az ~ 2.5*beta; on the V3 a.36/b.14 is the sweet spot
    #   (larger starts spiking at the window edges).
    # beta_reelin_peak: reel-in peak elevation (rad; V3 1.2). What you set is
    #   what the raw curve does -- if the top U-turn is then too sharp for the
    #   curvature limit, the SPLINE is locally faired around it (the knob
    #   itself is never lowered). Too high starves the kite of apparent wind
    #   at the top -> AoA blows past its limit
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

# Per-kite constants, so the generator runs without any flight data. Set by
# ``configure_kite`` from the kite's
# folder (``cycle_kites.resolve_kite``):
#   R0                start radius (m)
#   CYCLE_DURATION_S  cycle-duration PRIOR (s) for a 3-lobe cycle; only sets
#                     n_points until a forward sim measures the actual
#                     duration (--check/--size recalibrate n_points
#                     from it -- at 13 m/s with 5 loops the V3 prior said
#                     213 s but the cycle flies in ~97 s)
#   WINCH_LAW         T = slope * (v_r - offset(u_p)); ``max_tether_force``
#                     is a HARDWARE rating replaced by the drum's own value
#                     from system.yaml (_winch_law_with_hardware_rating), the
#                     profile value is only the fallback for a system file
#                     without a drum entry
#   DEPOWER_BAND      (powered, depowered) ROM depower inputs from
#                     rom_config.yaml controls.input_depower, in the ROM's
#                     own u_p unit (V3: power-tape metres); every synthetic
#                     depower profile and the winch reference use it
#   DEPOWER_RATE      |d(u_p)/dt| bound emitted into the config (the KCU's
#                     actuation speed from system.yaml, else the default)
#   SYSTEM_CONFIG_PATH, OUTPUT_PATH, SEED_PLOT_PATH  the kite's files
KITE_NAME = None
KITE_DIR = None
SYSTEM_CONFIG_PATH = None
R0 = None
CYCLE_DURATION_S = None
WINCH_LAW = None
DEPOWER_BAND = None
DEPOWER_RATE = None
MIN_TURN_RADIUS = None  # m; emitted as sim_parameters.min_turn_radius when set
OUTPUT_PATH = None
SEED_PLOT_PATH = None
_ARTIFICIAL_BASE = None  # module ARTIFICIAL before any per-kite override


def configure_kite(name=None):
    """Load the kite ``name`` (a folder under data/ or a path to one) into
    this module's globals, from its system.yaml, rom_config.yaml and
    cycle_profile.yaml (``cycle_kites.resolve_kite``).

    Also configures ``run_full_cycle_opti`` (same kite, same wind) so the
    forward simulations here evaluate the seed at the optimizer's wind.
    Idempotent; the module is configured for ``DEFAULT_KITE`` on import
    (see the call below ARTIFICIAL).
    """
    global KITE_NAME, KITE_DIR, SYSTEM_CONFIG_PATH, R0, CYCLE_DURATION_S, WINCH_LAW
    global DEPOWER_BAND, DEPOWER_RATE, OUTPUT_PATH, SEED_PLOT_PATH
    global _ARTIFICIAL_BASE
    kite = resolve_kite(name)
    KITE_NAME = kite["name"]
    KITE_DIR = Path(kite["dir"])
    SYSTEM_CONFIG_PATH = Path(kite["system_config"])
    R0 = float(kite["r0"])
    CYCLE_DURATION_S = float(kite["cycle_duration_s"])
    WINCH_LAW = dict(kite["winch_law"])
    DEPOWER_BAND = tuple(kite["depower_band"])
    OUTPUT_PATH = Path(kite["cycle_config_dir"]) / SEED_FILENAME
    SEED_PLOT_PATH = Path(kite["results_dir"]) / "seed_path.png"
    # KCU depower actuation speed: the hardware value from system.yaml when
    # the control_system block states one, else the numerical default.
    with SYSTEM_CONFIG_PATH.open("r", encoding="utf-8") as f:
        system_cfg = yaml.safe_load(f)
    cs = get_kite(system_cfg).get("control_system", {}).get("structure", {})
    rate = (cs.get("depower") or {}).get("rate")
    DEPOWER_RATE = list(rate) if rate else list(DEFAULT_OPTI_LIMITS["depower_rate"])
    global MIN_TURN_RADIUS
    MIN_TURN_RADIUS = kite.get("min_turn_radius")
    # Synthetic-shape knobs: the module's method knobs overlaid with the
    # kite's seed block, which carries the pattern size (required, see
    # cycle_kites.REQUIRED_SEED) and may override any other knob.
    if _ARTIFICIAL_BASE is None:
        _ARTIFICIAL_BASE = dict(ARTIFICIAL)
    seed_knobs = dict(kite.get("artificial") or {})
    unknown = sorted(set(seed_knobs) - set(_ARTIFICIAL_BASE) - set(REQUIRED_SEED))
    if unknown:
        raise ValueError(
            f"{KITE_NAME}: unknown seed keys {unknown}; allowed are "
            f"{sorted(set(_ARTIFICIAL_BASE) | set(REQUIRED_SEED))}"
        )
    ARTIFICIAL.clear()
    ARTIFICIAL.update(_ARTIFICIAL_BASE)
    ARTIFICIAL.update(seed_knobs)
    # Optimizer bound overrides emitted into the config: the module defaults
    # plus the kite's own (the AoA range its ROM was identified on, from
    # rom_config.yaml validity). The depower-gain bound is in m/s per unit
    # u_p, so it scales with the kite's depower band: [-8 m/s over the full
    # band, 0], which is the V3's [-20, 0] on its 0.4 m band.
    OPTI_LIMITS_OVERRIDE.clear()
    OPTI_LIMITS_OVERRIDE.update(_OPTI_LIMITS_OVERRIDE_BASE)
    band_width = abs(DEPOWER_BAND[1] - DEPOWER_BAND[0])
    OPTI_LIMITS_OVERRIDE["winch_offset_depower_gain"] = [
        round(-_DEPOWER_GAIN_SPAN_M_S / band_width, 6),
        0.0,
    ]
    OPTI_LIMITS_OVERRIDE.update(kite.get("opti_limits_override") or {})
    import run_full_cycle_opti as _rfco

    _rfco.configure_kite(KITE_DIR)
    return kite


def _winch_law_with_hardware_rating(winch):
    """Overlay the drum's force rating from system.yaml onto a fitted law.

    The force-law SHAPE (slope, offset, depower gain) is regressed from
    flight data, but the clamp ``max_tether_force`` is a rating of the
    hardware -- it lives in system.yaml
    (``components.ground_station.drums[0].max_tether_force``) and must not be
    restated here. Missing drum rating -> the profile's value is kept, and
    one of the two must exist.
    """
    from awetrim.utils.system_config import get_drum

    with Path(SYSTEM_CONFIG_PATH).open("r", encoding="utf-8") as f:
        rating = get_drum(yaml.safe_load(f)).get("max_tether_force")
    winch = dict(winch)
    if rating is not None:
        winch["max_tether_force"] = float(rating)
    elif "max_tether_force" not in winch:
        raise ValueError(
            f"{KITE_NAME}: no tension rating -- set drums[0].max_tether_force "
            "in system.yaml or winch_law.max_tether_force in cycle_profile.yaml"
        )
    return winch


# Reel-in depower depth: fraction of the powered->depowered span the synthetic
# bump reaches (1.0 = fully depowered at the top). The seed need not close
# radially: the optimizer (free_speed winch) closes it in its first stage.
DEPOWER_DEPTH = 1.0

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
# Anti-hesitation floor emitted into the config (sigma' >= this x mean
# sigma'; see the emitted "parametric_speed_floor" below).
PARAMETRIC_SPEED_FLOOR = 0.2
# The winch drive's acceleration capability is HARDWARE: it lives in
# system.yaml (components.ground_station.drums[0].max_winch_acceleration) and
# reaches the NLP through SystemModel.hardware_limits, so it is deliberately
# NOT emitted into the generated config. Same for the reel-speed range
# (min/max_tether_speed -> the speed_radial bounds). Set
# sim_parameters["winch_acceleration"] only to make a single run more
# conservative than the hardware.

# Optimizer bound overrides emitted into the generated config. A full cycle
# reaches higher reel-in elevation than the default C_beta range, and
# winch_offset_depower_gain has no global default. run_full_cycle_opti.py reads
# these from the YAML -- this script is the single source of truth, so tune
# them here and regenerate instead of hand-editing the output file.
_OPTI_LIMITS_OVERRIDE_BASE = {
    "C_phi": [-1.0, 1.0],
    "C_beta": [0.01, 1.4],  # up to ~80 deg elevation for the reel-in arc
}
# The winch_offset_depower_gain bound is set per kite by configure_kite:
# [-_DEPOWER_GAIN_SPAN_M_S / depower band width, 0] (m/s per unit u_p).
_DEPOWER_GAIN_SPAN_M_S = 8.0
OPTI_LIMITS_OVERRIDE = dict(_OPTI_LIMITS_OVERRIDE_BASE)

configure_kite(DEFAULT_KITE)

# Output config (loaded verbatim by run_full_cycle_opti.py) and the seed-path
# figure (wind-window view + az/el/depower vs s, written on every successful
# generation next to the optimizer's own outputs; results/ is git-ignored, the
# YAML is tracked -- see plot_seed_path): OUTPUT_PATH / SEED_PLOT_PATH, set
# per kite by configure_kite.

# Geometric creation-time guard (1/m): refuse to write a path sharper than
# this sampled max curvature. Every build enforces it ON THE FITTED SPLINE
# (fair_periodic_spline_to_curvature_limit: control points around a violation
# move minimally, the parametric knobs are never touched), so the limit holds
# for ANY --loops value -- the sharpest point of the synthetic cycle is the
# reel-out/reel-in handover, and how sharp it is depends strongly on n_loops.
# Override with --curvature-limit; 0 disables both the fairing and the guard.
CURVATURE_LIMIT_1PM = 0.08
CURVATURE_SAMPLES = 1000

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

    ``depth`` scales the bump (fraction of the powered->depowered span).
    """
    from awetrim.kinematics.parametrized_patterns import reelin_bump

    bump = reelin_bump(
        s_grid,
        reelout_fraction=reelout_fraction,
        ramp_fraction=ramp_fraction,
        reelin_center=reelin_center,
    )
    return DEPOWER_BAND[0] + depth * bump * (
        DEPOWER_BAND[1] - DEPOWER_BAND[0]
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


def _artificial_path_parameters(art, r0=None):
    """YAML-ready spline path parameters for the synthetic knob set ``art``."""
    return make_full_cycle_bspline_path_parameters(
        M=_resolve_artificial_M(art),
        r0=float(R0 if r0 is None else r0),
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
        n_halves=art.get("n_halves"),
        reelin_turn_radius=art.get("reelin_turn_radius", DUBINS_TURN_RADIUS_M),
        reelin_min_turn_radius=art.get("reelin_min_turn_radius"),
        downloops=True,
    )


def build_config(
    depower_depth=1.0,
    duration_s=None,
    curvature_limit=CURVATURE_LIMIT_1PM,
):
    """Assemble the reelout config dict.

    The path is fully synthetic and uses the constants ``WINCH_LAW`` / ``R0`` /
    ``CYCLE_DURATION_S``. ``depower_depth`` scales the synthetic reel-in
    depower bump (``DEPOWER_DEPTH``).
    ``duration_s`` overrides the cycle-duration estimate that sets
    ``n_points`` -- pass the MEASURED duration of a forward sim (see
    ``_measured_cycle_duration``) so the grid tracks the actual flight time
    instead of the CYCLE_DURATION_S prior. ``curvature_limit`` (1/m, falsy
    disables) is enforced ON THE FITTED SPLINE: control points around any
    violation are moved minimally (see
    ``fair_periodic_spline_to_curvature_limit``) -- the parametric knobs are
    never changed to satisfy it.
    """
    art = ARTIFICIAL
    # Physical loop time is set by the figure geometry and the kite's trim
    # speed, so cycle duration grows ~linearly with loop count (the
    # reel-out/reel-in split stays roughly constant -- it is fixed by the
    # reel speed ratio). This is only a PRIOR: it knows nothing about the
    # target wind or the tuned shape, so any flow that forward-simulates
    # replaces it with the measured duration via ``duration_s``.
    duration = float(CYCLE_DURATION_S) * float(art["n_loops"]) / 3.0
    winch = _winch_law_with_hardware_rating(WINCH_LAW)
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
        # figures are never changed. (bow_shape "dubins" designs its
        # reel-in already: no sketch to repair, only the fairing below.)
        des = {"changed": False}
        if art.get("bow_shape") != "dubins":
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
                "depower_rate": list(DEPOWER_RATE),
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
                # equality; the sim always marches with the force law. The
                # rate limit and the v_r bounds come from the drum entry in
                # system.yaml, not from here.
                "winch_mode": WINCH_MODE,
                # No hesitation points: the spline's parametric speed sigma'
                # stays >= this fraction of its mean (PhaseParameterized;
                # replaced by turn_radius_sigma_floor when min_turn_radius is
                # set). Seeds sit at >= 0.28, converged optima at >= 0.47; the
                # artefact that broke the 8 m/s cycles collapses it to < 0.01.
                "parametric_speed_floor": PARAMETRIC_SPEED_FLOOR,
                # One closed period: the seam interval (last node -> node 0)
                # gets the continuity, rate and AoA rows and its energy and
                # time, and closes the cycle (run_full_cycle_opti then drops
                # the one-sided close_radial_cycle pin, which left the last
                # node unconstrained and the seam uncounted).
                "periodic_wrap": True,
                # s in [0, 1] over a whole cycle: the march's solves keep their
                # warm start at this small path rate (PhaseParameterized).
                "slow_path_rate": True,
            },
        }
    }
    if MIN_TURN_RADIUS:
        # Dense geometric floor on the flown turn radius (with its
        # hesitation-point companion): the node-wise NLP can otherwise fold
        # the spline into a cusp BETWEEN two nodes -- the kite stops on the
        # sphere for a fraction of a second, the tension collapses and the
        # trim still closes at every node.
        config["reelout"]["sim_parameters"]["min_turn_radius"] = float(MIN_TURN_RADIUS)
    return config


def _reelin_window_mask(s, art):
    """True inside the reel-in window (bump > 0) for the knob set ``art``."""
    f = float(art["reelout_fraction"])
    c = float(art.get("reelin_center", 0.5))
    h = 0.5 * (1.0 - f)
    d = (np.asarray(s, dtype=float) - c + 0.5) % 1.0 - 0.5
    return np.abs(d) <= h


def plot_seed_path(config, art=None, save_path=None, show=False):
    """Plot the generated seed trajectory (pure geometry, no forward sim).

    Left: wind-window view (azimuth vs elevation) of the fitted periodic
    B-spline the optimizer warm-starts from, with the analytic synthetic
    target it was fitted to, flight-direction arrows, the ``s = 0`` seam, the
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
            n_halves=art.get("n_halves"),
            r0=r0,
            reelin_turn_radius=art.get("reelin_turn_radius", DUBINS_TURN_RADIUS_M),
            reelin_min_turn_radius=art.get("reelin_min_turn_radius"),
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
    import run_full_cycle_opti as rfco

    from awetrim.system.factory import create_system_model_from_yaml

    # Evaluate the seed at the SAME wind the optimizer will use: a shape that
    # trims at one wind speed can be infeasible at another (higher wind
    # rides the max-force clamp, lower wind starves the reel-in top), so the
    # feasibility report is only meaningful at the target wind.
    WIND_CONFIG = rfco.WIND_CONFIG

    # Work on a copy with strict trimming OFF: this diagnostic run should show
    # how far the trim gets even when a node fails (the optimizer runs strict).
    reelout = copy.deepcopy(config["reelout"])
    reelout["sim_parameters"]["require_full_trajectory"] = False
    sim = reelout["sim_parameters"]

    system_model = create_system_model_from_yaml(yaml_path=SYSTEM_CONFIG_PATH)
    system_model.wind = rfco.build_wind_model(**WIND_CONFIG)
    print(
        f"Simulating seed ({KITE_NAME}) at WIND_CONFIG from run_full_cycle_opti: "
        + rfco.describe_wind(WIND_CONFIG)
    )

    # The optimizer's own march: the same start state, and a seed the kite
    # cannot turn through at its own roll gain is marched at the smallest
    # assisted one the optimizer would continue from (march_with_roll_assist).
    _, phase, assist, failure = rfco.march_with_roll_assist(
        system_model, reelout, run_plots=run_plots
    )
    if assist is not None and assist != 1.0:
        print(f"[info] the seed trims only at {assist:g} x the kite's roll gain "
              f"(at 1: {failure}); run_full_cycle_opti continues it back to 1")
    if phase is None:
        raise RuntimeError(f"seed march failed at every roll-gain scale: {failure}")
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


# --size: the path's turn radius is held at margin x the tightest turn the kite
# holds on that course (system.turn_radius), and the first margin whose march is
# complete and inside the steering limit at the kite's own roll gain is kept.
SIZE_MARGINS = (1.4, 1.6, 1.8, 2.0)
SIZE_HEIGHT_CLEARANCE_M = 2.0
SIZE_FIT_SAMPLES = 900
SIZE_FIT_MIN_RATIO = 0.99


def _size_turn_limit(config, system_model):
    """``limit(margin)`` of the seed: the steering-limited turn radius at its
    reel-out and at its reel-in depower and mean elevation, blended along the
    path by the depower profile. Also returns the seed's elevation samples."""
    reelout = config["reelout"]
    path = reelout["path_parameters"]
    r0 = float(path["r0"])
    profile = np.asarray(reelout["sim_parameters"]["input_depower_profile"], dtype=float)
    s_profile = np.linspace(0.0, 1.0, profile.size)
    powered, depowered = float(profile.min()), float(profile.max())

    def weight(u):
        return np.clip(
            (np.interp(u, s_profile, profile) - powered) / (depowered - powered), 0.0, 1.0
        )

    u = np.linspace(0.0, 1.0, 1000, endpoint=False)
    basis = periodic_bspline_basis_matrices(u, int(path["M"]))[0]
    elevation = basis @ np.asarray(path["C_beta"], dtype=float)
    winch = Winch(pattern_config=reelout["radial_parameters"])
    tables = []
    for depower, mask in (
        (powered, weight(u) < 0.5),
        (depowered, weight(u) >= 0.5),
    ):
        table = steering_limited_turn_radius(
            system_model,
            distance_radial=r0,
            angle_elevation=float(elevation[mask].mean()),
            input_depower=depower,
            winch=winch,
        )
        print(f"[size] depower {depower:.2f}: {table.summary()}")
        tables.append(table)

    def limit(margin):
        first, second = (CourseTurnLimit(t.radius, r0=r0, margin=margin) for t in tables)
        return BlendedTurnLimit(first, second, weight)

    return limit, elevation


def _march_own_roll_gain(config, system_model):
    """Forward simulation at the kite's own roll gain, without the assist:
    ``(phase, max |u_s|, reason)``, reason None when complete and within limits."""
    import run_full_cycle_opti as rfco

    reelout = copy.deepcopy(config["reelout"])
    reelout["sim_parameters"]["require_full_trajectory"] = False
    n_points = int(reelout["sim_parameters"]["n_points"])
    system_model.kite.gain_roll_scale = 1.0
    try:
        phase, _ = rfco.Phase(
            system_model=system_model,
            pattern_config=reelout,
            start_state=rfco.build_start_state(reelout),
        ).run_simulation(phase_sim=True, seed_warm_start=False)
    except RuntimeError as exc:
        return None, None, f"march failed ({exc})"
    steering = np.abs(np.asarray(phase.return_variable("input_steering"), dtype=float))
    if steering.size < n_points:
        return phase, float(steering.max()), f"march stopped at node {steering.size}/{n_points}"
    if steering.max() > steering_limit(system_model):
        return phase, float(steering.max()), (
            f"max |u_s| {steering.max():.3f} over the limit "
            f"{steering_limit(system_model):.3f}"
        )
    return phase, float(steering.max()), None


def _size_seed(config, curvature_limit, margins=SIZE_MARGINS):
    """The seed path refitted to the turns the kite holds at this wind, for the
    first of ``margins`` that marches. ``(config, phase, system_model)``,
    config None when none does."""
    import run_full_cycle_opti as rfco

    from awetrim.system.factory import create_system_model_from_yaml

    system_model = create_system_model_from_yaml(yaml_path=SYSTEM_CONFIG_PATH)
    system_model.wind = rfco.build_wind_model(**rfco.WIND_CONFIG)
    print(f"[size] wind: {rfco.describe_wind(rfco.WIND_CONFIG)}")
    limit, elevation = _size_turn_limit(config, system_model)
    reelout = config["reelout"]
    r0 = float(reelout["path_parameters"]["r0"])
    override = reelout["sim_parameters"].get("opti_limits_override") or {}
    height_min = override.get("height", DEFAULT_OPTI_LIMITS["height"])[0]
    for margin in margins:
        fitted, report = fit_spline_to_turn_limit(
            reelout["path_parameters"],
            limit(margin),
            M=None,
            mirror=False,
            el_min=float(np.arcsin((float(height_min) + SIZE_HEIGHT_CLEARANCE_M) / r0)),
            el_max=float(elevation.max()) + 0.05,
            n_samples=SIZE_FIT_SAMPLES,
        )
        print(
            f"[size] margin {margin:g}: {report['status']}, ratio {report['ratio']:.2f}, "
            f"turns >= {report['turn_radius_min_m']:.1f} m, moved "
            f"{np.degrees(report['max_move']):.1f} deg"
        )
        if report["ratio"] < SIZE_FIT_MIN_RATIO:
            continue
        trial = copy.deepcopy(config)
        trial["reelout"]["path_parameters"] = fitted
        phase, steering, reason = _march_own_roll_gain(trial, system_model)
        if reason is None:
            n_old = int(trial["reelout"]["sim_parameters"]["n_points"])
            measured = _measured_cycle_duration(phase, n_old)
            n_new = int(np.clip(np.ceil(measured * NPOINTS_PER_SECOND), 50, N_POINTS_MAX))
            if abs(n_new - n_old) > 0.1 * n_old:
                print(
                    f"[size] measured cycle duration {measured:.1f} s -> "
                    f"n_points {n_old} -> {n_new}"
                )
                trial = build_config(
                    depower_depth=DEPOWER_DEPTH,
                    duration_s=measured,
                    curvature_limit=curvature_limit,
                )
                trial["reelout"]["path_parameters"] = fitted
                phase, steering, reason = _march_own_roll_gain(trial, system_model)
        if reason is None:
            print(f"[size] margin {margin:g} marches: max |u_s| {steering:.3f}")
            return trial, phase, system_model
        print(f"[size] margin {margin:g}: {reason}")
    return None, None, system_model


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


# Steering gets a small slack over the hardware bound in the feasibility report:
# the flown KCU saturates at the bound and exact tracking of ANY production figure
# needs ~4*pi/T_loop of turn rate, so a few-percent overshoot at a handful of nodes
# is the physical floor -- the NLP clips it.
STEER_SLACK = 0.05


def _feasibility_metrics(phase, system_model, config):
    """Numeric feasibility summary of a simulated cycle.

    For the printed report: trim truncation,
    radial closure gap, and steering / AoA excursions vs the bounds the NLP
    will impose, with the worst-violation location in ``s`` for the
    knob<->symptom rules.
    """
    art = ARTIFICIAL
    sim = config["reelout"]["sim_parameters"]
    # The bounds the NLP will impose, in its precedence: the config's
    # opti_limits_override on top of the hardware limits on top of the
    # numerical defaults (PhaseParameterized._resolve_opti_limits + override).
    hw = dict(getattr(system_model, "hardware_limits", None) or {})
    hw.update(sim.get("opti_limits_override") or {})

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
    cycle. The radial closure gap is reported but not required (the
    optimizer closes the cycle). Violations are localized in
    ``s`` and mapped to the generator knob that relieves them. Returns True
    for a clean warm start.
    """
    m = _feasibility_metrics(phase, system_model, config)

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
        # Information only: the seed marches its own force law, which the
        # free_speed optimizer drops; its first stage closes the cycle.
        print(
            f"[info] radial closure: r_end - r0 = {m['gap']:+.1f} m "
            f"(r0 = {m['r0']:.1f} m); the optimizer's first stage closes it"
        )

    if m["steer_n_viol"] and m["steer_excess"] <= STEER_SLACK:
        # Within the warm-start slack: the flown KCU saturates at exactly this
        # bound (commanding well past it), so exact tracking of any production
        # figure needs a little more than the actuator gives -- the NLP clips
        # these nodes. Do not fail the verdict on them.
        print(
            f"[WARN] steering grazes [{m['steer_lb']}, {m['steer_ub']}] at "
            f"{m['steer_n_viol']} node(s) "
            f"(min {m['steer_min']:.2f}, max {m['steer_max']:.2f}) -- within "
            f"the +-{STEER_SLACK:.2f} warm-start slack, NLP clips these"
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
            else "NOT feasible as a warm start -- run --size or tune the knobs"
        )
    )
    return ok


def main(
    run_plots: bool = False,
    check: bool = False,
    loops: int = None,
    reentry: str = None,
    reelin_shape: str = None,
    cross_at: float = None,
    curvature_limit: float = CURVATURE_LIMIT_1PM,
    reelin_center: float = None,
    kite: str = None,
    wind_speed: float = None,
    wind_height: float = None,
    output: str = None,
    size: bool = False,
) -> int:
    global OUTPUT_PATH, SEED_PLOT_PATH
    configure_kite(kite or KITE_DIR)
    # The forward sims evaluate the seed at run_full_cycle_opti's wind
    # (_simulate_cycle), so the override goes there.
    import run_full_cycle_opti as _rfco

    _rfco.WIND_CONFIG = override_wind(_rfco.WIND_CONFIG, wind_speed, wind_height)
    if output:
        OUTPUT_PATH = Path(output)
        SEED_PLOT_PATH = OUTPUT_PATH.with_name(OUTPUT_PATH.stem + "_seed_path.png")
    print(f"Kite: {KITE_NAME} ({SYSTEM_CONFIG_PATH})")
    if cross_at is not None:
        ARTIFICIAL["reelin_cross_pos"] = float(cross_at)
    if reelin_shape == "dubins":
        # Designed reel-in: --loops is taken as-is (visible reel-out lobes
        # after the landing half-lobe); n_loops only sizes the spline (M).
        # Its arcs may tighten down to the kite's turn-radius floor.
        if not loops:
            print("--reelin dubins needs --loops")
            return 1
        ARTIFICIAL["bow_shape"] = "dubins"
        ARTIFICIAL["n_halves"] = int(loops)
        ARTIFICIAL["n_loops"] = (0.5 * int(loops) + 1.0) / float(
            ARTIFICIAL["reelout_fraction"]
        )
        if MIN_TURN_RADIUS:
            ARTIFICIAL["reelin_min_turn_radius"] = float(MIN_TURN_RADIUS)
        print(
            f"[shape] reel-in: dubins -- {int(loops)} visible lobes, arcs "
            f"{ARTIFICIAL.get('reelin_turn_radius', DUBINS_TURN_RADIUS_M):g} m "
            f"down to {ARTIFICIAL.get('reelin_min_turn_radius') or 'half that'} m"
        )
        loops = None
    if loops:
        # CLI override in the WYSIWYG unit: half figure-eights (lobes) VISIBLE
        # during reel-out. Derive the internal continuous full-period count
        # (and the parity-matching psi_entry -- the count is always honoured
        # EXACTLY; the peel-off crossing mirrors by pi as its parity needs);
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
            other = (
                loops + 1 if loops else "an odd" if reentry == "opposite" else "an even"
            )
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
        ARTIFICIAL["az_reelin_amp"] = s_exit * abs(float(ARTIFICIAL["az_reelin_amp"]))
        shape = reelin_shape or "smooth"
        s_thr = s_exit if shape == "smooth" else -s_exit
        thr_mag = abs(float(ARTIFICIAL.get("az_reelin_through", 0.0)))
        ARTIFICIAL["az_reelin_through"] = 0.0 if shape == "smooth" else s_thr * thr_mag

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
    phase = system_model = None
    config = build_config(depower_depth=DEPOWER_DEPTH, curvature_limit=curvature_limit)
    if size:
        config, phase, system_model = _size_seed(config, curvature_limit)
        if config is None:
            print(
                f"\nNo margin of {SIZE_MARGINS} marches at this wind -- NOT "
                "written. Continue from a neighbouring wind instead "
                "(run_full_cycle_opti.py --continue-from)."
            )
            return 1
    elif check:
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
    plot_seed_path(config, art=ARTIFICIAL, save_path=SEED_PLOT_PATH, show=False)

    # Feasibility: simulate (or reuse the --size sim) and check the
    # trajectory against the bounds the NLP will impose on it as a warm start.
    if run_plots or (check and phase is None):
        phase, system_model = _simulate_cycle(config, run_plots=run_plots)
    if phase is not None:
        _feasibility_report(phase, system_model, config)

    if run_plots:
        plot_seed_path(config, art=ARTIFICIAL, save_path=None, show=True)

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
        "--loops",
        type=int,
        default=None,
        help="Number of half figure-eights (lobes) flown during reel-out "
        "-- the count you see on the sphere, honoured exactly; the "
        "internal full-period n_loops and the parity-matching psi_entry "
        "are derived from it",
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
        choices=("smooth", "cross", "dubins"),
        default=None,
        help="How the reel-in gets from the peel-off to the tangential "
        "landing (the sides themselves are not a choice: --loops parity "
        "ties exit to entry, and flipping both would only mirror the "
        "whole cycle). 'smooth' (default): top loop on the exit side, one "
        "clean arc, no crossing. 'cross': up the middle, out to the "
        "opposite side near the top, DOWN parked on that side, then cross "
        "az = 0 low (see --cross-at) to the exit azimuth. 'dubins': the "
        "reel-in is DESIGNED instead of faded out -- bounded-curvature "
        "Dubins paths on the sphere from the peel-off to a level apex at "
        "beta_reelin_peak and down to the landing (needs --loops; see "
        "parametrized_patterns.dubins_full_cycle_angles)",
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
    parser.add_argument(
        "--kite",
        default=DEFAULT_KITE,
        help="Kite folder under data/ (or a path to one) holding system.yaml, "
        "rom_config.yaml and cycle_profile.yaml (wind, winch law, seed size; "
        f"see cycle_kites.py); default {DEFAULT_KITE}",
    )
    parser.add_argument(
        "--wind-speed",
        type=float,
        default=None,
        help="Wind speed (m/s) at --wind-height replacing the kite profile's "
        "amplitude (law and z0 kept); pass the same to run_full_cycle_opti.py",
    )
    parser.add_argument(
        "--wind-height",
        type=float,
        default=None,
        help="Height (m) of --wind-speed (default the profile's height_ref)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Seed YAML to write instead of the kite's "
        f"cycle_configs/{SEED_FILENAME} (pass it to run_full_cycle_opti.py "
        "--config); the seed-path figure goes next to it",
    )
    parser.add_argument(
        "--size",
        action="store_true",
        help="Refit the seed path to the turns the kite holds at this wind: "
        f"the first of the margins {SIZE_MARGINS} on the steering-limited turn "
        "radius whose forward simulation is complete and inside the steering "
        "limit at the kite's own roll gain. Writes nothing if none is",
    )
    args = parser.parse_args()
    raise SystemExit(
        main(
            kite=args.kite,
            run_plots=args.plot,
            check=args.check,
            loops=args.loops,
            reentry=args.reentry,
            reelin_shape=args.reelin_shape,
            cross_at=args.cross_at,
            curvature_limit=args.curvature_limit,
            reelin_center=args.reelin_center,
            wind_speed=args.wind_speed,
            wind_height=args.wind_height,
            output=args.output,
            size=args.size,
        )
    )

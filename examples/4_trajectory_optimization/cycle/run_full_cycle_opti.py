"""Full pumping-cycle optimization as a single periodic spline + depower profile.

Instead of stitching a reel-out ``Phase`` to a parametrized ``ReelinSimple``
(``awetrim.timeseries.Cycle``), the **entire** pumping cycle is one
``spline_periodic`` ``Phase``:

  * the path is a single periodic B-spline over the whole cycle (reel-out
    figures + reel-in), generated -- synthetic and trim-feasible by default --
    by ``fit_periodic_cycle_config.py``;
  * the depower input ``l_dp(s)`` is one per-node profile spanning the whole cycle
    (``optimize_depower_profile``), free for the optimizer; and
  * the winch force law carries a **depower-dependent offset**
    (``winch_offset_depower_gain``) so the same law reels out when powered and
    reels in when depowered -- this is what makes a one-phase cycle close; and
  * the operating radius ``r0`` is a free design parameter: the NLP's node-0
    radius pin follows it and the hard closure ``r[-1] == r[0]`` drags the
    whole radius band with it (bounded by ``DEFAULT_OPTI_LIMITS["r0"]`` and
    the per-node ``distance_radial`` band).

The wind is a constant analytic profile, the ``wind:`` block of the kite's
cycle_profile.yaml (``--wind-config`` replaces the block, ``--wind-speed`` /
``--wind-height`` override its speed).
The objective is cycle-average mechanical power (a single monolithic NLP, no
alternating / no separate reel-in phase).

Run ``fit_periodic_cycle_config.py`` first to (re)generate the config.

On a successful solve, detailed seed-vs-optimized comparison figures
(trajectory / wind window, per-node time series, power + energy + winch plane
+ metrics table) are built straight from the NLP's own trajectory -- no
re-simulation -- and saved as PNGs next to the optimized YAML/CSV in
``RESULTS_DIR`` (see ``cycle_comparison_plots.py``); ``--plot`` additionally
shows them interactively.

Usage:
    python examples/4_trajectory_optimization/cycle/run_full_cycle_opti.py
        [--kite KITE_FOLDER] [--plot] [--no-optimize]
"""

import argparse
import copy
import math
from pathlib import Path

import numpy as np
import yaml
from scipy.signal import find_peaks

from cycle_kites import (
    DEFAULT_KITE,
    SEED_FILENAME,
    load_wind_config,
    override_wind,
    resolve_kite,
)
from cycle_kites import wind_tag as _wind_tag

from awetrim.environment.wind_factory import create_wind_model
from awetrim.kinematics.parametrized_patterns import (
    PeriodicBSpline,
    reelin_control_point_mask,
)
from awetrim.system.factory import create_system_model_from_yaml
from awetrim.utils.time_scheme import left_step, trapezoid_increment, trapezoid_step
from awetrim.timeseries import roll_assist
from awetrim.timeseries.phase import Phase, step_box_active, trajectory_mean_power

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Everything that belongs to a KITE is read from its data folder
# (system.yaml, the ROM config it names, cycle_profile.yaml; see ``cycle_kites.py``)
# into these globals by ``configure_kite`` (``--kite``; the module is
# configured for DEFAULT_KITE on import):
#   KITE_CONFIG_PATH   the kite's system.yaml (mass, tether, drum envelope,
#                      KCU actuator ranges -> the optimizer's hardware limits)
#   CYCLE_CONFIG_PATH  the seed written by fit_periodic_cycle_config.py
#   RESULTS_DIR        optimized YAML/CSV/figures
#   DEPOWER_BAND       (powered, depowered) ROM depower inputs the seed's
#                      profile must stay inside, in the ROM's u_p unit (the
#                      reference flight's matched tapes, else the ROM's
#                      controls.input_depower; cycle_kites.resolve_kite)
#   WIND_CONFIG        the constant wind profile the cycle is optimized in --
#                      ``model_type`` selects the law in
#                      awetrim.environment.profile_laws (or "tabulated" with
#                      heights/speeds); an analytic law's amplitude is
#                      ``speed_wind_ref`` at ``height_ref``; every other key is
#                      forwarded to ``create_wind_model`` (see
#                      build_wind_model). Regenerate the seed
#                      (fit_periodic_cycle_config.py --size) after changing
#                      it: an old seed will not be trim-feasible at a
#                      different wind. Wind DIRECTION is irrelevant (the whole
#                      cycle lives in the course-aligned wind frame).
KITE_NAME = None
KITE_DIR = None
KITE_CONFIG_PATH = None
CYCLE_CONFIG_PATH = None
RESULTS_DIR = None
# Kite (and tether) dynamics in the march and the NLP: --dynamic.
QUASI_STEADY = True
DEPOWER_BAND = None
WIND_CONFIG = None


def configure_kite(name=None):
    """Load the kite ``name`` (a folder under data/ or a path to one) into
    this module's globals (``cycle_kites.resolve_kite``)."""
    global KITE_NAME, KITE_DIR, KITE_CONFIG_PATH, CYCLE_CONFIG_PATH, RESULTS_DIR
    global DEPOWER_BAND, WIND_CONFIG
    kite = resolve_kite(name)
    KITE_NAME = kite["name"]
    KITE_DIR = Path(kite["dir"])
    KITE_CONFIG_PATH = Path(kite["system_config"])
    CYCLE_CONFIG_PATH = Path(kite["cycle_config_dir"]) / SEED_FILENAME
    RESULTS_DIR = Path(kite["results_dir"])
    DEPOWER_BAND = tuple(kite["depower_band"])
    WIND_CONFIG = dict(kite["wind"])
    return kite


configure_kite(DEFAULT_KITE)

# Staged solve, warm-started stage to stage.
#
# Closure must be a HARD equality for the power rounds, NOT a soft penalty. A
# penalty lives in the objective, so it does not register in inf_pr and the
# optimizer quietly opens the cycle: the power reward for reeling out forever
# beats any finite penalty (exterior penalty is exact only as w -> inf), the
# radius runs away, and the per-node trim blows up (objective -> -6, inf_pr
# explodes). The hard equality makes the runaway simply infeasible. The seed
# is generally OPEN (its forward simulation marches the seed's force law,
# which free_speed drops), so the first stage is a pure feasibility round
# with the hard closure, and the power rounds start from a closed point.
#
# Step control is PROXIMAL, not boxed: ``trust_region_weight`` adds
# w * sum(((p - p_warmstart) / scale)^2) to the objective, with scale = the
# parameter's DEFAULT_OPTI_LIMITS bound width (Phase._trust_region_penalty).
# The power objective is a flat ridge, so without damping IPOPT takes huge
# steps (||d|| ~ 1e2) along near-null shape directions that merge/destroy the
# figure-eight loops and strand the iterate outside the trim-feasible region
# (inf_pr explodes, never recovers). The quadratic anchor adds curvature to
# exactly those null directions -- bounding the step -- while barely resisting
# directions with a real power gradient, so ONE solve travels the whole way to
# the optimum (no re-centered repeat passes, the old scheme). The anchor
# biases the optimum toward the seed, so a second, warm-started polish stage
# re-anchors at the main optimum with 10x less weight and shakes the bias out.
#
# ``step_bounds`` is a hard +/-delta box per solve. The closing stage keeps a
# tight one (+/-0.15 rad, +/-50 m): it needs only a small move, and a wider
# box let it reshape the pattern or stall locally infeasible. The power and
# polish boxes are 4x that: a box-scale study (LEI-V3, 7 half-figures, 2026-10-28)
# found the tight box binding at the optimum and costing 3% of the power at
# 6-10 m/s at 100 m and 15% at 12 m/s; with this split every case converged
# without a topology change, at the best power of all variants (1.45 / 8.26 /
# 11.64 kW at 6 / 10 / 12 m/s). The driver warns when a box is active.
# Roll-gain continuation. A seed is a fixed path; a kite that turns less per
# unit steering than its geometry assumes (the LEI-V3 bare aerostructural ROM:
# 1.6x the flown steering for the flown turns) cannot be forward-simulated
# through its tight parts, although a cycle exists that it flies. The seed is
# then marched with the kite's roll gain k(u_p) scaled up -- the smallest of
# ROLL_ASSIST_SCALES that trims it -- the first (feasibility) stage is solved
# there, and continuation stages step the scale back to 1, each halving the
# excess (to 1 once below ROLL_ASSIST_MIN_STEP) and warm-started from the
# previous optimum; a failed step is retried from its midpoint, at most
# ROLL_ASSIST_MAX_HALVINGS times. The power and polish stages always run at
# scale 1: the result is the kite's own optimum. A seed that trims at scale 1
# is optimized exactly as before.
# The ladder, the march and the halving schedule are awetrim.timeseries.
# roll_assist (shared with the reel-out server's cold solve).


def march_with_roll_assist(system_model, pattern_config, *, run_plots=False,
                           seed_warm_start=False, scales=None):
    """``roll_assist.march_with_roll_assist`` on a fresh Phase of
    ``pattern_config`` started from ``build_start_state``. Returns ``(cycle,
    phase, scale, failure)``; see there."""
    return roll_assist.march_with_roll_assist(
        system_model.kite,
        lambda: Phase(
            system_model=system_model,
            pattern_config=pattern_config,
            start_state=build_start_state(pattern_config),
            quasi_steady=QUASI_STEADY,
        ),
        int(pattern_config["sim_parameters"]["n_points"]),
        scales=scales,
        run_plots=run_plots,
        seed_warm_start=seed_warm_start,
    )


def _roll_assist_stages(stages, assist):
    """The stage list for a seed marched at roll-gain scale ``assist``."""
    if not assist or assist == 1.0:
        return [dict(stage, roll_scale=1.0) for stage in stages]
    plan = [dict(stages[0], roll_scale=assist)]
    for scale in roll_assist.continuation_scales(assist):
        plan.append(dict(stages[0], roll_scale=scale, continuation=True, halvings=0,
                         label=f"roll-gain continuation | k(u_p) x {scale:g}"))
    return plan + [dict(stage, roll_scale=1.0) for stage in stages[1:]]


def _retry_continuation(stages, i) -> bool:
    """Insert the midpoint before a failed continuation step ``i``."""
    stage = stages[i]
    if not stage.get("continuation") or stage["halvings"] >= roll_assist.ROLL_ASSIST_MAX_HALVINGS:
        return False
    previous = stages[i - 1]["roll_scale"]
    mid = 0.5 * (previous + stage["roll_scale"])
    stage["halvings"] += 1
    stages.insert(i, dict(stage, roll_scale=mid,
                          label=f"roll-gain continuation | k(u_p) x {mid:g} (step halved)"))
    print(f"    continuation step failed; retrying from the midpoint x {mid:g}")
    return True


#: Times a failed power stage is retried with its step box halved.
MAX_BOX_HALVINGS = 2
#: A stalled power stage whose last iterate is this close to feasible (IPOPT
#: inf_pr on the hand-scaled rows) is resumed from it, at most MAX_RESUMES times.
RESUME_TOLERANCE = 0.05
MAX_RESUMES = 2
#: An optimum on the step box's edge is solved again with the box re-centred,
#: at most MAX_RECENTRES times, until the pass gains less than RECENTRE_MIN_GAIN.
MAX_RECENTRES = 3
RECENTRE_MIN_GAIN = 1e-3


def _trajectory_mean_power(trajectory):
    """Cycle-mean power [W] of an optimized trajectory (awetrim.timeseries.phase)."""
    return trajectory_mean_power(trajectory)


def _resume_stalled(stage, cycle) -> bool:
    """True when a failed stage left a nearly feasible iterate behind
    (Phase.run_simulation_opti's resume_tolerance) and may run again from it.

    A stall -- the iteration limit reached while still climbing the flat power
    ridge with tiny accepted steps -- is not a wrong region: the next solve,
    from that iterate with a fresh box, barrier and quasi-Newton memory, goes
    on where it stopped (the turn-rate ROM at 7 m/s stopped at a trim residual
    of 6.5e-3). Restarting from the previous optimum instead repeats the same
    crawl."""
    if not getattr(cycle, "adopted_failed_iterate", False):
        return False
    if stage.get("resumes", 0) >= MAX_RESUMES:
        return False
    stage["resumes"] = stage.get("resumes", 0) + 1
    print(f"    stage stalled near feasibility; resuming from its last iterate "
          f"({stage['resumes']}/{MAX_RESUMES}), step box re-centred")
    return True


def _recentre_box(stage, previous_power, power) -> bool:
    """True when an optimum on the step box's edge should be solved again with
    the box re-centred on it: the box, not the optimum, stopped the move. Not
    after MAX_RECENTRES passes, nor once a re-centred pass gained less than
    RECENTRE_MIN_GAIN of the power (the box only touched)."""
    if stage.get("recentres", 0) >= MAX_RECENTRES:
        return False
    if previous_power is not None and power is not None:
        gain = (power - previous_power) / max(abs(previous_power), 1e-9)
        if gain < RECENTRE_MIN_GAIN:
            print(f"    step box active, but the re-centred pass gained {gain:.2%}: done")
            return False
    stage["recentres"] = stage.get("recentres", 0) + 1
    print(f"    step box active at the optimum; solving again with the box re-centred "
          f"({stage['recentres']}/{MAX_RECENTRES})")
    return True


def _retry_halved_box(stages, i) -> bool:
    """Halve the step box of a failed power stage ``i`` and run it again.

    The box bounds how far one solve may move the path. A box too wide for the
    kite lets one IPOPT step tear the figure eight apart, and the iterate ends
    where the kite cannot trim (restoration, iteration limit); the same stage
    from the same point converges with a smaller box (the turn-rate ROM at
    6 m/s: +-0.6 rad failed, +-0.3 converged). The wide box stays the first try,
    since a binding box costs power. The closing stage keeps its own box
    (``scale_box: False``)."""
    stage = stages[i]
    if not stage.get("step_bounds") or stage.get("scale_box", True) is False:
        return False
    if stage.get("box_halvings", 0) >= MAX_BOX_HALVINGS:
        return False
    # the stage's own box, restored for re-centred passes once a halved one converged
    stage.setdefault("step_bounds_full", dict(stage["step_bounds"]))
    stage["step_bounds"] = {k: 0.5 * float(v) for k, v in stage["step_bounds"].items()}
    stage["box_halvings"] = stage.get("box_halvings", 0) + 1
    print(f"    stage failed; retrying it with the step box halved "
          f"({stage['box_halvings']}/{MAX_BOX_HALVINGS}): {stage['step_bounds']}")
    return True


STAGES = [
    # Feasibility first: the seed's forward simulation marches the seed's
    # force law and is generally OPEN (the free_speed NLP drops that law), so
    # the first solve only projects it onto the closed, trimmed set -- the
    # objective is the proximal anchor alone ("zero" target), i.e. the
    # smallest move that closes the cycle. Climbing the power ridge from an
    # infeasible start overshoots into regions the trim cannot hold. The
    # depower profile is HELD at the seed's (fix_depower): a free profile has
    # no productivity incentive in this round and closes the cycle by reeling
    # in fully powered, which the power round can only undo through large
    # moves (failed at 8 m/s at 100 m); held, the round closes the cycle
    # through the free reel speed and the shape.
    {
        "label": "close | feasibility, proximal anchor only",
        "params": ["C_phi", "C_beta", "r0"],
        "target": "zero",
        "fix_depower": True,
        # A CONTINUATION (--continue-from) anchors the profile instead
        # (sim_parameters.depower_anchor_weight): its depower was optimal at
        # another wind, and in the rated region the extra force can only be
        # shed by depowering -- held, 11 -> 12 m/s at 100 m was locally
        # infeasible with the reel-out speed at its cap.
        "depower_anchor": 0.05,
        # The one-sided radius pin, not the seam closure: a seed open by tens
        # of metres ends with steering and reel speed far from node 0, and
        # closing it while also holding the seam's rate rows was locally
        # infeasible at 12 m/s at 100 m. The later stages close the seam.
        "periodic": False,
        "closure_weight": 0.0,
        "hard_close": True,
        "warm_start": False,
        "trust_region_weight": 0.05,
        "step_bounds": {"C_phi": 0.15, "C_beta": 0.15, "r0": 50.0},
        # This stage needs only a small move; a wider box let it reshape the
        # pattern (10 m/s at 100 m) or stall locally infeasible (12 m/s), so
        # --step-scale leaves it alone.
        "scale_box": False,
    },
    # ``r0`` frees the operating radius: the NLP's node-0 radius pin follows
    # the r0 parameter (seeded from path_parameters["r0"], bounded by
    # DEFAULT_OPTI_LIMITS["r0"]) and the hard closure r[-1] == r[0] drags the
    # whole radius band with it.
    {
        "label": "shape+r0 | proximal trust region",
        "params": ["C_phi", "C_beta", "r0"],
        "closure_weight": 0.0,
        "hard_close": True,
        # Anchor weight: a typical old per-pass move (0.15 rad over a ~1.6
        # bound width) costs a few % of the O(1) scaled power objective.
        # Raise it if the topology-signature warning fires; lower it if the
        # polish stage still moves the shape far.
        "warm_start": True,  # tight-barrier start from the closed point
        "trust_region_weight": 0.2,
        "step_bounds": {"C_phi": 0.6, "C_beta": 0.6, "r0": 200.0},
        "repeat": 1,  # old scheme: 4 passes, re-centered each time; new scheme: 1 pass
    },
    {
        "label": "polish | weak anchor",
        "params": ["C_phi", "C_beta", "r0"],
        "closure_weight": 0.0,
        "hard_close": True,
        "trust_region_weight": 0.005,
        "warm_start": True,  # tight-barrier IPOPT start from the main optimum
        "step_bounds": {"C_phi": 0.6, "C_beta": 0.6, "r0": 400.0},
    },
]

# Per-control-point backstop box. The scalar C_phi / C_beta deltas in
# ``step_bounds`` are expanded into per-point vectors before each pass (see
# ``_expand_step_bounds``): control points whose B-spline support overlaps the
# reel-in arc get REELIN_STEP_SCALE times the base delta. The reel-in bow is
# a single arc with no loops to destroy -- and the part of the seed that was
# FLOWN rather than optimized, so it has the most headroom. The mask is
# rebuilt every pass because the arc drifts in s as the shape moves. Set
# REELIN_STEP_SCALE = 1.0 to recover the uniform box.
REELIN_STEP_SCALE = 1.0
# Samples in the top fraction of the elevation range count as the reel-in arc.
REELIN_ELEVATION_FRACTION = 0.75

# NLP iteration cap per stage. Every other solver key (opti_limits_override,
# expand_nlp, ipopt_robust, optimize_depower_profile, require_full_trajectory)
# comes from the YAML, which fit_periodic_cycle_config.py emits as the single
# source of truth -- validate_full_cycle_config() refuses configs that lack
# them or that mix path / depower / winch data from different sources.
MAX_ITER = 1000

# Control-smoothness term of the power stages as a fraction of the power
# objective at each stage's start (sim_parameters.reg_fraction). On the LEI-V3
# at 6 m/s a fixed weight of 10 (about this fraction) cost 0.2 % (bare ROM) to
# 1.2 % (flight-corrected) of the power and halved the steering's total
# variation; a weight of 50 cost 3 %. An explicit reg_weight in the seed wins.
REG_FRACTION = 0.1

# Downsample the loaded config to this many nodes before solving (None = keep the
# config's n_points). The depower profile + per-node trim + closure all scale with
# n_points, so this is the single biggest conditioning lever -- 150 nodes resolves
# the path fine and solves far more steadily than 400. The spline shape (M control
# points) is independent of n_points, so only the sim grid / depower profile change.
N_POINTS = None


def build_wind_model(
    model_type, speed_wind_ref=None, height_ref=100.0, **profile_kwargs
):
    """Wind model with the reference speed ``speed_wind_ref`` given at ``height_ref``.

    ``model_type`` is any analytic law of ``awetrim.environment.profile_laws``
    (uniform, logarithmic, power_law, explog, jet) or ``"tabulated"`` (then
    ``heights``/``speeds`` replace the reference amplitude); extra keys such
    as ``z0``, ``alpha``, ``jet_amplitude``/``jet_height``/``jet_width`` or
    ``direction_wind`` are forwarded to ``create_wind_model``.
    """
    return create_wind_model(
        model_type,
        U_ref=speed_wind_ref,
        z_ref=height_ref,
        **profile_kwargs,
    )


def describe_wind(wind_config=None):
    """One-line human description of a WIND_CONFIG."""
    cfg = WIND_CONFIG if wind_config is None else wind_config
    if cfg["model_type"] == "tabulated":
        pairs = ", ".join(f"{v:g}@{h:g}" for h, v in zip(cfg["heights"], cfg["speeds"]))
        return f"tabulated profile (m/s @ m): {pairs}"
    text = (
        f"{cfg['model_type']} profile, {cfg['speed_wind_ref']:g} m/s @ "
        f"{cfg['height_ref']:g} m"
    )
    if "z0" in cfg:
        text += f", z0={cfg['z0']:g}"
    if cfg["model_type"] == "jet":
        text += (
            f", jet +{cfg['jet_amplitude']:g} m/s @ {cfg['jet_height']:g} m "
            f"(width {cfg['jet_width']:g} m)"
        )
    if float(cfg.get("veer_rate", 0.0)) != 0.0:
        text += f", veer {math.degrees(float(cfg['veer_rate'])):g} deg/m"
        if "height_direction_ref" in cfg:
            text += f" about {cfg['height_direction_ref']:g} m"
    return text


def wind_tag(wind_config=None):
    """Filename tag identifying the wind the run was solved at."""
    return _wind_tag(WIND_CONFIG if wind_config is None else wind_config)


def load_reelout_config(path):
    with Path(path).open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if "reelout" not in cfg:
        raise KeyError(
            f"{path} has no 'reelout' section -- run fit_periodic_cycle_config.py"
        )
    return cfg["reelout"]


def validate_full_cycle_config(reelout_config):
    """Fail loudly when the cycle config is stale, hand-edited, or mixed.

    The single-phase cycle only trims when the path spline, the depower profile
    and the winch's depower offset are mutually consistent -- i.e. when they all
    come straight out of ``fit_periodic_cycle_config.py``. A config assembled
    from different sources breaks the per-node trim (and every NLP stage seeded
    from it) in hard-to-trace ways, so refuse it up front. Checks:

      * the solver keys the fit script emits are present;
      * the depower profile has length ``n_points + 1``, stays inside the
        powered..depowered actuation band, and matches ``input_depower`` at
        node 0;
      * no "powered while parked high" nodes: wherever the spline is near its
        peak elevation (the reel-in arc), the depower-shifted winch offset must
        command reel-in (offset < 0). A positive offset there demands reel-out
        at high elevation, which the quasi-steady trim cannot hold.
    """
    path = reelout_config.get("path_parameters", {})
    radial = reelout_config.get("radial_parameters", {})
    sim = reelout_config.get("sim_parameters", {})
    problems = []

    if reelout_config.get("pattern_type") != "spline_periodic":
        problems.append(
            f"pattern_type is {reelout_config.get('pattern_type')!r}, "
            "expected 'spline_periodic'"
        )
    required = (
        (
            "radial_parameters",
            radial,
            (
                "slope_winch_ro",
                "offset_winch_ro",
                "winch_offset_depower_gain",
                "winch_depower_ref",
            ),
        ),
        (
            "sim_parameters",
            sim,
            (
                "n_points",
                "input_depower_profile",
                "optimize_depower_profile",
                "opti_limits_override",
                "require_full_trajectory",
            ),
        ),
    )
    for section_name, section, keys in required:
        for key in keys:
            if key not in section:
                problems.append(f"{section_name}.{key} is missing")
    if "optimize_depower_profile" in sim and not sim["optimize_depower_profile"]:
        problems.append("sim_parameters.optimize_depower_profile must be true")

    profile = np.asarray(sim.get("input_depower_profile", []), dtype=float)
    n_points = int(sim.get("n_points", 0))
    if profile.size and profile.size != n_points + 1:
        problems.append(
            f"input_depower_profile has {profile.size} entries, "
            f"expected n_points + 1 = {n_points + 1}"
        )

    lo, hi = DEPOWER_BAND
    band_tol = 0.02 * max(1.0, abs(hi - lo) / 0.4)  # 0.02 m on the V3 band
    if profile.size and (
        profile.min() < lo - band_tol or profile.max() > hi + band_tol
    ):
        problems.append(
            f"input_depower_profile spans [{profile.min():.3f}, {profile.max():.3f}], "
            f"outside the powered..depowered band [{lo:.2f}, {hi:.2f}]"
        )
    scalar_dep = sim.get("input_depower")
    if (
        profile.size
        and scalar_dep is not None
        and abs(float(scalar_dep) - profile[0]) > 1e-6
    ):
        problems.append(
            f"sim_parameters.input_depower ({float(scalar_dep):.3f}) != "
            f"input_depower_profile[0] ({profile[0]:.3f})"
        )

    if not problems:
        M = int(path["M"])
        pattern = PeriodicBSpline(
            M=M,
            C_phi=np.asarray(path["C_phi"], dtype=float).reshape((M, 1)),
            C_beta=np.asarray(path["C_beta"], dtype=float).reshape((M, 1)),
            s_init=float(path.get("s_init", 0.0)),
            s_final=float(path.get("s_final", 1.0)),
            downloops=bool(path.get("downloops", True)),
        )
        r0 = float(path["r0"])
        s_nodes = np.linspace(0.0, 1.0, n_points + 1)
        elevation = np.array([float(pattern.elevation(r0, s)) for s in s_nodes])

        # Zero-force reeling speed per node (Winch.tension_curve offset). Only
        # the very top of the cycle (top 10% of the elevation range) must be
        # depowered enough to reel in: a positive offset there means the kite
        # is still powered while parked high. A wider band false-positives
        # when the reel-in peak sits barely above the figure-eight tops.
        offset = float(radial["offset_winch_ro"]) + float(
            radial["winch_offset_depower_gain"]
        ) * (profile - float(radial["winch_depower_ref"]))
        el_thresh = elevation.min() + 0.9 * (elevation.max() - elevation.min())
        bad = (elevation >= el_thresh) & (offset > 0.0)
        if bad.any():
            s_bad = s_nodes[bad]
            problems.append(
                f"depower profile is misaligned with the reel-in arc: at "
                f"{int(bad.sum())} node(s) (s in [{s_bad.min():.2f}, {s_bad.max():.2f}]) "
                "the path is near peak elevation but the winch offset still "
                "commands reel-out (kite powered while parked high)"
            )

    if problems:
        bullets = "\n  - ".join(problems)
        raise ValueError(
            f"Full-cycle config failed validation:\n  - {bullets}\n"
            "The config must come straight from fit_periodic_cycle_config.py -- "
            "do not hand-edit the YAML or mix path/depower/winch data from "
            "different sources."
        )


def build_start_state(reelout_config):
    """Initial state of the cycle at the spline radius r0, and the first
    node's solver guess (a reel-out state at the maximum tension). The one
    start state of the seed march (fit_periodic_cycle_config) and the
    optimizer's baseline. (A reel-in guess at a depowered first node was
    tried on 2026-10-29: it converged to another root and broke the LEI-V3
    seed, so the march keeps this one.)"""
    sim = reelout_config.get("sim_parameters", {})
    profile = sim.get("input_depower_profile")
    depower0 = (
        float(profile[0])
        if profile
        else float(sim.get("input_depower", DEPOWER_BAND[0]))
    )
    return {
        "t": 0,
        "s": 0,
        "s_dot": 2,
        "input_steering": 0,
        "tension_tether_ground": reelout_config["radial_parameters"][
            "max_tether_force"
        ],
        "speed_radial": 1.0,
        "distance_radial": reelout_config["path_parameters"]["r0"],
        "input_depower": depower0,
    }


def _downsample_n_points(reelout_config, n_points):
    """Resample the sim grid (and depower profile) to ``n_points`` nodes.

    Only the per-node grid changes -- the spline shape (M control points) is
    independent of n_points -- so this is a clean conditioning lever.
    """
    sim = reelout_config["sim_parameters"]
    old = sim.get("input_depower_profile")
    sim["n_points"] = int(n_points)
    if old is not None and len(old) > 1:
        old = np.asarray(old, dtype=float)
        s_old = np.linspace(0.0, 1.0, len(old))
        s_new = np.linspace(0.0, 1.0, int(n_points) + 1)
        sim["input_depower_profile"] = np.interp(s_new, s_old, old).tolist()
        sim["input_depower"] = float(sim["input_depower_profile"][0])
    print(f"Downsampled cycle to n_points={int(n_points)}")


def _resample_trajectory(trajectory, sim):
    """A stored trajectory on this run's node grid (``sim['n_points']`` nodes
    from ``start_angle`` to ``end_angle``, the seam excluded), each column
    interpolated in s -- the coarse pass of --coarse-points hands its optimum
    to the full grid this way. Unchanged when it is already on the grid."""
    n = int(sim["n_points"])
    s_old = np.asarray(trajectory.get("s", []), dtype=float)
    if len(s_old) == n or len(s_old) < 2:
        return trajectory
    start, end = float(sim["start_angle"]), float(sim["end_angle"])
    s_new = start + (end - start) * np.arange(n) / n
    # a closed cycle interpolates across the seam (towards node 0), an open one clamps
    period = {"period": end - start} if sim.get("periodic_wrap") else {}
    print(f"Warm start resampled from {len(s_old)} to {n} nodes")
    out = {name: np.interp(s_new, s_old, np.asarray(values, dtype=float), **period)
           for name, values in trajectory.items() if len(values) == len(s_old)}
    out["s"] = s_new
    return out


def _closure_residual(result, periodic=False, time_scheme="euler"):
    """Radial closure (m) from the optimizer's own trajectory, or None.

    ``periodic`` (periodic_wrap): the seam interval's continuity residual in
    the NLP's own time scheme (euler: r[N-1] + v_r[N-1] dt; bdf2: the
    trapezoidal step and increment, awetrim.utils.time_scheme) minus r[0];
    otherwise |r[-1] - r[0]|.
    """
    traj = getattr(result, "optimized_trajectory", {}) or {}
    r = traj.get("distance_radial")
    if r is None or len(r) < 2:
        return None
    if periodic and {"s", "s_dot", "speed_radial"} <= set(traj):
        ds = 1.0 - float(traj["s"][-1])
        sd, v = traj["s_dot"], traj["speed_radial"]
        if time_scheme == "bdf2":
            dt = trapezoid_step(ds, float(sd[-1]), float(sd[0]))
            dr = trapezoid_increment(dt, float(v[-1]), float(v[0]))
        else:
            dr = float(v[-1]) * left_step(ds, max(float(sd[-1]), 1e-9))
        return float(abs(r[-1] + dr - r[0]))
    return float(abs(r[-1] - r[0]))


def _pattern_signature(path_params, n_samples=720):
    """Topology signature of the spline path: number of prominent azimuth turns.

    Each figure-eight loop swings the azimuth once to each side (~2 turns per
    loop), the reel-in bow adds 1-2 more. The count is invariant to smooth
    reshaping but changes as soon as the optimizer merges or destroys loops --
    exactly the failure mode the per-stage ``step_bounds`` trust region guards
    against. Compared against the seed after every stage pass; a mismatch is
    reported loudly even when the solve itself "succeeded".
    """
    M = int(path_params["M"])
    pattern = PeriodicBSpline(
        M=M,
        C_phi=np.asarray(path_params["C_phi"], dtype=float).reshape((M, 1)),
        C_beta=np.asarray(path_params["C_beta"], dtype=float).reshape((M, 1)),
        s_init=float(path_params.get("s_init", 0.0)),
        s_final=float(path_params.get("s_final", 1.0)),
        downloops=bool(path_params.get("downloops", True)),
    )
    r0 = float(path_params["r0"])
    s = np.linspace(0.0, 1.0, n_samples, endpoint=False)
    phi = np.array([float(pattern.azimuth(r0, si)) for si in s])
    # Relative threshold ignores small reshaping; the 0.05 rad absolute floor
    # stops collapsed wiggles (loops shrunk to un-flyable amplitude) from
    # still counting as turns.
    prominence = max(0.2 * float(phi.max() - phi.min()), 0.05)
    pad = n_samples // 4  # wrap-pad so turns at the s=0 seam count exactly once
    turns = 0
    for signal in (phi, -phi):
        wrapped = np.r_[signal[-pad:], signal, signal[:pad]]
        peaks, _ = find_peaks(wrapped, prominence=prominence)
        turns += int(np.sum((peaks >= pad) & (peaks < pad + n_samples)))
    return turns


def _param_snapshot(pattern_config, names):
    """Current values of the named parameters (any config section), as arrays."""
    snap = {}
    for name in names:
        for section in ("path_parameters", "radial_parameters", "sim_parameters"):
            if name in pattern_config.get(section, {}):
                snap[name] = np.asarray(pattern_config[section][name], dtype=float)
                break
    return snap


def _expand_step_bounds(step_bounds, path_parameters):
    """Expand scalar spline deltas into per-control-point trust-region vectors.

    Non-spline entries (r0, winch parameters) pass through unchanged. For
    C_phi / C_beta the scalar becomes a per-point vector: the base delta on
    the figure-eight control points and ``REELIN_STEP_SCALE`` x base on points
    whose support overlaps the reel-in arc (top ``REELIN_ELEVATION_FRACTION``
    of the current elevation range).
    """
    expanded = dict(step_bounds)
    spline_names = [n for n in ("C_phi", "C_beta") if n in step_bounds]
    if not spline_names or REELIN_STEP_SCALE == 1.0:
        return expanded
    mask = reelin_control_point_mask(
        path_parameters, elevation_fraction=REELIN_ELEVATION_FRACTION
    )
    for name in spline_names:
        base = float(step_bounds[name])
        expanded[name] = np.where(mask, REELIN_STEP_SCALE * base, base).tolist()
    print(
        f"Trust region: {int(mask.sum())}/{mask.size} spline control points on "
        f"the reel-in arc get {REELIN_STEP_SCALE:g}x the step bound"
    )
    return expanded


def _step_box_active(before, after, step_bounds):
    """True when any parameter moved to (>= 98% of) its per-solve box edge
    (awetrim.timeseries.phase.step_box_active; per-element deltas from
    ``_expand_step_bounds``, zero deltas ignored)."""
    return step_box_active(before, after, step_bounds)


def main(
    run_plots: bool = False,
    optimize: bool = True,
    kite: str = None,
    config_path=None,
    run_tag: str = "",
    skip_validation: bool = False,
    stage_repeats: int = None,
    step_scale: float = None,
    wind_speed: float = None,
    wind_height: float = None,
    warm_start_csv=None,
    wind_config: str = None,
    winch_inertia_mass: float = None,
    winch_friction=None,
    winch_drivetrain: bool = False,
    dynamic: bool = False,
    tether_inertia: bool = True,
    n_points: int = None,
) -> int:
    """``config_path`` overrides the kite's seed YAML (a variant such as the
    same seed with ``winch_mode: force_law``); ``run_tag`` is appended to the
    output filenames so variants do not overwrite each other. ``wind_speed``
    (m/s at ``wind_height``, default the profile's reference height) replaces
    the profile wind's amplitude, keeping its law (``cycle_kites.override_wind``);
    the seed must have been generated at the same wind. ``wind_config`` (a
    YAML wind block, ``cycle_kites.load_wind_config``) replaces the profile's
    wind before that override.

    ``warm_start_csv`` is a previous run's optimized trajectory
    (``full_cycle_optimized_<tag>.csv``, on the same node grid as the config):
    stage 1 is seeded from it instead of from a forward simulation, which a
    model that cannot march the start shape (another ROM, say) would fail.
    The baseline is then simulated only for the comparison, and skipped if it
    cannot be trimmed."""
    global WIND_CONFIG, QUASI_STEADY, N_POINTS
    # Node-grid resolution (resolution studies, the --coarse-points pass):
    # resample the cycle's grid (the spline shape is independent of it); a
    # warm-start trajectory on another grid is interpolated onto it. Set on
    # every call, so a second main() in one process does not inherit the first's.
    N_POINTS = int(n_points) if n_points else None
    configure_kite(kite or KITE_DIR)
    if wind_config:
        WIND_CONFIG = load_wind_config(wind_config)
    QUASI_STEADY = not dynamic
    if dynamic:
        print("Dynamic model: kite accelerations (implicit Euler) in the march and the NLP")
    WIND_CONFIG = override_wind(WIND_CONFIG, wind_speed, wind_height)
    print(f"Kite: {KITE_NAME} ({KITE_CONFIG_PATH}); wind: {describe_wind()}")
    if stage_repeats:
        # Re-centred passes of the main stage (the old sequential scheme):
        # the remedy when a run ends with the backstop step box active.
        STAGES[1]["repeat"] = int(stage_repeats)
    if step_scale:
        # The power stages' backstop boxes scaled by one factor (1 = the boxes
        # above); a stage with scale_box False (the closing stage) keeps its own.
        for stage in STAGES:
            if not stage.get("scale_box", True):
                continue
            stage["step_bounds"] = {
                k: float(v) * float(step_scale)
                for k, v in (stage.get("step_bounds") or {}).items()
            }
        print(f"Step boxes scaled x{float(step_scale):g}")
    reelout_config = load_reelout_config(config_path or CYCLE_CONFIG_PATH)
    if skip_validation:
        # Continuation from an OPTIMIZED config (a previous run's output): its
        # depower profile legitimately spans the KCU hardware range rather
        # than the seed's identified band, so the seed validator would refuse
        # it.
        print("Seed validation skipped (continuation run)")
    else:
        validate_full_cycle_config(reelout_config)
    sim = reelout_config["sim_parameters"]
    # A seed or continuation file written before the key: every cycle has this rate.
    sim.setdefault("slow_path_rate", True)
    if winch_inertia_mass is not None:
        # Drum + drive inertia reflected onto the tether [kg] (opti_phase).
        sim["winch_inertia_mass"] = float(winch_inertia_mass)
    if winch_friction is not None:
        # Drivetrain friction in the objective: True = system.yaml drums[0],
        # or a tether-referred (F_c [N], c_v [N s/m]) pair.
        sim["winch_friction"] = winch_friction
    if winch_drivetrain:
        # The system file's drum inertia + friction, in the march and the NLP.
        sim["winch_drivetrain"] = True
    if N_POINTS:
        _downsample_n_points(reelout_config, N_POINTS)

    # Snapshot of the seed as actually solved (post-downsample): the staged
    # optimization writes the optimized shape/winch back into reelout_config
    # in place, and the comparison plots need the original to compare against.
    seed_config = copy.deepcopy(reelout_config)

    # The ROM's tether block (lumped, with its inertia by default);
    # --no-tether-inertia drops the tether's inertial load at the kite.
    system_model = create_system_model_from_yaml(
        yaml_path=KITE_CONFIG_PATH,
        tether_config=(
            None if tether_inertia else {"model": "rigid_lumped", "inertia": False}
        ),
    )
    system_model.wind = build_wind_model(**WIND_CONFIG)

    assist = 1.0
    cycle = Phase(
        system_model=system_model,
        pattern_config=reelout_config,
        start_state=build_start_state(reelout_config),
        quasi_steady=QUASI_STEADY,
    )
    if warm_start_csv:
        import pandas as pd

        trajectory = pd.read_csv(warm_start_csv)
        cycle._warm_start_trajectory = _resample_trajectory(
            {name: trajectory[name].to_numpy(dtype=float) for name in trajectory.columns},
            reelout_config["sim_parameters"],
        )
        cycle._warm_start_source = "optimum"
        print(f"Warm start: stage 1 seeded from {warm_start_csv} ({len(trajectory)} nodes)")

    print("Simulating the experimental-fit cycle (baseline)...")
    if warm_start_csv:
        try:
            # a stored warm start (--warm-start) seeds stage 1, not this march
            baseline_phase, _ = cycle.run_simulation(
                run_plots=run_plots, phase_sim=True, seed_warm_start=False
            )
        except RuntimeError as exc:
            print(f"Baseline forward simulation failed: {exc}")
            print("Continuing from the warm-start trajectory, without a baseline.")
            baseline_phase = None
    else:
        # seed_warm_start: this march doubles as the NLP's warm start, so
        # stage 1 does not re-run the identical (~0.35 s/node) simulation;
        # later passes seed from the previous optimum and skip it as well.
        # A seed the kite cannot turn through is marched at an assisted roll
        # gain (march_with_roll_assist) and continued back to 1 below.
        cycle, baseline_phase, assist, failure = march_with_roll_assist(
            system_model, reelout_config, run_plots=run_plots, seed_warm_start=True
        )
        if assist is None:
            # require_full_trajectory: a node of the fitted cycle failed to trim.
            print(f"Baseline forward simulation failed: {failure}")
            print(
                "The fitted initial guess cannot be trimmed over the whole cycle, "
                f"not even at {max(roll_assist.ROLL_ASSIST_SCALES):g} x the roll gain; re-tune "
                "fit_periodic_cycle_config.py and regenerate the config before "
                "optimizing."
            )
            return 1
    if baseline_phase is not None:
        if assist != 1.0:
            print(f"(baseline marched at {assist:g} x the roll gain)")
        print(baseline_phase.energy_metrics())

    if optimize:
        seed_wrap = bool(sim.get("periodic_wrap", False))
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        tag = wind_tag() + (f"_{run_tag}" if run_tag else "")
        seed_signature = _pattern_signature(reelout_config["path_parameters"])
        print(
            f"Seed pattern signature: {seed_signature} azimuth turns "
            "(~2 per figure-eight loop + reel-in bow)"
        )
        result = None
        aborted = False
        stages = _roll_assist_stages(STAGES, assist)
        i = -1
        while i + 1 < len(stages):
            i += 1
            stage = stages[i]
            retry = False
            cycle.system_model.kite.gain_roll_scale = float(stage["roll_scale"])
            # Soft closure (penalty) until the final hard-close stage. Each stage
            # warm-starts from the previous one: run_simulation_opti writes the
            # optimized shape/winch back into the config and advances start_state.
            sim["radial_closure_weight"] = float(stage["closure_weight"])
            # A periodic_wrap seed closes the cycle through the seam interval
            # (last node -> node 0, with its continuity, rates, AoA, energy
            # and time); the one-sided radius pin would leave the last node
            # unconstrained and the seam uncounted, and the two exclude each
            # other. A stage may ask for the pin instead ("periodic": False);
            # older seeds without the wrap keep the pin throughout.
            wrap = seed_wrap and bool(stage.get("periodic", True))
            sim["periodic_wrap"] = wrap
            sim["close_radial_cycle"] = bool(stage["hard_close"]) and not wrap
            anchor = float(stage.get("depower_anchor", 0.0)) if skip_validation else 0.0
            sim["fix_depower_profile"] = bool(stage.get("fix_depower", False)) and not anchor
            sim["depower_anchor_weight"] = anchor
            step_bounds = stage.get("step_bounds") or {}
            target = stage.get("target", "power")
            # The power stages select the smoothest of the near-equal-power
            # trajectories (the ridge is flat in shape and steering; without
            # it the reel-in of a kite steering at its band zigzags).
            sim["reg_fraction"] = REG_FRACTION if target == "power" else 0.0
            # IPOPT's acceptable-point objective change (Phase.run_opti; 1e-4 default)
            sim["acceptable_obj_change_tol"] = float(stage.get("acceptable_obj_change_tol", 1e-4))
            repeats = int(stage.get("repeat", 1))
            trust_weight = float(stage.get("trust_region_weight", 0.0))
            warm_start = bool(stage.get("warm_start", False))
            # Power and polish stages re-run themselves from where they stopped:
            # a stall near feasibility resumes from its last iterate, and an
            # optimum on the step box's edge is solved again with the box
            # re-centred on it (_recentre_or_resume). The closing stage keeps
            # its single tight solve.
            recentre = bool(step_bounds) and stage.get("scale_box", True) is not False
            rep, previous_power = 0, None
            while rep < repeats:
                rep_tag = f" (pass {rep + 1}/{repeats})" if repeats > 1 else ""
                print(
                    f"\n=== Stage {i + 1}/{len(stages)}: {stage['label']}{rep_tag} ===\n"
                    f"    params={stage['params']} + depower profile  "
                    f"(target={target}, trust_region_weight={trust_weight:g})"
                )
                # Rebuilt every pass: the reel-in arc drifts in s as the shape
                # moves, and the box re-centers on the current config anyway.
                expanded_bounds = _expand_step_bounds(
                    step_bounds, cycle.pattern_config["path_parameters"]
                )
                sim["param_step_bound"] = expanded_bounds
                before = _param_snapshot(cycle.pattern_config, step_bounds)
                try:
                    stage_result = cycle.run_simulation_opti(
                        optimization_params=stage["params"],
                        target=target,
                        max_iter=int(stage.get("max_iter") or MAX_ITER),
                        trust_region_weight=trust_weight,
                        warm_start=warm_start,
                        resume_tolerance=RESUME_TOLERANCE if recentre else None,
                    )
                except RuntimeError as exc:
                    if _retry_continuation(stages, i):
                        retry = True
                        break
                    # require_full_trajectory: the warm-start re-simulation of the
                    # previous stage's config failed a node mid-cycle. A padded
                    # seed is worse than stopping, so abort the staging here.
                    print(
                        f"Stage {i + 1} aborted (warm-start simulation failed): {exc}"
                    )
                    print("Stopping staged optimization; keeping the last good result.")
                    aborted = True
                    break
                if stage_result is None:
                    if recentre and _resume_stalled(stage, cycle):
                        repeats += 1
                        rep += 1
                        continue
                    if _retry_continuation(stages, i) or _retry_halved_box(stages, i):
                        retry = True
                        break
                    print(f"Stage {i + 1} failed; stopping (keeping last good config).")
                    aborted = True
                    break
                result = stage_result
                res = _closure_residual(
                    result,
                    bool(sim.get("periodic_wrap", False)),
                    sim.get("time_scheme", "euler" if QUASI_STEADY else "bdf2"),
                )
                print(
                    f"Stage {i + 1}{rep_tag} done. radial closure = "
                    f"{'n/a' if res is None else f'{res:.2f} m'}"
                )
                signature = _pattern_signature(cycle.pattern_config["path_parameters"])
                print(
                    f"    pattern signature: {signature} azimuth turns "
                    f"(seed {seed_signature})"
                    + (
                        ""
                        if signature == seed_signature
                        else "  ** WARNING: pattern topology changed vs seed **"
                    )
                )
                # The step box is a backstop that should never bind: the
                # proximal anchor is what shapes the step. If a parameter
                # still lands on the box edge, the true optimum lies beyond
                # it -- widen the box (or add a "repeat" pass, which
                # re-centers the box like the old sequential scheme did).
                after = _param_snapshot(cycle.pattern_config, step_bounds)
                box_active = _step_box_active(before, after, expanded_bounds)
                power = _trajectory_mean_power(result.optimized_trajectory)
                if power is not None:
                    print(f"    mean power {power / 1e3:.3f} kW")
                last = rep == repeats - 1
                if box_active and last and recentre and _recentre_box(stage, previous_power, power):
                    repeats += 1
                    if "step_bounds_full" in stage:
                        # a halved box got the stage through; the re-centred passes
                        # move from a converged point and get the full box back
                        stage["step_bounds"] = step_bounds = stage.pop("step_bounds_full")
                        print(f"    full step box restored: {step_bounds}")
                elif box_active and last:
                    print(
                        "    ** WARNING: backstop step box active at the stage "
                        "optimum -- widen step_bounds or add a repeat pass **"
                    )
                if rep < repeats - 1 and step_bounds and not box_active:
                    print(
                        "    trust region inactive (optimum interior) -- "
                        "skipping remaining passes"
                    )
                    break
                previous_power = power
                rep += 1
            if retry:
                i -= 1  # the inserted midpoint step runs next
                continue
            if aborted:
                break
            result.save_config_to_yaml(
                RESULTS_DIR / f"full_cycle_optimized_{tag}_stage{i + 1}.yaml"
            )
        cycle.system_model.kite.gain_roll_scale = 1.0
        if assist != 1.0:
            path = " -> ".join(f"{s['roll_scale']:g}" for s in stages
                               if s["roll_scale"] != 1.0 or s is stages[-1])
            print(f"Roll-gain continuation: {path} (the result is at scale 1)"
                  if not aborted else
                  f"Roll-gain continuation aborted at scale {stage['roll_scale']:g}")
            if aborted and stage["roll_scale"] != 1.0:
                return 1

        if result is None:
            return 1
        result.save_config_to_yaml(RESULTS_DIR / f"full_cycle_optimized_{tag}.yaml")
        result.save_trajectory_csv(RESULTS_DIR / f"full_cycle_optimized_{tag}.csv")

        # Seed-vs-optimized comparison figures, built straight from the NLP's
        # own per-node trajectory (no re-simulation). Best-effort: a plotting
        # failure must not mask a successful optimization.
        try:
            from cycle_comparison_plots import save_cycle_comparison_plots

            save_cycle_comparison_plots(
                baseline_phase=baseline_phase,
                result=result,
                seed_config=seed_config,
                optimized_config=cycle.pattern_config,
                out_dir=RESULTS_DIR,
                tag=tag,
                show=run_plots,
            )
        except Exception as exc:
            print(f"(cycle comparison plots unavailable: {exc})")

    if optimize and aborted:
        # The last good stage's result is written, but it is not the optimum:
        # a caller (run_cycle_power_curve.py) must not count it as converged.
        print("Full cycle optimization INCOMPLETE: a stage failed; the files hold "
              "the last good stage's result")
        return 2
    print("Full cycle optimization complete")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot", action="store_true", help="Show overview plots")
    parser.add_argument(
        "--no-optimize",
        action="store_true",
        help="Only simulate the experimental-fit cycle (skip the NLP)",
    )
    parser.add_argument(
        "--kite",
        default=DEFAULT_KITE,
        help="Kite folder under data/ (or a path to one) holding system.yaml, "
        "the ROM config it names and cycle_profile.yaml (see cycle_kites.py); "
        f"default {DEFAULT_KITE}",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Seed YAML to optimize instead of the kite's default "
        "(cycle_configs/full_cycle_periodic_from_exp.yaml)",
    )
    parser.add_argument(
        "--tag",
        default="",
        help="Suffix appended to the output filenames (variants of one wind)",
    )
    parser.add_argument(
        "--stage-repeats",
        type=int,
        default=None,
        help="Number of re-centred passes of the main stage (default 1); "
        "use 2-3 when a run ends with the backstop step box active",
    )
    parser.add_argument(
        "--continue-from",
        dest="continue_from",
        default=None,
        help="Optimized config YAML of a previous run to continue from "
        "(re-runs the stages from it, seed validation skipped) -- use when "
        "the previous run ended with the backstop step box active",
    )
    parser.add_argument(
        "--warm-start",
        dest="warm_start",
        default=None,
        help="Optimized trajectory CSV of a previous run on the same node grid "
        "(full_cycle_optimized_<tag>.csv): seeds stage 1 instead of a forward "
        "simulation, e.g. to optimize with another ROM from that optimum "
        "(pair it with --continue-from that run's optimized YAML)",
    )
    parser.add_argument(
        "--step-scale",
        type=float,
        default=None,
        help="Scale the power and polish stages' backstop step boxes (C_phi, "
        "C_beta, r0) by this factor (default 1; the closing stage keeps its box)",
    )
    parser.add_argument(
        "--wind-speed",
        type=float,
        default=None,
        help="Wind speed (m/s) at --wind-height replacing the profile's "
        "amplitude (law and z0 kept); the seed must be generated at the same wind",
    )
    parser.add_argument(
        "--wind-height",
        type=float,
        default=None,
        help="Height (m) of --wind-speed (default the profile's height_ref)",
    )
    parser.add_argument(
        "--wind-config",
        dest="wind_config",
        default=None,
        help="YAML wind block replacing the profile's wind: (e.g. a jet or a "
        "veering wind); the seed must be generated at the same wind",
    )
    parser.add_argument(
        "--winch-inertia-mass",
        type=float,
        default=None,
        help="Winch inertia as an equivalent tether mass J/r_drum^2 (kg); the "
        "drive force T - m dv_r/dt is then held below max_tether_force",
    )
    parser.add_argument(
        "--winch-friction",
        default=None,
        help="Charge drivetrain friction in the objective: 'system' for "
        "system.yaml drums[0], or 'F_c,c_v' tether-referred (N, N s/m)",
    )
    parser.add_argument(
        "--winch-drivetrain",
        action="store_true",
        help="Fly the system file's winch drivetrain (drum inertia + friction, "
        "system.yaml drums[0]) in the forward simulation and the NLP",
    )
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="Kite (and tether) accelerations in the forward march and the NLP "
        "(implicit, sim_parameters.time_scheme: bdf2 by default) instead of quasi-steady",
    )
    parser.add_argument(
        "--no-tether-inertia",
        dest="tether_inertia",
        action="store_false",
        help="Drop the lumped tether's inertia at the kite (on by default: "
        "centrifugal, Coriolis and turn terms; with --dynamic also its accelerations)",
    )
    parser.add_argument(
        "--n-points",
        type=int,
        default=None,
        help="Resample the cycle to this many nodes (resolution study); a "
        "--warm-start on another grid is interpolated onto it",
    )
    parser.add_argument(
        "--coarse-points",
        type=int,
        default=None,
        help="Solve the stages first on this many nodes, then on the full grid "
        "from that result: the coarse pass finds the pattern (cheap iterations), "
        "the fine pass refines it. Default: 2 x M (2 nodes per spline control "
        "interval) for a cold solve, off with --continue-from / --warm-start; 0 "
        "switches it off. LEI-V3, turn law: 5.5 m/s failed cold on the full grid "
        "alone and converged with it in 312 s",
    )
    parser.add_argument(
        "--stage-max-iter",
        type=int,
        default=None,
        help="IPOPT iteration cap of the power and polish stages (default "
        "MAX_ITER); a stage stalled at the cap resumes from its last iterate "
        "(--max-resumes), so a small cap restarts a crawl sooner",
    )
    parser.add_argument(
        "--max-resumes",
        type=int,
        default=None,
        help=f"Resumes of a stalled power or polish stage (default {MAX_RESUMES})",
    )
    parser.add_argument(
        "--shape-obj-tol",
        type=float,
        default=None,
        help="IPOPT acceptable_obj_change_tol of the shape stage (default 1e-4)",
    )
    parser.add_argument(
        "--polish-obj-tol",
        type=float,
        default=None,
        help="IPOPT acceptable_obj_change_tol of the polish stage (default 1e-4): "
        "a larger value ends the polish once it only adds fractions of a percent",
    )
    args = parser.parse_args()
    if args.shape_obj_tol:
        STAGES[1]["acceptable_obj_change_tol"] = float(args.shape_obj_tol)
    if args.polish_obj_tol:
        STAGES[2]["acceptable_obj_change_tol"] = float(args.polish_obj_tol)
    if args.stage_max_iter:
        for stage in STAGES:
            if stage.get("scale_box", True) is not False:
                stage["max_iter"] = int(args.stage_max_iter)
        print(f"Power and polish stages capped at {args.stage_max_iter} iterations")
    if args.max_resumes is not None:
        MAX_RESUMES = int(args.max_resumes)
    if args.winch_friction is not None:
        args.winch_friction = (
            True
            if args.winch_friction == "system"
            else [float(x) for x in args.winch_friction.split(",")]
        )
    warm_start = args.warm_start
    if warm_start is None and args.continue_from:
        # The continued run's trajectory seeds stage 1 directly: re-marching
        # its optimal path at another wind with the seed's force law is what
        # fails (no root at a node, both steering laws), while the trajectory
        # itself is a complete, feasible starting point on the same grid.
        sibling = Path(args.continue_from).with_suffix(".csv")
        if sibling.is_file():
            warm_start = str(sibling)
            print(f"--continue-from: its trajectory {sibling.name} seeds stage 1")
    common = dict(
        kite=args.kite,
        stage_repeats=args.stage_repeats,
        wind_speed=args.wind_speed,
        wind_height=args.wind_height,
        wind_config=args.wind_config,
        winch_inertia_mass=args.winch_inertia_mass,
        winch_friction=args.winch_friction,
        winch_drivetrain=args.winch_drivetrain,
        dynamic=args.dynamic,
        tether_inertia=args.tether_inertia,
    )
    config_path = args.continue_from or args.config
    skip_validation = args.continue_from is not None
    if args.coarse_points is None and not args.no_optimize:
        # Default: a cold solve takes the coarse pass, a continued one does not
        # (it starts next to its optimum; the coarse round only adds work).
        # The coarse grid has 2 nodes per spline control interval.
        if args.continue_from or args.warm_start:
            args.coarse_points = 0
        else:
            configure_kite(args.kite or KITE_DIR)
            M = int(load_reelout_config(config_path or CYCLE_CONFIG_PATH)["path_parameters"]["M"])
            args.coarse_points = 2 * M
            print(f"Cold solve: coarse pass on {args.coarse_points} nodes (2 x M; --coarse-points 0 "
                  "switches it off)")
    if args.coarse_points and not args.no_optimize:
        # Coarse-to-fine: every stage first on a coarse grid (cheap iterations,
        # it finds the pattern), then the full grid continues from its optimum
        # and trajectory -- a refinement from a closed, trimmed point.
        coarse_tag = f"{args.tag}_coarse" if args.tag else "coarse"
        print(f"=== Coarse pass: {args.coarse_points} nodes ===")
        coarse_code = main(
            run_plots=False, optimize=True, config_path=config_path, run_tag=coarse_tag,
            skip_validation=skip_validation, step_scale=args.step_scale,
            warm_start_csv=warm_start, n_points=args.coarse_points, **common,
        )
        # appended, not with_suffix: the tag holds dots (wind_5.5at6m, z0_0.0002)
        coarse = f"{RESULTS_DIR / f'full_cycle_optimized_{wind_tag()}_{coarse_tag}'}"
        if coarse_code == 0 and Path(coarse + ".yaml").is_file():
            full_n = args.n_points or int(
                load_reelout_config(config_path or CYCLE_CONFIG_PATH)["sim_parameters"]["n_points"])
            print(f"=== Fine pass: {full_n} nodes, from the coarse optimum ===")
            config_path, skip_validation = coarse + ".yaml", True
            warm_start, n_points = coarse + ".csv", full_n
        else:
            print("Coarse pass did not converge; solving the full grid from the start")
            n_points = args.n_points
        # step_scale already scaled the module's stage boxes in the coarse pass
        step_scale = None
    else:
        n_points, step_scale = args.n_points, args.step_scale
    exit_code = main(
        run_plots=args.plot,
        optimize=not args.no_optimize,
        config_path=config_path,
        run_tag=args.tag,
        skip_validation=skip_validation,
        step_scale=step_scale,
        warm_start_csv=warm_start,
        n_points=n_points,
        **common,
    )
    if args.plot:
        import matplotlib.pyplot as plt

        plt.show()
    raise SystemExit(exit_code)

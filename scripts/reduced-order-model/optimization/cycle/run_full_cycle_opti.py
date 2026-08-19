"""Full pumping-cycle optimization as a single periodic spline + depower profile.

This is the "full opti" counterpart to ``run_cycle_simulation.py`` (which stitches
a reel-out ``Phase`` to a parametrized ``ReelinSimple``). Here the **entire**
pumping cycle is one ``spline_periodic`` ``Phase``:

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

The wind is a constant logarithmic profile, tunable here via ``WIND_CONFIG``.
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
    python scripts/reduced-order-model/optimization/cycle/run_full_cycle_opti.py [--plot]
"""

import argparse
import copy
from pathlib import Path

import numpy as np
import yaml
from scipy.signal import find_peaks

from awetrim.environment.wind_factory import create_wind_model
from awetrim.identification.controls import (
    ROM_DEPOWERED_INPUT_DEPOWER,
    ROM_POWERED_INPUT_DEPOWER,
)
from awetrim.kinematics.parametrized_patterns import (
    PeriodicBSpline,
    reelin_control_point_mask,
)
from awetrim.system.factory import create_system_model_from_yaml
from awetrim.timeseries.phase import Phase
from awetrim.utils.config_paths import (
    LEI_V3_CYCLE_CONFIG_DIR,
    LEI_V3_SYSTEM_CONFIG,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
KITE_CONFIG_PATH = LEI_V3_SYSTEM_CONFIG
CYCLE_CONFIG_PATH = LEI_V3_CYCLE_CONFIG_DIR / "full_cycle_periodic_from_exp.yaml"

RESULTS_DIR = (
    Path("results") / KITE_CONFIG_PATH.parent.name / "optimization" / "full_cycle"
)

# Tunable constant logarithmic wind.
WIND_CONFIG = {
    "speed_wind_at_100": 14.0,  # m/s at 100 m
    "z0": 0.03,  # roughness length (m)
    "model_type": "logarithmic",
}

# Staged solve, warm-started stage to stage.
#
# Closure must be a HARD equality for the power rounds, NOT a soft penalty. A
# penalty lives in the objective, so it does not register in inf_pr and the
# optimizer quietly opens the cycle: the power reward for reeling out forever
# beats any finite penalty (exterior penalty is exact only as w -> inf), the
# radius runs away, and the per-node trim blows up (objective -> -6, inf_pr
# explodes). The hard equality makes the runaway simply infeasible. (A soft
# closure round -- target "zero", closure_weight ~1e5, hard_close off -- is
# only needed when the seed itself is OPEN; fit_periodic_cycle_config.py now
# emits closed trim-feasible seeds, so no feasibility round is staged.)
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
# ``step_bounds`` remains only as a BACKSTOP box (hard +/-delta per solve):
# the main stage's box is sized to the old scheme's total 4-pass travel
# budget and should never be active -- the driver warns when it is. The
# polish keeps the old tight box; warm-started from the main optimum it
# should sit strictly inside it.
STAGES = [
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
        "warm_start": False,  # no anchor on the seed, just the trust region
        "trust_region_weight": 0.05,
        "step_bounds": {"C_phi": 0.15, "C_beta": 0.15, "r0": 50.0},
        "repeat": 1,  # old scheme: 4 passes, re-centered each time; new scheme: 1 pass
    },
    {
        "label": "polish | weak anchor",
        "params": ["C_phi", "C_beta", "r0"],
        "closure_weight": 0.0,
        "hard_close": True,
        "trust_region_weight": 0.005,
        "warm_start": True,  # tight-barrier IPOPT start from the main optimum
        "step_bounds": {"C_phi": 0.4, "C_beta": 0.4, "r0": 100.0},
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

# Downsample the loaded config to this many nodes before solving (None = keep the
# config's n_points). The depower profile + per-node trim + closure all scale with
# n_points, so this is the single biggest conditioning lever -- 150 nodes resolves
# the path fine and solves far more steadily than 400. The spline shape (M control
# points) is independent of n_points, so only the sim grid / depower profile change.
N_POINTS = None


def build_wind_model(speed_wind_at_100, z0, model_type):
    """Wind model with the reference speed given at 100 m (any analytic law)."""
    return create_wind_model(
        model_type,
        U_ref=speed_wind_at_100,
        z_ref=100.0,
        z0=z0,
        direction_wind=0,  # wind blowing along +x
    )


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

    lo, hi = ROM_POWERED_INPUT_DEPOWER, ROM_DEPOWERED_INPUT_DEPOWER
    band_tol = 0.02
    if profile.size and (
        profile.min() < lo - band_tol or profile.max() > hi + band_tol
    ):
        problems.append(
            f"input_depower_profile spans [{profile.min():.3f}, {profile.max():.3f}] m, "
            f"outside the powered..depowered band [{lo:.2f}, {hi:.2f}] m"
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
    """Initial state for the cycle, started at the spline radius r0 (reel-out)."""
    sim = reelout_config.get("sim_parameters", {})
    profile = sim.get("input_depower_profile")
    depower0 = float(profile[0]) if profile else float(sim.get("input_depower", 1.7))
    return {
        "t": 0,
        "s": 0,
        "s_dot": 2,
        "input_steering": 0,
        "tension_tether_ground": reelout_config["radial_parameters"][
            "max_tether_force"
        ],
        "speed_radial": 1.0,  # positive: cycle begins reeling out
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


def _closure_residual(result):
    """|r[-1] - r[0]| (m) from the optimizer's own trajectory, or None."""
    traj = getattr(result, "optimized_trajectory", {}) or {}
    r = traj.get("distance_radial")
    if r is None or len(r) < 2:
        return None
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
    """True when any parameter moved to (>= 98% of) its per-solve box edge.

    ``step_bounds`` values may be scalars or per-element vectors (see
    ``_expand_step_bounds``); elements with a zero delta are ignored.
    """
    for name, delta in step_bounds.items():
        if name not in before or name not in after:
            continue
        moved = np.abs(
            np.asarray(after[name], dtype=float).ravel()
            - np.asarray(before[name], dtype=float).ravel()
        )
        deltas = np.asarray(delta, dtype=float).ravel()
        if deltas.size == 1:
            deltas = np.full(moved.size, deltas[0])
        live = deltas > 0
        if np.any(moved[live] >= 0.98 * deltas[live]):
            return True
    return False


def main(run_plots: bool = False, optimize: bool = True) -> int:
    reelout_config = load_reelout_config(CYCLE_CONFIG_PATH)
    validate_full_cycle_config(reelout_config)
    sim = reelout_config["sim_parameters"]
    if N_POINTS:
        _downsample_n_points(reelout_config, N_POINTS)

    # Snapshot of the seed as actually solved (post-downsample): the staged
    # optimization writes the optimized shape/winch back into reelout_config
    # in place, and the comparison plots need the original to compare against.
    seed_config = copy.deepcopy(reelout_config)

    system_model = create_system_model_from_yaml(yaml_path=KITE_CONFIG_PATH)
    system_model.wind = build_wind_model(**WIND_CONFIG)

    cycle = Phase(
        system_model=system_model,
        pattern_config=reelout_config,
        start_state=build_start_state(reelout_config),
    )

    print("Simulating the experimental-fit cycle (baseline)...")
    try:
        baseline_phase, _ = cycle.run_simulation(run_plots=run_plots, phase_sim=True)
    except RuntimeError as exc:
        # require_full_trajectory: a node of the fitted cycle failed to trim.
        print(f"Baseline forward simulation failed: {exc}")
        print(
            "The fitted initial guess cannot be trimmed over the whole cycle; "
            "re-tune fit_periodic_cycle_config.py and regenerate the config "
            "before optimizing."
        )
        return 1
    print(baseline_phase.energy_metrics())

    if optimize:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        tag = f"wind_{WIND_CONFIG['speed_wind_at_100']:g}_z0_{WIND_CONFIG['z0']:g}"
        seed_signature = _pattern_signature(reelout_config["path_parameters"])
        print(
            f"Seed pattern signature: {seed_signature} azimuth turns "
            "(~2 per figure-eight loop + reel-in bow)"
        )
        result = None
        aborted = False
        for i, stage in enumerate(STAGES):
            # Soft closure (penalty) until the final hard-close stage. Each stage
            # warm-starts from the previous one: run_simulation_opti writes the
            # optimized shape/winch back into the config and advances start_state.
            sim["radial_closure_weight"] = float(stage["closure_weight"])
            sim["close_radial_cycle"] = bool(stage["hard_close"])
            step_bounds = stage.get("step_bounds") or {}
            target = stage.get("target", "power")
            repeats = int(stage.get("repeat", 1))
            trust_weight = float(stage.get("trust_region_weight", 0.0))
            warm_start = bool(stage.get("warm_start", False))
            for rep in range(repeats):
                rep_tag = f" (pass {rep + 1}/{repeats})" if repeats > 1 else ""
                print(
                    f"\n=== Stage {i + 1}/{len(STAGES)}: {stage['label']}{rep_tag} ===\n"
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
                        max_iter=MAX_ITER,
                        trust_region_weight=trust_weight,
                        warm_start=warm_start,
                    )
                except RuntimeError as exc:
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
                    print(f"Stage {i + 1} failed; stopping (keeping last good config).")
                    aborted = True
                    break
                result = stage_result
                res = _closure_residual(result)
                print(
                    f"Stage {i + 1}{rep_tag} done. closure |r[-1]-r[0]| = "
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
                if box_active and rep == repeats - 1:
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
            if aborted:
                break
            result.save_config_to_yaml(
                RESULTS_DIR / f"full_cycle_optimized_{tag}_stage{i + 1}.yaml"
            )

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
    args = parser.parse_args()
    exit_code = main(run_plots=args.plot, optimize=not args.no_optimize)
    if args.plot:
        import matplotlib.pyplot as plt

        plt.show()
    raise SystemExit(exit_code)

"""Full pumping cycle as ONE slanted "up-loop" figure eight.

One closed period of a periodic spline holds the whole cycle: the low lobe
(plus the two crossing legs) is the reel-out and the high arc is the reel-in.
The pattern is the one a colleague found to cut the steering reversals and
to give more power at MEDIUM wind than the multi-lobe cycle of
``run_full_cycle_opti.py`` -- so the steering-reversal count is a first-class
metric here, next to the mean cycle power.

Formulation (all in ``opti_phase``):

  * ``winch_mode: free_speed`` -- the reel speed is a rate-limited control;
  * ``periodic_wrap: true`` -- the seam interval node N-1 -> node 0 carries
    the same continuity / rate / AoA rows as every interior one, so the
    radius closes (start radius == end radius) through the continuity row
    and the period is free (the s-grid is one period, the node times are
    decisions);
  * objective: mean cycle power; depower per node inside the ROM's identified
    band; alpha ``stall_margin_deg`` below the stall; a dense
    ``min_turn_radius`` floor; the height band z = r sin(beta)
    (``min_height_m`` / ``max_height_m``, the node-wise ``height`` rows) as
    the binding vertical limit -- the ``C_beta`` spline hull is loose.

Seed: :func:`awetrim.kinematics.parametrized_patterns.slanted_eight_angles`
fitted to a periodic B-spline (``make_slanted_eight_bspline_path_parameters``),
faired to the turn-radius floor, with the depower window centred on the
high-lobe apex; the forward march with the kite's winch force law measures
the cycle duration (-> ``n_points``) and reports feasibility. The optimizer
then runs three warm-started stages: S0 fixes the path and frees ``r0``
(closes the cycle), S1 frees the shape inside a tight box (re-centred
passes while any step box is active, at most 6; every pass is saved), S2
polishes with a weak anchor.

Wind: NEVER the profile's top-level ``wind:``. The CLI flags
(``--wind-law --wind-speed --wind-height --z0``) win over the
``uploop_eight.wind`` block of the kite's cycle_profile.yaml; without either
the script stops. The solved wind is printed and tagged into every output.

Outputs in ``results/<kite>/optimization/uploop_eight/``: the seed YAML +
path figure, ``uploop_eight_optimized_<wind tag>[_stageK].yaml/.csv``, the
summary figure and the standard cycle-comparison figures.

Usage:
    python scripts/reduced-order-model/optimization/studies/run_uploop_eight_opti.py
        [--kite KITE] [--wind-speed 10 --wind-height 100 --z0 0.03]
        [--no-optimize] [--plot] [--yes] [--baseline-seed] [--compare]
        [--baseline-csv CSV] [--scale-lobes 1.0]
"""

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import yaml

# The full-cycle helpers live next door in ../cycle/.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cycle"))

import cycle_comparison_plots as ccp
import fit_periodic_cycle_config as fpcc  # configures DEFAULT_KITE on import
import run_full_cycle_opti as rfco
from cycle_kites import DEFAULT_KITE, SEED_FILENAME, resolve_kite, wind_tag

from awetrim.kinematics.parametrized_patterns import (
    PeriodicBSpline,
    _great_circle_deviation,
    count_self_crossings,
    eight_crossing_s,
    fair_periodic_spline_to_curvature_limit,
    lobe_signed_areas,
    make_slanted_eight_bspline_path_parameters,
    slanted_eight_landmarks,
    symmetrize_periodic_path,
)
from awetrim.system.factory import create_system_model_from_yaml
from awetrim.timeseries.phase import Phase
from awetrim.utils.control_metrics import (
    STEERING_DEADBAND,
    count_steering_reversals,
    steering_reversal_pairs,
)
from awetrim.utils.defaults import DEFAULT_OPTI_LIMITS

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Wind used when a CLI wind flag is given without the others.
CLI_WIND_DEFAULTS = {"model_type": "logarithmic", "height_ref": 100.0, "z0": 0.03}

# Shape / window defaults of the uploop_eight block (angles in rad); the
# kite's block overrides them key by key.
SHAPE_DEFAULTS = {
    "az_low": np.radians(-5.0),
    "el_low": np.radians(29.5),
    "radius_low": np.radians(7.5),
    "az_high": np.radians(20.0),
    "el_high": np.radians(45.0),
    "radius_high": np.radians(10.0),
    "sense": -1,
}
WINDOW_DEFAULTS = {
    "reelin_fraction": 0.25,
    "ramp_fraction": 0.4,
    "min_height_m": 40.0,
    "max_height_m": 400.0,
    "stall_margin_deg": 2.0,
    "min_turn_radius": 25.0,
    # Upper bound of the operating radius r0 (lower stays
    # DEFAULT_OPTI_LIMITS["r0"][0]); 300 m = the numerical default.
    "r0_max_m": 300.0,
}
SHAPE_KEYS = tuple(SHAPE_DEFAULTS)

# Spline-hull bounds (opti_limits_override), identical for the eight and the
# baseline cycle: phi in [-0.6, 0.8] rad; C_beta LOOSE (only keeps the spline
# sane, must never bind). The binding vertical limit is the HEIGHT band of
# the profile (min_height_m / max_height_m -> the node-wise ``height`` rows
# of opti_phase, z = r sin(beta)), decided 2026-10-08: the flown 2019 cycle
# bottoms at ~44 m at r0 237 m; the eight's seed bottoms at ~21.4 deg, i.e.
# ~86 m at r0 236.7 m (seed height report).
C_PHI_BOUNDS = [-0.6, 0.8]
C_BETA_BOUNDS = [0.05, 1.50]

# --shape symmetric: a LEVEL eight, both lobe centres at one elevation and
# az_center -/+ az_offset (tilt 0); sense -1 climbs each lobe on its OUTER
# side and sweeps low across the middle. Two reel-in windows, one centred on
# each side-climb apex (half a period apart), splitting reelin_fraction
# equally. Everything else (height band, wind, turn radius, stall margin,
# depower band, STAGES) is the uploop_eight block's; only the C_phi box is
# made symmetric. Profile block: symmetric_eight (angles in rad).
SYMMETRIC_DEFAULTS = {
    "az_center": 0.0,
    "az_offset": np.radians(25.0),
    "el_center": np.radians(30.0),
    "radius": np.radians(10.0),
    "sense": -1,
}
SYMMETRIC_WINDOW_KEYS = ("reelin_fraction", "ramp_fraction", "r0_max_m")
C_PHI_BOUNDS_SYMMETRIC = [-0.8, 0.8]

# Shape of the current run (set by main): kind, mirror meridian and how the
# two lobes are told apart in the topology check ("elevation": low / high,
# "azimuth": left / right).
SPEC = {"kind": "slanted", "az_center": 0.0, "topology_by": "elevation"}

# Staged solve, warm-started stage to stage (see run_full_cycle_opti.STAGES
# for the proximal trust region + backstop box rationale). Never "zero": the
# seed closes through the wrap row and S0 already targets power.
# r0 travels at most R0_STEP_M per pass (backstop box). S1 re-centres and
# repeats while ANY step box (r0 or a shape box) is active, at most
# R0_MAX_PASSES passes.
R0_STEP_M = 100.0
R0_MAX_PASSES = 6
STAGES = [
    {
        "label": "fixed path | r0 closes the cycle",
        "params": ["r0"],
        "trust_region_weight": 0.05,
        "step_bounds": {"r0": R0_STEP_M},
        "repeat": 1,
        "warm_start": False,
    },
    {
        "label": "shape | C_phi, C_beta, r0 in a tight box",
        "params": ["C_phi", "C_beta", "r0"],
        "trust_region_weight": 0.05,
        "step_bounds": {"C_phi": 0.10, "C_beta": 0.10, "r0": R0_STEP_M},
        # re-centred passes while any step box is active, at most
        # R0_MAX_PASSES
        "repeat": R0_MAX_PASSES,
        "repeat_while_box_active": True,
        "warm_start": False,
    },
    {
        "label": "polish | weak anchor",
        "params": ["C_phi", "C_beta", "r0"],
        "trust_region_weight": 0.005,
        "step_bounds": {"C_phi": 0.10, "C_beta": 0.10, "r0": R0_STEP_M},
        "repeat": 1,
        "warm_start": True,
    },
]
MAX_ITER = 1000
N_PATH_SAMPLES = 720

ARCHIVED_CSV_FOOTNOTE = (
    "archived full-cycle runs predate the 2026-10-06/07 ROM / apparent-wind "
    "frame rework; their u_p went down to 1.34 m (outside the identified "
    "band), they carry 10.4 m turns and nodes on the s_dot floor, and the "
    "wind may differ (the old wind_10_z0_0.03 files use the same logarithmic "
    "law) -- indicative only."
)


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
def resolve_wind(args, block):
    """The wind the cycle is solved in: CLI flags > ``uploop_eight.wind`` > error."""
    cli = {
        "model_type": args.wind_law,
        "speed_wind_ref": args.wind_speed,
        "height_ref": args.wind_height,
        "z0": args.z0,
    }
    if any(v is not None for v in cli.values()):
        if cli["speed_wind_ref"] is None:
            raise SystemExit("--wind-speed is required with any other wind flag")
        wind = dict(CLI_WIND_DEFAULTS)
        wind.update({k: v for k, v in cli.items() if v is not None})
        if wind["model_type"] == "uniform":
            wind.pop("z0", None)
        return wind
    if block.get("wind"):
        return dict(block["wind"])
    raise SystemExit(
        "no wind for the up-loop eight: set uploop_eight.wind in the kite's "
        "cycle_profile.yaml or pass --wind-speed (the profile's top-level "
        "wind: is deliberately not inherited)"
    )


def resolve_shape(block, scale_lobes=1.0):
    """(shape kwargs of slanted_eight_angles, window / limit knobs)."""
    shape = dict(SHAPE_DEFAULTS)
    window = dict(WINDOW_DEFAULTS)
    for key, value in block.items():
        if key in shape:
            shape[key] = float(value)
        elif key in window:
            window[key] = float(value)
        elif key != "wind":
            raise ValueError(
                f"uploop_eight: unknown key {key!r}; allowed "
                f"{sorted(shape) + sorted(window) + ['wind']}"
            )
    shape["sense"] = int(np.sign(shape["sense"])) or -1
    for key in ("radius_low", "radius_high"):
        shape[key] *= float(scale_lobes)
    return shape, window


# ---------------------------------------------------------------------------
# Seed config
# ---------------------------------------------------------------------------
def resolve_symmetric_shape(block, window, scale_lobes=1.0):
    """(shape kwargs of slanted_eight_angles, window, az_center) of the
    level eight from the ``symmetric_eight`` profile block."""
    cfg = dict(SYMMETRIC_DEFAULTS)
    window = dict(window)
    for key, value in block.items():
        if key in cfg:
            cfg[key] = float(value)
        elif key in SYMMETRIC_WINDOW_KEYS:
            window[key] = float(value)
        else:
            raise ValueError(
                f"symmetric_eight: unknown key {key!r}; allowed "
                f"{sorted(cfg) + list(SYMMETRIC_WINDOW_KEYS)}"
            )
    radius = cfg["radius"] * float(scale_lobes)
    shape = {
        "az_low": cfg["az_center"] - cfg["az_offset"],
        "el_low": cfg["el_center"],
        "radius_low": radius,
        "az_high": cfg["az_center"] + cfg["az_offset"],
        "el_high": cfg["el_center"],
        "radius_high": radius,
        "sense": int(np.sign(cfg["sense"])) or -1,
    }
    return shape, window, float(cfg["az_center"])


def uploop_limits_override(window, kite, c_phi_bounds=None):
    """The ONE set of optimizer bound overrides of this formulation: the ROM
    depower band, the height band (``min_height_m`` / ``max_height_m``, the
    binding vertical limit), the loose spline hull (``C_BETA_BOUNDS``,
    ``C_PHI_BOUNDS``) and the kite's
    own overrides (its ROM AoA validity). Shared by the eight's seed and the
    baseline seed so both are optimized inside the same box."""
    z_min, z_max = float(window["min_height_m"]), float(window["max_height_m"])
    if z_max <= z_min:
        raise ValueError("uploop_eight: max_height_m must exceed min_height_m")
    override = {
        "input_depower": [float(v) for v in sorted(fpcc.DEPOWER_BAND)],
        "height": [z_min, z_max],
        "r0": [float(DEFAULT_OPTI_LIMITS["r0"][0]), float(window["r0_max_m"])],
        "C_beta": [float(v) for v in C_BETA_BOUNDS],
        "C_phi": [
            float(v)
            for v in (
                c_phi_bounds
                if c_phi_bounds is not None
                else (
                    C_PHI_BOUNDS_SYMMETRIC
                    if SPEC["kind"] == "symmetric"
                    else C_PHI_BOUNDS
                )
            )
        ],
    }
    override.update(kite.get("opti_limits_override") or {})
    return override


def hull_bounds_text(sim_parameters):
    """``{"hull_c_phi": "-34..46 deg", "height_band": "40..400 m"}`` of a config."""
    override = sim_parameters.get("opti_limits_override") or {}
    lo, hi = override.get("C_phi", DEFAULT_OPTI_LIMITS["C_phi"])
    z_lo, z_hi = override.get("height", DEFAULT_OPTI_LIMITS["height"])
    return {
        "hull_c_phi": f"{np.degrees(float(lo)):.0f}..{np.degrees(float(hi)):.0f} deg",
        "height_band": f"{float(z_lo):g}..{float(z_hi):g} m",
    }


def seed_height_report(path_parameters, window, label):
    """Min / max height z = r0 sin(beta) of a seed path at its r0, flagged
    against the profile's height floor."""
    r0 = float(path_parameters["r0"])
    _, _, beta = sample_path(path_parameters)
    z = r0 * np.sin(beta)
    z_min = float(window["min_height_m"])
    ok = float(z.min()) >= z_min
    print(
        f"[{'ok  ' if ok else 'WARN'}] {label} height at r0={r0:.1f} m: "
        f"{z.min():.1f}..{z.max():.1f} m (floor {z_min:g} m"
        + ("" if ok else "; the NLP must lift it") + ")"
    )
    return float(z.min()), float(z.max())


def project_path_into_hull(path, override):
    """Clip the spline control points into the hull bounds of ``override``
    (``C_phi`` / ``C_beta``): a B-spline lies inside the convex hull of its
    control polygon, so the clipped curve is inside the bounds too (the NLP
    does NOT clip -- a coefficient outside the box makes the shape stage
    infeasible). Returns ``(path, n_clipped)``; the input dict is untouched.
    """
    path = dict(path)
    n_clipped = 0
    for key in ("C_phi", "C_beta"):
        bounds = override.get(key)
        if bounds is None:
            continue
        values = np.asarray(path[key], dtype=float)
        clipped = np.clip(values, float(bounds[0]), float(bounds[1]))
        n_clipped += int(np.count_nonzero(~np.isclose(clipped, values)))
        path[key] = np.round(clipped, 6).tolist()
    return path, n_clipped


def hull_and_fair(path, override, curvature_limit, label):
    """Project ``path`` into the hull, fair it to ``curvature_limit`` (1/m),
    re-project once if the fairing pushed a point back out, and enforce the
    curvature guard. Returns ``(path, n_clipped)`` with the projection count
    of the first pass (what the log reports)."""
    apex_before = float(np.degrees(np.max(path["C_beta"])))
    path, n_clipped = project_path_into_hull(path, override)
    if n_clipped:
        print(
            f"[hull] {label}: projected {n_clipped} control points into the hull "
            f"(apex {apex_before:.1f} -> {np.degrees(np.max(path['C_beta'])):.1f} deg)"
        )
    path, fair = fair_periodic_spline_to_curvature_limit(path, curvature_limit)
    if fair["changed"]:
        print(
            f"[fair] {label} spline curvature {fair['max_before']:.4g} -> "
            f"{fair['max_after']:.4g} 1/m (limit {curvature_limit:.4g}, "
            f"sharpest at s={fair['s_at_max_before']:.3f}); moved "
            f"{len(fair['touched'])}/{path['M']} control points, path moved "
            f"<= {np.degrees(fair['max_path_move']):.2f} deg"
        )
    if not fair["converged"]:
        print(f"[fair] WARN: {label}: curvature limit not reachable within the trust box")
    path, n_again = project_path_into_hull(path, override)
    if n_again:
        print(
            f"[hull] WARN: {label}: fairing pushed {n_again} control points back "
            "out of the hull; re-projected once (curvature re-checked below)"
        )
    fpcc.enforce_curvature_limit(path, curvature_limit)
    return path, n_clipped


def uploop_curvature_limit(window):
    """Fairing limit (1/m): the write-time guard, tightened to the turn-radius
    floor -- stage 0 flies the FIXED seed path (only r0 free), so a seed
    sharper than ``min_turn_radius`` is infeasible there by geometry."""
    min_turn_radius = float(window["min_turn_radius"] or 0.0)
    limit = fpcc.CURVATURE_LIMIT_1PM
    if min_turn_radius > 0.0:
        limit = min(limit, 1.0 / min_turn_radius)
    return limit


def build_uploop_config(kite, shape, window, duration_s=None):
    """``{"reelout": ...}`` config of the slanted-eight cycle (the schema of
    ``fit_periodic_cycle_config.build_config``), plus the seed landmarks.

    The spline is faired to the turn-radius floor: stage 0 flies the FIXED
    seed path (only r0 free), so a seed sharper than ``min_turn_radius``
    would make that stage infeasible by geometry.
    """
    r0 = float(kite["r0"])
    path, fit = make_slanted_eight_bspline_path_parameters(r0, **shape)
    print(
        f"Slanted eight fitted with M={fit['M']} "
        f"(max deviation {fit['fit_error_m']:.3f} m at r0={r0:.1f} m"
        + ("" if fit["converged"] else "; NOT within the 1 m target")
        + ")"
    )
    landmarks = slanted_eight_landmarks(**shape)
    min_turn_radius = float(window["min_turn_radius"] or 0.0)
    opti_limits_override = uploop_limits_override(window, kite)
    path, _ = hull_and_fair(
        path, opti_limits_override, uploop_curvature_limit(window), "eight"
    )
    symmetric = SPEC["kind"] == "symmetric"
    if symmetric:
        # Exactly mirror-symmetric coefficients (the LSQ fit of the mirrored
        # target already is, up to the fairing): the mirror rows hold at the
        # seed, so the fixed-path stage 0 is consistent with them.
        before = np.r_[path["C_phi"], path["C_beta"]]
        path = symmetrize_periodic_path(path, az_center=SPEC["az_center"])
        path["C_phi"] = np.round(path["C_phi"], 9).tolist()
        path["C_beta"] = np.round(path["C_beta"], 9).tolist()
        print(
            "Symmetrized the seed spline (max coefficient change "
            f"{np.max(np.abs(np.r_[path['C_phi'], path['C_beta']] - before)):.2e} rad)"
        )
        path, n_clipped = project_path_into_hull(path, opti_limits_override)
        if n_clipped:
            print(
                f"[hull] WARN: the symmetrized seed had {n_clipped} control points "
                "outside the hull; projected"
            )
        fpcc.enforce_curvature_limit(path, uploop_curvature_limit(window))

    winch = fpcc._winch_law_with_hardware_rating(kite["winch_law"])
    duration = float(kite["cycle_duration_s"] if duration_s is None else duration_s)
    n_points = int(
        np.clip(np.ceil(duration * fpcc.NPOINTS_PER_SECOND), 50, fpcc.N_POINTS_MAX)
    )
    if symmetric:
        if n_points % 2:  # node i pairs with node i + N/2
            n_points += 1 if n_points + 1 <= fpcc.N_POINTS_MAX else -1
    s_grid = np.linspace(0.0, 1.0, n_points + 1, endpoint=True)
    reelin_fraction = float(window["reelin_fraction"])
    if symmetric:
        # Two reel-in windows half a period apart, one per side-climb apex,
        # each half the reel-in fraction: the profile is mirror-symmetric
        # on the even node grid.
        s_apex = float(landmarks["s_lobe_apex"][0])
        centres = (s_apex, (s_apex + 0.5) % 1.0)
        bumps = [
            fpcc._synthetic_depower(
                s_grid,
                1.0 - 0.5 * reelin_fraction,
                float(window["ramp_fraction"]),
                reelin_center=c,
            )
            for c in centres
        ]
        u_dep = bumps[0] + bumps[1] - float(fpcc.DEPOWER_BAND[0])
        where = "two windows of {:.3f} centred on the side apices s={:.3f}, {:.3f}".format(
            0.5 * reelin_fraction, *centres
        )
    else:
        s_apex = float(landmarks["s_high_apex"])
        u_dep = fpcc._synthetic_depower(
            s_grid,
            1.0 - reelin_fraction,
            float(window["ramp_fraction"]),
            reelin_center=s_apex,
        )
        where = f"{reelin_fraction:.2f} of the period centred on the apex s={s_apex:.3f}"
    print(
        f"Seed: n_points={n_points} ({duration:.0f} s at "
        f"{fpcc.NPOINTS_PER_SECOND:g} nodes/s), depower {where}"
    )

    sim_parameters = {
        "start_angle": 0.0,
        "end_angle": 1.0,
        "n_points": n_points,
        "input_depower": float(u_dep[0]),
        "optimize_depower_profile": True,
        "input_depower_profile": u_dep.round(6).tolist(),
        "depower_rate": list(fpcc.DEPOWER_RATE),
        "solver_accept_residual_norm": 1.0e-3,
        "opti_limits_override": opti_limits_override,
        "expand_nlp": True,
        "ipopt_robust": False,
        "require_full_trajectory": True,
        "winch_mode": "free_speed",
        "periodic_wrap": True,
        "close_radial_cycle": False,
        "radial_closure_weight": 0.0,
        "stall_margin_deg": float(window["stall_margin_deg"]),
    }
    if min_turn_radius > 0.0:
        sim_parameters["min_turn_radius"] = min_turn_radius
    if SPEC.get("mirror"):
        sim_parameters["mirror_symmetry"] = True
        sim_parameters["mirror_azimuth"] = float(SPEC["az_center"])
    config = {
        "reelout": {
            "pattern_type": "spline_periodic",
            "path_parameters": path,
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
                "winch_offset_depower_gain": winch["winch_offset_depower_gain"],
                "winch_depower_ref": winch["winch_depower_ref"],
            },
            "sim_parameters": sim_parameters,
        }
    }
    return config, landmarks


def apply_uploop_formulation(reelout, window, kite):
    """Returns ``(reelout, n_clipped)``; ``n_clipped`` is the number of
    control points projected into the hull (for the table footnote)."""
    return _apply_uploop_formulation(reelout, window, kite)


def _apply_uploop_formulation(reelout, window, kite):
    """Put another seed (the classic multi-lobe cycle) on the SAME formulation:
    periodic wrap, free reel speed, the ROM depower band, the stall margin and
    the turn-radius floor. Its depower profile is kept; its path goes through
    the same ``hull_and_fair`` as the eight's: projected into the shared
    (loose) spline hull, then faired to the turn-radius floor (stage 0 flies
    the fixed path, so a seed sharper than the floor is infeasible there by
    geometry). The binding vertical limit is the shared height band."""
    sim = reelout["sim_parameters"]
    sim["winch_mode"] = "free_speed"
    sim["periodic_wrap"] = True
    sim["close_radial_cycle"] = False
    sim["stall_margin_deg"] = float(window["stall_margin_deg"])
    min_turn_radius = float(window["min_turn_radius"] or 0.0)
    if min_turn_radius > 0.0:
        sim["min_turn_radius"] = min_turn_radius
    sim["radial_closure_weight"] = 0.0
    # The normal cycle keeps the slanted C_phi box whatever --shape is.
    sim["opti_limits_override"] = uploop_limits_override(
        window, kite, c_phi_bounds=C_PHI_BOUNDS
    )
    reelout["path_parameters"], n_clipped = hull_and_fair(
        reelout["path_parameters"],
        sim["opti_limits_override"],
        uploop_curvature_limit(window),
        "baseline",
    )
    return reelout, n_clipped


# ---------------------------------------------------------------------------
# Path diagnostics
# ---------------------------------------------------------------------------
def sample_path(path_parameters, n_samples=N_PATH_SAMPLES):
    """(s, azimuth, elevation) [rad] over one period of a periodic spline."""
    M = int(path_parameters["M"])
    pattern = PeriodicBSpline(
        M=M,
        C_phi=np.asarray(path_parameters["C_phi"], dtype=float).reshape((M, 1)),
        C_beta=np.asarray(path_parameters["C_beta"], dtype=float).reshape((M, 1)),
        s_init=float(path_parameters.get("s_init", 0.0)),
        s_final=float(path_parameters.get("s_final", 1.0)),
        downloops=bool(path_parameters.get("downloops", True)),
    )
    r0 = float(path_parameters["r0"])
    s = np.linspace(0.0, 1.0, int(n_samples), endpoint=False)
    phi = np.asarray(pattern.azimuth(r0, s)).ravel()
    beta = np.asarray(pattern.elevation(r0, s)).ravel()
    return s, phi, beta


def path_topology(path_parameters):
    """``(crossings, senses)`` of a path: the self-crossing count and, for an
    eight, the signs of the (low, high) lobe areas (+1 anticlockwise)."""
    s, phi, beta = sample_path(path_parameters)
    crossings = count_self_crossings(phi, beta)
    senses = None
    if crossings == 1:
        a_low, a_high = lobe_signed_areas(
            phi, beta, s, eight_crossing_s(phi, beta, s), by=SPEC["topology_by"]
        )
        senses = (int(np.sign(a_low)), int(np.sign(a_high)))
    return crossings, senses


def describe_topology(crossings, senses):
    text = f"{crossings} self-crossing(s)"
    if senses is not None:
        names = {1: "anticlockwise", -1: "clockwise", 0: "degenerate"}
        first, second = (
            ("left", "right") if SPEC["topology_by"] == "azimuth" else ("low", "high")
        )
        text += (
            f"; {first} lobe {names[senses[0]]}, {second} lobe {names[senses[1]]}"
        )
    return text


def check_topology(path_parameters, reference, what):  # returns the topology
    """Warn (never abort) when the path's crossings / lobe senses changed."""
    topo = path_topology(path_parameters)
    print(f"    {what}: {describe_topology(*topo)}")
    if topo != reference:
        print(
            "    ** WARNING: pattern topology changed vs the seed "
            f"({describe_topology(*reference)}) **"
        )
    return topo


def symmetry_report(path_parameters, series=None):
    """Mirror diagnostics of a path (and its flown series): apex elevation
    on each side of the mirror meridian, the largest great-circle defect
    between the point at s + 1/2 and the mirror image of the point at s, and
    the reel-in arcs (v_r < 0 runs of the series: count, apex elevation and
    side of each, durations)."""
    az_c = float(SPEC["az_center"])
    s, phi, beta = sample_path(path_parameters)
    half = s.size // 2
    left, right = phi < az_c, phi >= az_c
    out = {
        "apex_left": float(np.degrees(beta[left].max())) if left.any() else None,
        "apex_right": float(np.degrees(beta[right].max())) if right.any() else None,
        "mirror_defect": float(
            np.degrees(
                np.max(
                    _great_circle_deviation(
                        np.roll(phi, -half), np.roll(beta, -half), 2.0 * az_c - phi, beta
                    )
                )
            )
        ),
    }
    out["apex_lr"] = (
        "n/a"
        if out["apex_left"] is None or out["apex_right"] is None
        else f"{out['apex_left']:.1f}/{out['apex_right']:.1f}"
    )
    if series and {"speed_radial", "angle_elevation", "angle_azimuth", "t"} <= series.keys():
        vr = series["speed_radial"]
        n = min(len(vr), len(series["t"]), len(series["angle_elevation"]))
        neg = vr[:n] < 0.0
        arcs = []
        if neg.any() and not neg.all():
            start = int(np.flatnonzero(~neg)[0])  # rotate to a reel-out node
            order = np.r_[np.arange(start, n), np.arange(0, start)]
            run = []
            for j in list(order) + [order[0]]:
                if neg[j]:
                    run.append(j)
                elif run:
                    el = np.degrees(series["angle_elevation"][run])
                    az = np.degrees(series["angle_azimuth"][run])
                    t = series["t"][:n]
                    dur = float(t[run[-1]] - t[run[0]]) if run[-1] >= run[0] else float(
                        t[-1] - t[run[0]] + t[run[-1]] - t[0]
                    )
                    side = "L" if az[int(np.argmax(el))] < np.degrees(az_c) else "R"
                    arcs.append((float(el.max()), side, dur))
                    run = []
        out["reelin_arcs"] = (
            f"{len(arcs)}: "
            + ", ".join(f"{el:.0f}{side} {dur:.0f}s" for el, side, dur in arcs)
            if arcs
            else "0"
        )
    return out


def describe_symmetry(sym):
    return (
        f"apex L/R {sym['apex_lr']} deg, mirror defect {sym['mirror_defect']:.2f} deg"
        + (f", reel-in arcs {sym['reelin_arcs']}" if "reelin_arcs" in sym else "")
    )


def ipopt_status(result):
    try:
        return str(result.solution.stats().get("return_status"))
    except Exception:
        return "n/a"


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
COMPARE_ROWS = (
    ("IPOPT status", "ipopt_status", 1.0, "s"),
    ("C_phi hull", "hull_c_phi", 1.0, "s"),
    ("height band (m)", "height_band", 1.0, "s"),
    ("height min (m)", "height_min", 1.0, ".1f"),
    ("height max (m)", "height_max", 1.0, ".1f"),
    ("r0 (m)", "r0", 1.0, ".1f"),
    ("average power (kW)", "avg_power", 1e-3, ".2f"),
    ("cycle duration (s)", "duration", 1.0, ".1f"),
    ("reel-out time fraction", "reelout_time_fraction", 1.0, ".2f"),
    ("seam closure (m)", "closure_seam", 1.0, "+.2f"),
    ("tension max (kN)", "tension_max", 1e-3, ".2f"),
    ("max |steering|", "steer_absmax", 1.0, ".3f"),
    ("steering reversals", "steering_reversals", 1.0, ".0f"),
    ("depower min (m)", "depower_min", 1.0, ".2f"),
    ("depower max (m)", "depower_max", 1.0, ".2f"),
)
SYMMETRIC_ROWS = (
    ("apex L/R (deg)", "apex_lr", 1.0, "s"),
    ("mirror defect (deg)", "mirror_defect", 1.0, ".2f"),
    ("reel-in arcs", "reelin_arcs", 1.0, "s"),
)


def wrapped_cycle_metrics(series, path_parameters=None):
    """``cycle_comparison_plots.cycle_metrics`` on the CLOSED cycle.

    The periodic wrap makes the seam interval (last node -> first node, of
    duration ``(s_period - s[-1] + s[0]) / s_dot[-1]``) part of the cycle,
    so energy, duration and power follow the NLP's own left-rule quadrature
    over ALL N intervals, ``dt_i = (s_{i+1} - s_i) / s_dot_i`` (the generic
    metric stops at the last node and the series' ``t`` is a right rule), and
    the
    closure is the seam residual ``r[0] - r[-1] - v_r[-1] dt_seam`` (zero on
    a converged wrap; ``r[-1] - r[0]`` is one reel step off by design).
    ``path_parameters`` gives the period ``s_final - s_init`` (1 when absent,
    the full-cycle splines' ``s in [0, 1)``).
    """
    m = ccp.cycle_metrics(series)
    if {"distance_radial", "angle_elevation"} <= series.keys():
        n_z = min(len(series["distance_radial"]), len(series["angle_elevation"]))
        z = series["distance_radial"][:n_z] * np.sin(series["angle_elevation"][:n_z])
        m["height_min"], m["height_max"] = float(z.min()), float(z.max())
    if path_parameters is not None and "r0" in path_parameters:
        m["r0"] = float(path_parameters["r0"])
    keys = {"s", "s_dot", "tension_tether_ground", "speed_radial", "distance_radial"}
    if not keys <= series.keys():
        return m
    n = min(len(series[k]) for k in keys)
    s, sd = series["s"][:n], series["s_dot"][:n]
    T, vr, r = (series[k][:n] for k in ("tension_tether_ground", "speed_radial", "distance_radial"))
    if n < 2:
        return m
    period = 1.0
    if path_parameters is not None:
        period = float(path_parameters.get("s_final", 1.0)) - float(
            path_parameters.get("s_init", 0.0)
        )
    # The NLP's own left rule, dt_i = (s_{i+1} - s_i) / s_dot_i over ALL N
    # intervals (the seam ends at s_0 + period). The series' ``t`` is NOT
    # used: series_from_optimizer_result rebuilds it with a right rule.
    dt = (np.r_[s[1:], s[0] + period] - s) / np.maximum(sd, 1e-9)
    duration = float(np.sum(dt))
    m["energy"] = float(np.sum(T * vr * dt))
    m["duration"] = duration
    m["avg_power"] = m["energy"] / max(duration, 1e-9)
    m["reelout_time_fraction"] = float(np.sum(dt[vr > 0.0]) / max(duration, 1e-9))
    m["closure_seam"] = float(r[0] - r[-1] - vr[-1] * dt[-1])
    return m


def series_from_csv(path):
    """Series dict from a saved optimizer trajectory CSV (same reconstruction
    as ``cycle_comparison_plots.series_from_optimizer_result``)."""
    data = np.genfromtxt(path, delimiter=",", names=True)
    series = {name: np.asarray(data[name], dtype=float).ravel() for name in data.dtype.names}
    ccp._ensure_power(series)
    if "t" not in series and {"s", "s_dot"} <= series.keys():
        ds = np.diff(series["s"], prepend=series["s"][0])
        series["t"] = np.cumsum(ds / np.maximum(series["s_dot"], 1e-9))
    return series


REFERENCE_LIMIT_KEYS = ("r0", "height", "C_beta", "input_depower", "angle_of_attack")


def reference_columns(out_dir, tag, override=None):
    """(label, metrics) of earlier runs saved in ``out_dir`` (slanted S1/S2,
    baseline S2), rebuilt from their YAML + CSV -- absent files are skipped.

    Only the wind is in the filename. Every result YAML stores its own
    ``sim_parameters.opti_limits_override``, so the shared limits
    (``REFERENCE_LIMIT_KEYS``; C_phi differs between shapes by design) are
    compared with ``override`` and a mismatch is printed."""
    from types import SimpleNamespace

    cols = []
    for label, prefix, stage in (
        ("ref slanted S1", "uploop_eight", 1),
        ("ref slanted S2", "uploop_eight", 2),
        ("ref baseline S2", "baseline", 2),
    ):
        # (no with_suffix: the wind tag itself carries dots, e.g. z0_0.03)
        stem = f"{prefix}_optimized_{tag}_stage{stage}"
        yaml_path, csv_path = out_dir / f"{stem}.yaml", out_dir / f"{stem}.csv"
        if not (yaml_path.is_file() and csv_path.is_file()):
            continue
        with yaml_path.open("r", encoding="utf-8") as f:
            path = yaml.safe_load(f)["reelout"]
        data = np.genfromtxt(csv_path, delimiter=",", names=True)
        traj = {n: np.asarray(data[n], dtype=float) for n in data.dtype.names}
        series = ccp.series_from_optimizer_result(
            SimpleNamespace(optimized_trajectory=traj), path["path_parameters"]
        )
        m = wrapped_cycle_metrics(series, path["path_parameters"])
        m.update(hull_bounds_text(path["sim_parameters"]))
        stored = path["sim_parameters"].get("opti_limits_override") or {}
        for key in REFERENCE_LIMIT_KEYS if override is not None else ():
            mine, theirs = override.get(key), stored.get(key)
            if mine is None and theirs is None:
                continue
            if (
                mine is None
                or theirs is None
                or not np.allclose(np.asarray(mine, float), np.asarray(theirs, float))
            ):
                print(
                    f"** WARNING: {label} ({yaml_path.name}) was solved with "
                    f"{key} = {theirs}, this run uses {mine} **"
                )
        cols.append((label, m))
    return cols


def print_metrics_table(columns, footnotes=()):
    """``columns``: list of (name, metrics dict)."""
    width = max(20, max(len(name) for name, _ in columns) + 2)
    header = f"{'metric':<26}" + "".join(f"{name:>{width}}" for name, _ in columns)
    print("\n--- Cycle comparison ---")
    print(header)
    print("-" * len(header))
    rows = COMPARE_ROWS + (
        SYMMETRIC_ROWS if any("apex_lr" in m for _, m in columns) else ()
    )
    for label, key, scale, spec in rows:
        cells = []
        for _, m in columns:
            value = m.get(key)
            if value is None:
                cells.append("n/a")
            elif spec == "s":
                cells.append(str(value)[:width - 1])
            else:
                cells.append(f"{value * scale:{spec}}")
        print(f"{label:<26}" + "".join(f"{c:>{width}}" for c in cells))
    for note in footnotes:
        print(f"* {note}")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def plot_uploop_summary(
    series,
    opt_path,
    seed_path=None,
    *,
    height_band=None,
    tension_band=None,
    vr_bounds=None,
    depower_band=None,
    steering_limit=None,
    title=None,
    save_path=None,
):
    """One-page summary of a solved cycle.

    Left: the (azimuth, elevation) chart with the flown path coloured by the
    reel speed (diverging, white at v_r = 0), the seed spline dashed, the
    reel-out <-> reel-in handovers marked, the height floor (and ceiling,
    when it is inside the axis) drawn at the solved r0 and direction arrows. Right, on a shared time axis: mechanical power with its mean,
    tether force within the winch band, reel speed within the drum bounds,
    depower within the ROM band, and steering with the actuator limit, the
    reversal deadband and every counted reversal marked.
    """
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import TwoSlopeNorm

    from awetrim.plotting.plotting import PALETTE

    ink, muted = PALETTE["Black"], "0.55"
    t = series["t"] - series["t"][0]
    phi = np.degrees(series["angle_azimuth"])
    beta = np.degrees(series["angle_elevation"])
    vr = series["speed_radial"]
    n = min(len(t), len(phi), len(beta), len(vr))
    t, phi, beta, vr = t[:n], phi[:n], beta[:n], vr[:n]

    fig = plt.figure(figsize=(14.0, 8.2))
    grid = fig.add_gridspec(
        5, 2, width_ratios=(1.15, 1.0), left=0.05, right=0.98, top=0.92,
        bottom=0.08, wspace=0.18, hspace=0.12,
    )
    ax_map = fig.add_subplot(grid[:, 0])
    axes = [fig.add_subplot(grid[k, 1]) for k in range(5)]
    for ax in axes[1:]:
        ax.sharex(axes[0])

    # --- chart: closed path coloured by v_r
    phi_c, beta_c = np.r_[phi, phi[0]], np.r_[beta, beta[0]]
    points = np.column_stack([phi_c, beta_c]).reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    vmax = max(float(np.max(np.abs(vr))), 1e-3)
    lc = LineCollection(
        segments, cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax),
        linewidth=2.6, zorder=3,
    )
    lc.set_array(np.r_[vr, vr[0]])
    ax_map.add_collection(lc)
    cbar = fig.colorbar(lc, ax=ax_map, pad=0.02, shrink=0.8)
    cbar.set_label(r"$v_\mathrm{r}$ (m s$^{-1}$)")
    if seed_path is not None:
        _, phi_s, beta_s = sample_path(seed_path)
        ax_map.plot(
            np.degrees(np.r_[phi_s, phi_s[0]]), np.degrees(np.r_[beta_s, beta_s[0]]),
            color=muted, linestyle="--", linewidth=1.2, label="seed spline", zorder=2,
        )
    # reel-out <-> reel-in handovers (sign changes of v_r, seam included)
    sign = np.sign(vr)
    for k in np.flatnonzero(sign != np.roll(sign, -1)):
        j = (k + 1) % n
        to_reelin = vr[j] < 0.0
        ax_map.plot(
            phi[j], beta[j], marker="v" if to_reelin else "^", ms=9, color="w",
            mec=ink, mew=1.4, zorder=6,
            label=("reel-out to reel-in" if to_reelin else "reel-in to reel-out"),
        )
    if height_band is not None:
        r0 = float(opt_path["r0"])
        for z_lim, what in zip(height_band, ("floor", "ceiling")):
            if z_lim is None or z_lim >= r0:
                continue  # unreachable at this radius: not a line on the chart
            el_lim = np.degrees(np.arcsin(float(z_lim) / r0))
            if what == "ceiling" and el_lim > beta.max() + 10.0:
                continue  # outside the axis
            ax_map.axhline(el_lim, color=muted, linewidth=0.9, linestyle=":")
            ax_map.text(
                0.01, el_lim, f" z = {float(z_lim):g} m at r0 = {r0:.0f} m",
                transform=ax_map.get_yaxis_transform(), va="bottom", fontsize=8,
                color=muted,
            )
    step = max(1, n // 24)
    d_phi, d_beta = np.gradient(phi_c)[:-1], np.gradient(beta_c)[:-1]
    norm = np.maximum(np.hypot(d_phi, d_beta), 1e-9)
    arrow_deg = 0.04 * max(np.ptp(phi), np.ptp(beta), 1.0)
    ax_map.quiver(
        phi[::step], beta[::step],
        (arrow_deg * d_phi / norm)[::step], (arrow_deg * d_beta / norm)[::step],
        color=ink, angles="xy", scale_units="xy", scale=1.0, width=0.004,
        headwidth=5, alpha=0.85, zorder=5,
    )
    ax_map.plot(phi[0], beta[0], "o", ms=7, color="w", mec=ink, mew=1.4, zorder=7, label="s = 0")
    ax_map.set_xlabel(r"azimuth $\phi$ ($^\circ$)")
    ax_map.set_ylabel(r"elevation $\beta$ ($^\circ$)")
    ax_map.set_aspect("equal", adjustable="datalim")
    ax_map.grid(True, alpha=0.3)
    handles, labels = ax_map.get_legend_handles_labels()
    seen = {}
    for h, lab in zip(handles, labels):
        seen.setdefault(lab, h)
    ax_map.legend(seen.values(), seen.keys(), loc="upper left", fontsize=8)
    ax_map.set_title("flown path, coloured by reel speed", fontsize=10)

    # --- time series
    power = series["mechanical_power"][:n] / 1e3
    ax = axes[0]
    ax.plot(t, power, color=ink, linewidth=1.6)
    p_mean = (
        wrapped_cycle_metrics(series, opt_path).get("avg_power", float(np.mean(power) * 1e3))
        / 1e3
    )
    ax.axhline(p_mean, color=PALETTE["Vermillion"], linewidth=1.1, linestyle="--")
    ax.text(t[-1], p_mean, f" mean {p_mean:.2f} kW", va="bottom", ha="right", fontsize=8,
            color=PALETTE["Vermillion"])
    ax.axhline(0.0, color=muted, linewidth=0.6)
    ax.set_ylabel(r"$P$ (kW)")

    ax = axes[1]
    ax.plot(t, series["tension_tether_ground"][:n] / 1e3, color=ink, linewidth=1.6)
    if tension_band is not None:
        ax.axhspan(tension_band[0] / 1e3, tension_band[1] / 1e3, color=muted, alpha=0.12, lw=0)
    ax.set_ylabel(r"$F_\mathrm{t}$ (kN)")

    ax = axes[2]
    ax.plot(t, vr, color=ink, linewidth=1.6)
    ax.axhline(0.0, color=muted, linewidth=0.6)
    if vr_bounds is not None:
        for v in vr_bounds:
            ax.axhline(v, color=muted, linewidth=0.9, linestyle=":")
    ax.set_ylabel(r"$v_\mathrm{r}$ (m s$^{-1}$)")

    ax = axes[3]
    if "input_depower" in series:
        ax.plot(t, series["input_depower"][:n], color=ink, linewidth=1.6)
    if depower_band is not None:
        ax.axhspan(min(depower_band), max(depower_band), color=muted, alpha=0.12, lw=0)
    ax.set_ylabel(r"$u_\mathrm{p}$ (m)")

    ax = axes[4]
    u_s = series["input_steering"][:n]
    ax.plot(t, u_s, color=ink, linewidth=1.6)
    ax.axhspan(-STEERING_DEADBAND, STEERING_DEADBAND, color=muted, alpha=0.15, lw=0)
    if steering_limit is not None:
        for v in (-steering_limit, steering_limit):
            ax.axhline(v, color=muted, linewidth=0.9, linestyle=":")
    pairs = steering_reversal_pairs(u_s, deadband=STEERING_DEADBAND, periodic=True)
    for i, j in pairs:
        t_rev = 0.5 * (t[i] + t[j]) if j > i else t[j]  # seam pair -> cycle start
        ax.axvline(t_rev, color=PALETTE["Vermillion"], linewidth=0.9, alpha=0.8)
    ax.set_ylabel(r"$u_\mathrm{s}$")
    ax.set_xlabel(r"$t$ (s)")
    ax.text(
        0.99, 0.95, f"{len(pairs)} reversals (deadband {STEERING_DEADBAND:g})",
        transform=ax.transAxes, ha="right", va="top", fontsize=8, color=PALETTE["Vermillion"],
    )
    for ax in axes:
        ax.grid(True, alpha=0.3)
    for ax in axes[:-1]:
        plt.setp(ax.get_xticklabels(), visible=False)
    if title:
        fig.suptitle(title, fontsize=11)
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"Saved summary figure to {save_path}")
    return fig


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def march_seed(reelout, kite, wind, run_plots=False):
    """Strict forward march of a cycle config; returns ``(cycle, phase)`` or
    ``(cycle, None)`` when a node failed to trim (require_full_trajectory)."""
    system_model = create_system_model_from_yaml(yaml_path=kite["system_config"])
    system_model.wind = rfco.build_wind_model(**wind)
    cycle = Phase(
        system_model=system_model,
        pattern_config=reelout,
        start_state=rfco.build_start_state(reelout),
    )
    try:
        phase, _ = cycle.run_simulation(
            run_plots=run_plots, phase_sim=True, seed_warm_start=True
        )
    except RuntimeError as exc:
        print(f"Seed forward simulation failed: {exc}")
        return cycle, None
    return cycle, phase


def feasibility_window(window, landmarks):
    """``artificial`` argument of ``fpcc._feasibility_report`` (it labels
    violations by ONE reel-in window). The slanted eight has one window on
    the high-lobe apex; the symmetric eight has two half a period apart, so
    only the first (left-apex) window is passed and a note says so."""
    reelin_fraction = float(window["reelin_fraction"])
    if SPEC["kind"] == "symmetric":
        print(
            "[note] the feasibility report labels violations by ONE reel-in "
            "window: the left-apex window is used, the mirrored right-apex "
            "window (s + 0.5) is labelled 'figure eights'"
        )
        return {
            "reelout_fraction": 1.0 - 0.5 * reelin_fraction,
            "reelin_center": float(landmarks["s_lobe_apex"][0]),
        }
    return {
        "reelout_fraction": 1.0 - reelin_fraction,
        "reelin_center": float(landmarks["s_high_apex"]),
    }


def report_seed(phase, system_model, config, landmarks, window):
    """Feasibility report of the marched seed plus the eight's own checks."""
    ok = fpcc._feasibility_report(
        phase, system_model, config,
        artificial=feasibility_window(window, landmarks),
    )
    series = ccp.series_from_phase(phase)
    if SPEC["kind"] == "symmetric":
        print(
            "[info] seed symmetry: "
            + describe_symmetry(symmetry_report(config["reelout"]["path_parameters"], series))
        )
    else:
        s, vr = series["s"], series["speed_radial"]
        n = min(len(s), len(vr))
        high = landmarks["high_lobe"](s[:n])
        low_out = float(np.mean(vr[:n][~high] > 0.0)) if (~high).any() else float("nan")
        high_in = float(np.mean(vr[:n][high] < 0.0)) if high.any() else float("nan")
        print(
            f"[{'ok  ' if low_out > 0.9 else 'WARN'}] reel-out on the low lobe: "
            f"{100 * low_out:.0f}% of its nodes have v_r > 0"
        )
        print(
            f"[{'ok  ' if high_in > 0.5 else 'WARN'}] reel-in on the high lobe: "
            f"{100 * high_in:.0f}% of its nodes have v_r < 0"
        )
    topo = path_topology(config["reelout"]["path_parameters"])
    print(f"[{'ok  ' if topo[0] == 1 else 'WARN'}] seed path: {describe_topology(*topo)}")
    seed_height_report(config["reelout"]["path_parameters"], window, "seed")
    reversals = count_steering_reversals(series["input_steering"])
    print(f"[info] seed steering reversals: {reversals} (deadband {STEERING_DEADBAND:g})")
    return ok, topo, series


def optimize_cycle(
    cycle, seed_phase, seed_topology, *, out_dir, prefix, tag, run_plots
):
    """The staged solve on an already-marched ``cycle``; returns
    ``(result, stage_metrics)`` with one (label, metrics) per stage."""
    sim = cycle.pattern_config["sim_parameters"]
    if not sim.get("periodic_wrap") or sim.get("close_radial_cycle"):
        raise ValueError(
            "optimize_cycle expects a config built for the periodic wrap "
            "(build_uploop_config / apply_uploop_formulation)"
        )
    seed_config = copy.deepcopy(cycle.pattern_config)
    result = None
    stage_metrics = []
    # Last pass whose crossings and lobe senses still match the seed.
    clean = None
    topo = seed_topology
    for i, stage in enumerate(STAGES):
        step_bounds = dict(stage.get("step_bounds") or {})
        repeats = int(stage.get("repeat", 1))
        trust_weight = float(stage.get("trust_region_weight", 0.0))
        warm_start = bool(stage.get("warm_start", False))
        aborted = False
        for rep in range(repeats):
            rep_tag = f" (pass {rep + 1}/{repeats})" if repeats > 1 else ""
            print(
                f"\n=== Stage {i}/{len(STAGES) - 1}: {stage['label']}{rep_tag} ===\n"
                f"    params={stage['params']} + depower + v_r profiles "
                f"(target=power, trust_region_weight={trust_weight:g}, "
                f"box={step_bounds})"
            )
            sim["param_step_bound"] = step_bounds
            before = rfco._param_snapshot(cycle.pattern_config, step_bounds)
            try:
                stage_result = cycle.run_simulation_opti(
                    optimization_params=stage["params"],
                    target="power",
                    max_iter=MAX_ITER,
                    trust_region_weight=trust_weight,
                    warm_start=warm_start,
                )
            except RuntimeError as exc:
                print(f"Stage {i} aborted (warm-start simulation failed): {exc}")
                aborted = True
                break
            if stage_result is None:
                print(f"Stage {i}{rep_tag} failed; keeping the last good result.")
                aborted = True
                break
            result = stage_result
            status = ipopt_status(result)
            series = ccp.series_from_optimizer_result(
                result, cycle.pattern_config["path_parameters"]
            )
            metrics = (
                wrapped_cycle_metrics(series, cycle.pattern_config["path_parameters"])
                if series
                else {}
            )
            metrics["ipopt_status"] = status
            metrics.update(hull_bounds_text(sim))
            print(
                f"Stage {i}{rep_tag} done: IPOPT {status}; avg power "
                f"{metrics.get('avg_power', float('nan')) / 1e3:.2f} kW, "
                f"{metrics.get('steering_reversals', 'n/a')} steering reversals, "
                f"seam closure {metrics.get('closure_seam', float('nan')):+.2f} m, "
                f"duration {metrics.get('duration', float('nan')):.1f} s, "
                f"r0 {float(cycle.pattern_config['path_parameters']['r0']):.1f} m"
            )
            topo = check_topology(
                cycle.pattern_config["path_parameters"], seed_topology, "pattern"
            )
            pass_label = f"S{i}" + (f" p{rep + 1}" if repeats > 1 else "")
            pass_file = f"{prefix}_optimized_{tag}_stage{i}"
            if repeats > 1:
                pass_file += f"_pass{rep + 1}"
                result.save_config_to_yaml(out_dir / f"{pass_file}.yaml")
                result.save_trajectory_csv(out_dir / f"{pass_file}.csv")
            if topo == seed_topology:
                clean = (pass_label, metrics, f"{pass_file}.yaml")
            if SPEC["kind"] == "symmetric":
                sym = symmetry_report(cycle.pattern_config["path_parameters"], series)
                metrics.update(sym)
                print(f"    symmetry: {describe_symmetry(sym)}")
            after = rfco._param_snapshot(cycle.pattern_config, step_bounds)
            box_active = rfco._step_box_active(before, after, step_bounds)
            if box_active:
                print("    ** WARNING: backstop step box active at this optimum **")
            if stage.get("repeat_while_box_active") and "r0" in step_bounds:
                # Re-centre while ANY box is active (r0 or a shape box): r0
                # alone leaving its edge would cut the shape's travel budget.
                r0_on_edge = rfco._step_box_active(
                    before, after, {"r0": step_bounds["r0"]}
                )
                print(
                    f"    r0 {'ON' if r0_on_edge else 'off'} its +/-{step_bounds['r0']:g} m "
                    f"step-box edge; boxes {'active' if box_active else 'inactive'}"
                )
                if rep < repeats - 1 and not box_active:
                    print("    no step box active -- stage done")
                    break
                if r0_on_edge and rep == repeats - 1:
                    print(f"    ** WARNING: r0 still on its step-box edge after {repeats} passes **")
            elif rep < repeats - 1 and step_bounds and not box_active:
                print("    trust region inactive (optimum interior) -- skipping remaining passes")
                break
        if result is not None and not aborted:
            result.save_config_to_yaml(out_dir / f"{prefix}_optimized_{tag}_stage{i}.yaml")
            result.save_trajectory_csv(out_dir / f"{prefix}_optimized_{tag}_stage{i}.csv")
            stage_metrics.append((f"S{i}", metrics))
        if aborted:
            break
    if result is None:
        return None, stage_metrics
    if clean is not None:
        stage_metrics.append((f"last clean pass ({clean[0]})", clean[1]))
    if topo != seed_topology:
        if clean is None:
            where = "no pass kept the seed topology"
        else:
            where = (
                f"last pass with the seed topology: {clean[0]} "
                f"({clean[1].get('avg_power', float('nan')) / 1e3:.2f} kW, "
                f"{clean[1].get('steering_reversals', 'n/a')} reversals), "
                f"saved as {clean[2]}"
            )
        print(
            f"\n** FINAL TOPOLOGY DIFFERS FROM THE SEED: {describe_topology(*topo)} "
            f"vs seed {describe_topology(*seed_topology)}; {where} **"
        )
    result.save_config_to_yaml(out_dir / f"{prefix}_optimized_{tag}.yaml")
    result.save_trajectory_csv(out_dir / f"{prefix}_optimized_{tag}.csv")
    try:
        ccp.save_cycle_comparison_plots(
            baseline_phase=seed_phase,
            result=result,
            seed_config=seed_config,
            optimized_config=cycle.pattern_config,
            out_dir=out_dir,
            tag=f"{prefix}_{tag}",
            show=run_plots,
            # closed-cycle quadrature + the seam closure next to r_end - r0
            metrics_fn=lambda s: wrapped_cycle_metrics(
                s, cycle.pattern_config["path_parameters"]
            ),
        )
    except Exception as exc:
        print(f"(cycle comparison plots unavailable: {exc})")
    return result, stage_metrics


def save_summary(cycle, result, seed_path, window, *, title, save_path):
    """The one-page summary of a solved cycle from the NLP's own series."""
    series = ccp.series_from_optimizer_result(
        result, cycle.pattern_config["path_parameters"]
    )
    hw = dict(getattr(cycle.system_model, "hardware_limits", None) or {})
    radial = cycle.pattern_config["radial_parameters"]
    plot_uploop_summary(
        series,
        cycle.pattern_config["path_parameters"],
        seed_path=seed_path,
        height_band=(float(window["min_height_m"]), float(window["max_height_m"])),
        tension_band=(float(radial["min_tether_force"]), float(radial["max_tether_force"])),
        vr_bounds=hw.get("speed_radial"),
        depower_band=fpcc.DEPOWER_BAND,
        steering_limit=float(
            (hw.get("input_steering") or DEFAULT_OPTI_LIMITS["input_steering"])[1]
        ),
        title=title,
        save_path=save_path,
    )



def _loss_suffix(args) -> str:
    """Output-name suffix of the objective variant (none = tether power)."""
    if getattr(args, "drive_losses", False):
        return "_drive"
    return "_friction" if args.winch_friction else ""


def _loss_label(args) -> str:
    if getattr(args, "drive_losses", False):
        return " + drive losses"
    return " + winch friction" if args.winch_friction else ""


def _apply_drive_losses(sim_parameters: dict, args) -> None:
    """Switch the objective from tether power to drive (shaft) power."""
    if args.winch_friction or getattr(args, "drive_losses", False):
        sim_parameters["winch_friction"] = True
    if getattr(args, "drive_losses", False):
        sim_parameters["drivetrain_load_fraction"] = True


def main(args):
    kite = fpcc.configure_kite(args.kite)
    block = dict(kite.get("uploop_eight") or {})
    wind = resolve_wind(args, block)
    rfco.WIND_CONFIG = wind
    shape, window = resolve_shape(block, scale_lobes=args.scale_lobes)
    if args.r0_max is not None:
        window["r0_max_m"] = float(args.r0_max)
    if args.symmetric and args.shape != "symmetric":
        raise SystemExit("--symmetric needs --shape symmetric")
    if args.shape == "symmetric":
        shape, window, az_center = resolve_symmetric_shape(
            dict(kite.get("symmetric_eight") or {}), window, scale_lobes=args.scale_lobes
        )
        if args.r0_max is not None:
            window["r0_max_m"] = float(args.r0_max)
        SPEC.update(
            kind="symmetric", az_center=az_center, topology_by="azimuth",
            mirror=bool(args.symmetric),
        )
        prefix = "symmetric_eight" + ("_mirror" if args.symmetric else "")
    else:
        prefix = "uploop_eight"
    prefix += _loss_suffix(args)
    tag = wind_tag(wind)
    out_dir = Path("results") / kite["name"] / "optimization" / "uploop_eight"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Kite: {kite['name']} ({kite['system_config']})")
    print(
        f"Shape: {SPEC['kind']}"
        + (" with mirror_symmetry rows" if SPEC.get("mirror") else "")
        + f" -> outputs {prefix}_*"
    )
    print(f"Wind solved at: {rfco.describe_wind(wind)}  [tag {tag}]")
    print(
        "Shape: low lobe ({:.1f}, {:.1f}) r {:.1f} deg; high lobe ({:.1f}, {:.1f}) "
        "r {:.1f} deg; sense {:+d}".format(
            *np.degrees([shape[k] for k in SHAPE_KEYS[:6]]), shape["sense"]
        )
    )
    print(f"Window / limits: {window}")

    # --- seed: build, march, recalibrate n_points, re-march
    config, landmarks = build_uploop_config(kite, shape, window)
    phase, system_model = fpcc._simulate_cycle(config, run_plots=False)
    duration = fpcc._measured_cycle_duration(
        phase, config["reelout"]["sim_parameters"]["n_points"]
    )
    if duration is None:
        print("Seed march truncated: the shape is not trimmable at this wind.")
        fpcc._feasibility_report(
            phase, system_model, config,
            artificial=feasibility_window(window, landmarks),
        )
        return 1
    print(f"Measured cycle duration {duration:.1f} s -> recalibrating n_points")
    config, landmarks = build_uploop_config(kite, shape, window, duration_s=duration)
    reelout = config["reelout"]
    _apply_drive_losses(reelout["sim_parameters"], args)
    # The optimizer updates the path IN PLACE (shared nested dict): keep the
    # seed shape for the summary figure.
    seed_path = copy.deepcopy(reelout["path_parameters"])
    cycle, phase = march_seed(reelout, kite, wind, run_plots=args.plot)
    if phase is None:
        return 1
    print(phase.energy_metrics())
    ok, seed_topology, seed_series = report_seed(
        phase, cycle.system_model, config, landmarks, window
    )
    rfco.validate_full_cycle_config(reelout)
    seed_yaml = out_dir / f"{prefix}_seed.yaml"
    with seed_yaml.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"Seed config saved to {seed_yaml}")
    fpcc.plot_seed_path(
        config, save_path=out_dir / f"{prefix}_seed_path.png", show=args.plot
    )
    seed_metrics = wrapped_cycle_metrics(seed_series, reelout["path_parameters"])
    seed_metrics.update(hull_bounds_text(reelout["sim_parameters"]))
    if SPEC["kind"] == "symmetric":
        seed_metrics.update(symmetry_report(reelout["path_parameters"], seed_series))
    columns = [("seed", seed_metrics)]

    if args.plot and not args.yes:
        answer = input("Proceed with the staged optimization? [y/N] ").strip().lower()
        if answer not in ("y", "yes"):
            print("Stopped before stage 0.")
            return 0
    if args.no_optimize:
        print_metrics_table(columns)
        return 0 if ok else 1

    if args.baseline_from_optimum:
        # the continuation run is about the baseline; the eight keeps its files
        print("\n--baseline-from-optimum: the eight is not re-optimised")
        result, stage_metrics = None, []
    else:
        result, stage_metrics = optimize_cycle(
            cycle, phase, seed_topology,
            out_dir=out_dir, prefix=prefix, tag=tag, run_plots=args.plot,
        )
    columns += stage_metrics
    if result is not None:
        label = {
            "uploop_eight": "up-loop eight",
            "symmetric_eight": "symmetric eight (free)",
            "symmetric_eight_mirror": "symmetric eight (mirror rows)",
        }[prefix.removesuffix(_loss_suffix(args)) if _loss_suffix(args) else prefix] + _loss_label(args)
        save_summary(
            cycle, result, seed_path, window,
            title=f"{label} -- {kite['name']}, {rfco.describe_wind(wind)}",
            save_path=out_dir / f"{prefix}_{tag}_summary.png",
        )

    baseline_clipped = 0
    footnote_baseline = None
    # A symmetric run's baseline must not overwrite the slanted run's files.
    b_prefix = "baseline" if SPEC["kind"] == "slanted" else "baseline_symmetric"
    b_root = b_prefix
    b_prefix += _loss_suffix(args)
    if args.baseline_seed:
        print("\n=== Baseline: the classic multi-lobe seed on the same formulation ===")
        baseline = rfco.load_reelout_config(Path(kite["cycle_config_dir"]) / SEED_FILENAME)
        if args.baseline_from_optimum:
            # Continuation: start from the stored tether-power optimum, so a
            # changed objective (drive losses) moves that optimum instead of
            # re-searching from the seed into another local optimum.
            opt_path = out_dir / f"{b_root}_optimized_{tag}.yaml"
            if not opt_path.exists():
                raise FileNotFoundError(
                    f"--baseline-from-optimum needs {opt_path} (run without a loss flag first)"
                )
            baseline = rfco.load_reelout_config(opt_path)
            b_prefix += "_warm"
            print(f"Baseline seeded from the stored optimum {opt_path.name}")
        try:
            if not args.baseline_from_optimum:
                # a stored optimum is already on the formulation; re-fairing
                # its (turn-radius-limited) path would fail the seed curvature cap
                baseline, baseline_clipped = apply_uploop_formulation(baseline, window, kite)
            _apply_drive_losses(baseline["sim_parameters"], args)
        except ValueError as exc:
            # The hull projection / fairing could not produce a seed that
            # honours both the hull and the turn-radius floor: report, keep
            # the eight's results and the table.
            print(f"Baseline seed cannot be put on the shared formulation: {exc}")
            footnote_baseline = f"baseline seed skipped: {exc}"
            b_phase = None
        else:
            footnote_baseline = None
            b_cycle, b_phase = march_seed(baseline, kite, wind, run_plots=False)
            if b_phase is None:
                print("Baseline seed does not trim at this wind; skipped.")
                footnote_baseline = "baseline seed skipped: a node failed to trim"
        if b_phase is not None:
            b_series = ccp.series_from_phase(b_phase)
            b_metrics = wrapped_cycle_metrics(b_series, baseline["path_parameters"])
            b_metrics.update(hull_bounds_text(baseline["sim_parameters"]))
            columns.append(("baseline seed", b_metrics))
            b_topology = path_topology(baseline["path_parameters"])
            print(f"Baseline seed path: {describe_topology(*b_topology)}")
            seed_height_report(baseline["path_parameters"], window, "baseline seed")
            b_seed_path = copy.deepcopy(baseline["path_parameters"])
            b_result, b_stages = optimize_cycle(
                b_cycle, b_phase, b_topology,
                out_dir=out_dir, prefix=b_prefix, tag=tag, run_plots=args.plot,
            )
            columns += [(f"baseline {label}", m) for label, m in b_stages]
            if b_result is not None:
                save_summary(
                    b_cycle, b_result, b_seed_path, window,
                    title=f"baseline cycle -- {kite['name']}, {rfco.describe_wind(wind)}",
                    save_path=out_dir / f"{b_prefix}_{tag}_summary.png",
                )

    footnotes = []
    if args.compare and SPEC["kind"] == "symmetric":
        refs = reference_columns(
            out_dir, tag, override=reelout["sim_parameters"]["opti_limits_override"]
        )
        if refs:
            columns += refs
            footnotes.append(
                "ref columns: the last saved slanted-eight / baseline results with "
                "this wind tag (only the wind is in the filename; their stored "
                "r0 / height / C_beta / depower / AoA limits are checked against "
                "this run's, any mismatch is printed above); IPOPT status not stored"
            )
    if args.baseline_seed and footnote_baseline:
        footnotes.append(footnote_baseline)
    if args.baseline_seed and baseline_clipped:
        footnotes.append(
            f"baseline seed: {baseline_clipped} spline control points were "
            "projected into the shared spline hull before fairing"
        )
    if args.baseline_csv:
        archived = series_from_csv(args.baseline_csv)
        columns.append(("archived*", wrapped_cycle_metrics(archived)))  # s in [0, 1)
        footnotes.append(ARCHIVED_CSV_FOOTNOTE)
    if args.compare or args.baseline_csv:
        print_metrics_table(columns, footnotes)
    print("Up-loop eight optimization complete")
    return 0 if result is not None else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--kite", default=DEFAULT_KITE, help="Kite folder under data/ (or a path)")
    parser.add_argument(
        "--wind-law",
        default=None,
        choices=("logarithmic", "uniform"),
        help="profile law (default logarithmic; other laws need their own "
        "parameters -- put them in the profile's uploop_eight.wind block)",
    )
    parser.add_argument("--wind-speed", type=float, default=None, help="m/s at --wind-height")
    parser.add_argument("--wind-height", type=float, default=None, help="m (default 100)")
    parser.add_argument("--z0", type=float, default=None, help="roughness length m (default 0.03)")
    parser.add_argument("--no-optimize", action="store_true", help="seed + report only")
    parser.add_argument("--plot", action="store_true", help="show the figures")
    parser.add_argument("--yes", action="store_true", help="skip the confirmation before stage 0")
    parser.add_argument(
        "--baseline-seed", action="store_true",
        help="also run the classic multi-lobe seed through the same pipeline",
    )
    parser.add_argument("--compare", action="store_true", help="print the metrics table")
    parser.add_argument(
        "--r0-max", type=float, default=None,
        help="upper bound of r0 [m] (overrides the profile's r0_max_m)",
    )
    parser.add_argument(
        "--shape", choices=("slanted", "symmetric"), default="slanted",
        help="slanted up-loop eight (default) or the level eight climbing at "
        "both sides (profile block symmetric_eight)",
    )
    parser.add_argument(
        "--symmetric", action="store_true",
        help="--shape symmetric only: add the half-period mirror rows "
        "(sim_parameters.mirror_symmetry)",
    )
    parser.add_argument(
        "--baseline-csv", default=None,
        help="archived optimizer trajectory CSV to add to the table (indicative only)",
    )
    parser.add_argument(
        "--scale-lobes", type=float, default=1.0, help="scale both lobe radii"
    )
    parser.add_argument(
        "--winch-friction", action="store_true",
        help="maximise the DRIVE power T v - P_loss(v) with the drum friction of "
        "system.yaml (sim_parameters.winch_friction); outputs get a _friction suffix",
    )
    parser.add_argument(
        "--drive-losses", action="store_true",
        help="maximise the SHAFT power: T v minus the drum friction AND the "
        "load-proportional drivetrain loss of system.yaml (winch_friction + "
        "drivetrain_load_fraction); outputs get a _drive suffix",
    )
    parser.add_argument(
        "--baseline-from-optimum", action="store_true",
        help="with --baseline-seed: start the baseline from its stored "
        "tether-power optimum instead of the seed (objective continuation); "
        "outputs get a further _warm suffix",
    )
    args = parser.parse_args()
    code = main(args)
    if args.plot:
        import matplotlib.pyplot as plt

        plt.show()
    raise SystemExit(code)

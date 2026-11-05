"""The kite-sized seed of a reel-out pattern.

A pattern's starting path is only flyable if every turn is at least as wide as
the kite holds at its steering limit (``system.turn_radius``), and a fixed path
that is flyable on one ROM or at one wind may not be on another. This refits the
path just enough (``kinematics.steering_limited_eight``). Used by the reel-out
server and the reel-out example scripts; a full cycle has two depowers and is
sized in ``examples/4_trajectory_optimization/cycle/fit_periodic_cycle_config.py``.
"""
import logging
import math
from typing import Any, Dict, Optional

import numpy as np

from awetrim.kinematics.parametrized_patterns import create_pattern_from_dict
from awetrim.kinematics.steering_limited_eight import (
    MAX_ELEVATION,
    CourseTurnLimit,
    fit_spline_to_turn_limit,
    spline_turn_geometry,
)
from awetrim.system.turn_radius import steering_limited_turn_radius
from awetrim.system.winch import Winch
from awetrim.utils.defaults import DEFAULT_OPTI_LIMITS

logger = logging.getLogger(__name__)

#: Margins of the kite-sized seed over the steering-limited turn radius, the
#: later ones tried when the cold solve from the earlier ones does not converge.
SEED_TURN_MARGINS = (1.2, 1.4)
#: Clearance [m] of the kite-sized seed above the optimizer's height floor.
SEED_HEIGHT_CLEARANCE_M = 2.0


def size_reelout_seed(
    system_model: Any,
    reelout_config: Dict[str, Any],
    *,
    margin: float = SEED_TURN_MARGINS[0],
    min_turn_radius: Optional[float] = None,
    pattern_limits: Optional[Dict[str, Any]] = None,
    mirror: bool = True,
    self_crossings: int = 1,
) -> bool:
    """Refit the seed path to the turns the kite holds at the model's wind.

    The kite's steering-limited turn radius per course (at the seed's tether
    length, mean elevation and depower, with the winch law) times ``margin``,
    and at least ``min_turn_radius`` [m], bounds the seed spline's curvature;
    the elevation stays within ``pattern_limits`` (degrees), else above the
    seed's own lowest point and the optimizer's height floor. ``mirror``: an
    eight, refitted mirror-symmetric on 16 coefficients; False keeps the path's
    own coefficients and shape, e.g. a helix. ``self_crossings`` is what the
    refit must still have (1 for an eight, 0 for a helix).

    Applied only when the seed is tighter than that somewhere; a failed fit
    keeps the seed. Mutates ``reelout_config`` and returns whether the seed was
    replaced; the path keeps its parameter span.
    """
    seed = reelout_config["path_parameters"]
    sim_parameters = reelout_config.get("sim_parameters", {})
    input_depower = sim_parameters.get("input_depower")
    if (
        reelout_config.get("pattern_type") != "spline_periodic"
        or not seed.get("downloops", True)
        or input_depower is None
    ):
        return False
    r0 = float(seed["r0"])
    # Sizing is optional: whatever goes wrong in it must not fail the caller.
    try:
        el_seed = np.asarray(
            create_pattern_from_dict("spline_periodic", seed).elevation(
                r0, np.linspace(0.0, 1.0, 400, endpoint=False)
            )
        ).ravel()
        table = steering_limited_turn_radius(
            system_model,
            distance_radial=r0,
            angle_elevation=float(np.mean(el_seed)),
            input_depower=float(input_depower),
            winch=Winch(pattern_config=reelout_config["radial_parameters"]),
        )
        floor = float(min_turn_radius or 0.0)

        def required(course, sign):
            return np.fmax(table.radius(course, sign), floor)

        limit = CourseTurnLimit(required, r0=r0, margin=margin)
        course, radius, sign = spline_turn_geometry(seed, r0)
        if np.all(radius >= limit.required_radius(course, sign)):
            return False
        limits = pattern_limits or {}
        el_min = limits.get("elevation_min")
        el_max = limits.get("elevation_max")
        # The optimizer's height row holds at every node: start above it.
        override = sim_parameters.get("opti_limits_override") or {}
        height_min = override.get("height", DEFAULT_OPTI_LIMITS["height"])[0]
        el_floor = max(
            float(el_seed.min()),
            math.asin((float(height_min) + SEED_HEIGHT_CLEARANCE_M) / r0),
        )
        fitted, report = fit_spline_to_turn_limit(
            seed,
            limit,
            **({} if mirror else {"M": None, "mirror": False}),
            el_min=math.radians(el_min) if el_min is not None else el_floor,
            el_max=math.radians(el_max) if el_max is not None else MAX_ELEVATION,
        )
    except Exception as exc:
        logger.warning("kite-sized seed: %s; keeping the configured seed", exc)
        return False
    if not (report["converged"] and report["ratio"] >= 0.99
            and report["self_crossings"] == self_crossings):
        logger.warning(
            "kite-sized seed did not fit (%s, ratio %.2f, %d crossings); "
            "keeping the configured seed",
            report["status"], report["ratio"], report["self_crossings"],
        )
        return False
    logger.info(
        "kite-sized seed: M %d, turns >= %.1f m, azimuth <= %.1f deg, "
        "elevation %.1f-%.1f deg",
        report["M"], report["turn_radius_min_m"],
        math.degrees(report["azimuth_max"]),
        *(math.degrees(e) for e in report["elevation_range"]),
    )
    fitted["s_init"], fitted["s_final"] = seed.get("s_init", 0.0), seed.get("s_final", 1.0)
    reelout_config["path_parameters"] = fitted
    return True

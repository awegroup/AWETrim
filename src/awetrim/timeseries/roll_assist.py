"""Roll-gain assist: start a phase optimisation the kite's own roll gain cannot.

A kite with little roll authority (e.g. the simulation-only LEI-V3 ROM, whose
k(u_p) is ~1.5x below the flight's) often cannot trim the SEED path at its own
roll gain: the forward march needs more steering than the turns allow and stops
at a failed node, so the NLP starts from a truncated, padded seed and does not
converge. The path itself is usually reachable -- the optimizer widens the
turns until the real gain suffices -- it is only the start that is not.

The assist marches the seed at the smallest scale on ``ROLL_ASSIST_SCALES``
that trims it (``Kite.gain_roll_scale``, a run-time scale on the whole k(u_p),
not a ROM parameter), solves there, and then halves the excess back to 1, each
solve warm-started from the previous optimum; a failed step is retried from
its midpoint, at most ``ROLL_ASSIST_MAX_HALVINGS`` times. The result is always
the kite's own optimum at scale 1. A seed that trims at scale 1 is unaffected.

Used by the full-cycle scripts (examples/4_trajectory_optimization/cycle/) and
the reel-out server's cold solve (awetrim.server.session).
"""
from typing import Any, Callable, List, Optional, Tuple

import numpy as np

ROLL_ASSIST_SCALES = (1.0, 1.25, 1.5, 1.75, 2.0)
ROLL_ASSIST_MIN_STEP = 0.05
ROLL_ASSIST_MAX_HALVINGS = 4


def fully_trimmed(phase, n_points: int) -> bool:
    """True if the marched ``phase`` covers all ``n_points`` nodes."""
    return phase is not None and np.asarray(phase.return_variable("s")).size >= n_points


def march_with_roll_assist(
    kite,
    make_phase: Callable[[], Any],
    n_points: int,
    *,
    scales=None,
    **run_kwargs,
) -> Tuple[Any, Any, Optional[float], Optional[Exception]]:
    """Forward-simulate a fresh ``make_phase()`` at the smallest roll-gain
    scale that trims it.

    Returns ``(cycle, phase, scale, failure)``: the Phase object (``kite`` left
    at that scale), the simulated phase, the scale, and the error of the
    scale-1 attempt (None if it succeeded). When no scale trims it, ``cycle``
    and ``phase`` are the attempt that got furthest (``None`` if all raised;
    ``kite`` left at its scale, which the returned ``reached_scale`` attribute
    of ``phase`` records) and ``scale`` is None. Not monotonic in the scale:
    too much roll gain can lose the first node's root, so every scale is
    tried. ``run_kwargs`` go to ``Phase.run_simulation`` (``phase_sim=True``
    always); a strict config (``require_full_trajectory``) raises on a failed
    node, a non-strict one returns a truncated phase.
    """
    best, first_failure = (None, None, None, -1), None
    for scale in scales or ROLL_ASSIST_SCALES:
        kite.gain_roll_scale = float(scale)
        cycle = make_phase()
        try:
            phase, _ = cycle.run_simulation(phase_sim=True, **run_kwargs)
        except RuntimeError as exc:
            print(f"  roll gain x {scale:g}: march failed ({exc})")
            first_failure = first_failure or exc
            continue
        if fully_trimmed(phase, n_points):
            if scale != 1.0:
                print(f"Roll-gain assist: the seed trims at {scale:g} x k(u_p) "
                      f"(not at 1: {first_failure}); the optimizer continues it back to 1")
            return cycle, phase, float(scale), first_failure
        reached = int(np.asarray(phase.return_variable("s")).size)
        print(f"  roll gain x {scale:g}: trim truncated at node {reached}")
        first_failure = first_failure or RuntimeError(f"trim truncated at node {reached}")
        if reached > best[3]:
            best = (cycle, phase, float(scale), reached)
    cycle, phase, scale, _ = best
    kite.gain_roll_scale = scale or 1.0
    if phase is not None:
        phase.reached_scale = scale
    return cycle, phase, None, first_failure


def continuation_scales(assist: Optional[float]) -> List[float]:
    """The scales after ``assist``: the excess over 1 halved each step, 1 once
    it is below ``ROLL_ASSIST_MIN_STEP`` (e.g. 2 -> [1.5, 1.25, 1.125, 1.0625, 1])."""
    scales = []
    scale = float(assist or 1.0)
    while scale > 1.0:
        scale = 1.0 + (scale - 1.0) / 2.0
        if scale - 1.0 < ROLL_ASSIST_MIN_STEP:
            scale = 1.0
        scales.append(scale)
    return scales


def continue_roll_gain(kite, assist: float, solve: Callable[[], Any]) -> Any:
    """Solve at each of ``continuation_scales(assist)`` with ``solve()`` (one
    warm-started solve at ``kite``'s current scale; None on failure). A failed
    step is retried after a solve at the midpoint from the last solved scale,
    at most ``ROLL_ASSIST_MAX_HALVINGS`` times per step. Returns the scale-1
    result, or None when a step still fails. ``kite`` is left at scale 1
    either way, so a caller never keeps an assisted model."""
    previous, result = float(assist), None
    try:
        for target in continuation_scales(assist):
            halvings, stack = 0, [target]
            while stack:
                scale = stack[-1]
                kite.gain_roll_scale = scale
                print(f"Roll-gain continuation: k(u_p) x {scale:g}")
                result = solve()
                if result is None:
                    if halvings >= ROLL_ASSIST_MAX_HALVINGS:
                        return None
                    halvings += 1
                    mid = 0.5 * (previous + scale)
                    print(f"    continuation step failed; retrying from the midpoint x {mid:g}")
                    stack.append(mid)
                    continue
                previous = stack.pop()
        return result
    finally:
        kite.gain_roll_scale = 1.0

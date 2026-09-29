# Copyright (c) 2023-2026 Oriol Cayon, Delft University of Technology
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""Convergence and timestep helpers for aerostructural coupling.

Convergence of the coupled solver is assessed on the GLOBAL NODAL FORCE
RESIDUAL, the sum of the internal structural forces and the applied external
forces,

    f_res = f_int + f_ext ,

with the fixed nodes zeroed. That norm can be judged two ways:

* RELATIVE (preferred) -- ``||f_res|| / F_tether`` against
  ``residual_tol_relative``, where ``F_tether = |sum f_ext|`` is the reaction
  the constrained bridle node carries. The measure is dimensionless, so one
  threshold means the same thing at 2 kN and at 8 kN of tether load.
* ABSOLUTE (legacy) -- ``||f_res||`` in newtons against ``tol``. One tolerance
  then means very different things across a sweep: measured on LEI-V3, the
  ``tol: 5`` N gate is 6.8e-4 relative at 7.4 kN but 1.3e-3 at 3.9 kN, loose
  enough that two solves of the SAME point reached CL 5-15% apart depending
  only on which side the iteration approached from.

Which one is active is decided by ``resolve_residual_tolerances``: configure
``residual_tol_relative`` to get the relative criterion, otherwise the absolute
``tol`` stays in force. The two are never silently interchanged -- a ``tol: 5``
read as a ratio would "converge" on iteration 1.

``resultant_tether_force`` is the single place the normalising force is
defined; ``relative_residual_norm`` the single place the ratio is formed.
"""

from __future__ import annotations

import logging

import numpy as np

# Dimensionless defaults for the coupled loop (see module docstring).
DEFAULT_RESIDUAL_TOL_RELATIVE = 1.0e-4
DEFAULT_STAGNATION_TOL_RELATIVE = 4.0e-5

#: Below this the normalising force carries no information and the ratio is
#: reported as NaN rather than exploding.
_MIN_FORCE_REFERENCE_N = 1.0e-9


def resultant_tether_force(external_force) -> float:
    """Resultant tether force [N] the shape hangs from.

    The spring forces are internal and pairwise self-equilibrated, so summing
    the nodal balance ``f_int + f_ext + f_reaction = 0`` over every node leaves
    ``|f_reaction| = |sum f_ext|``: the whole applied load (aerodynamic,
    gravity, inertial) is carried by the constrained bridle/KCU node, i.e. by
    the tether. This resultant is the reference the residual is normalised by.

    Args:
        external_force: nodal external forces, ``(n_nodes, 3)`` or flattened.

    Returns:
        Euclidean norm of the resultant external force [N].
    """
    forces = np.asarray(external_force, dtype=float).reshape(-1, 3)
    return float(np.linalg.norm(forces.sum(axis=0)))


def relative_residual_norm(residual, force_reference: float) -> float:
    """``||f_int + f_ext|| / F_tether`` -- the coupled convergence measure [-].

    Returns NaN when the normalising force is unusable (non-finite or ~0), so
    callers can tell "not converged" from "criterion undefined".
    """
    residual_norm = float(
        np.linalg.norm(np.atleast_1d(np.asarray(residual, dtype=float)))
    )
    reference = float(force_reference)
    if not np.isfinite(reference) or reference <= _MIN_FORCE_REFERENCE_N:
        return float("nan")
    return residual_norm / reference


def resolve_residual_tolerances(solver_config) -> tuple[float, float, bool]:
    """Pick the convergence measure and its tolerances from the solver config.

    ``residual_tol_relative`` selects the dimensionless criterion; without it
    the legacy absolute ``tol`` [N] stays in force. Returns
    ``(residual_tol, stagnation_tol, is_relative)`` so every consumer of the
    tolerance -- convergence, stagnation, adaptive dt, the stiffness trigger --
    reads the same number in the same units.
    """
    relative = solver_config.get("residual_tol_relative")
    if relative is None:
        return (
            float(solver_config["tol"]),
            float(solver_config["stagnation_tol"]),
            False,
        )
    residual_tol = float(relative)
    if not np.isfinite(residual_tol) or residual_tol <= 0.0:
        raise ValueError(
            f"residual_tol_relative must be a positive ratio, got {relative!r}"
        )
    if residual_tol >= 1.0:
        # A newton value pasted into the relative key would converge instantly.
        raise ValueError(
            f"residual_tol_relative is a RATIO, not a force [N]; got {relative!r}"
        )
    stagnation_tol = float(
        solver_config.get(
            "stagnation_tol_relative", DEFAULT_STAGNATION_TOL_RELATIVE
        )
    )
    return residual_tol, stagnation_tol, True


def resolve_fallback_tolerance(solver_config, residual_tol: float) -> float:
    """Looser tolerance a PLATEAUED solve may still be accepted at [-] or [N].

    Some conditions do not have a fixed point the iteration can reach. A
    steered, near-stall or heavily loaded case settles into a LIMIT CYCLE
    instead: measured on LEI-V3 at u_dp 0.18 / u_s 0.1 tetherless, the relative
    residual repeats a period-4 orbit between 7.2e-4 and 1.0e-3 from iteration
    ~40 to 100, with the Aitken factor already pinned at its 0.05 floor -- so
    there is no damping left to add and more iterations cannot help.

    Rejecting those points loses whole rows of a sweep; silently loosening the
    tolerance for everyone reintroduces the path-dependence the relative
    criterion exists to remove. So they are accepted at a SEPARATE, looser
    tolerance and flagged, with the cycle's own span reported alongside as the
    honest uncertainty on the point.

    Defaults to ``fallback_factor`` (10) times the main tolerance. Returning
    something <= residual_tol disables the fallback.
    """
    explicit = solver_config.get("residual_tol_relative_fallback")
    if explicit is None:
        explicit = solver_config.get("residual_tol_fallback")
    if explicit is not None:
        return float(explicit)
    factor = float(solver_config.get("residual_tol_fallback_factor", 10.0))
    if not np.isfinite(factor) or factor <= 1.0:
        return float(residual_tol)
    return float(residual_tol) * factor


def element_elongations(
    struc_nodes,
    connectivity,
    rest_lengths,
    *,
    pulley_pairs=None,
):
    """Per-element elongation ``(l - l0) / l0`` [-], in element order.

    Pulleys are the subtlety. PSS represents a pulley rope as TWO elements, one
    per arm, each naming the other; the pulley slides until the tension in both
    arms is equal, which means an individual arm's length change carries no
    information -- the arm can lengthen purely because the pulley slid. What is
    physically strained is the ROPE:

        (l_a + l_b - l0_a - l0_b) / (l0_a + l0_b)

    so both arms of a pulley report that shared value. Treating the arms as
    independent elements is how a pulley arm acquires a large spurious
    "elongation" and gets stiffened for it.

    Args:
        struc_nodes: (n_nodes, 3) node positions.
        connectivity: (n_elements, >=2) node index pairs, in element order.
        rest_lengths: per-element rest length [m], same order.
        pulley_pairs: the solver's ``pulley_line_to_other_node_pair_dict`` --
            ``{str(element_index): [node_a, node_b, l0_other, l0_self, ci]}``,
            naming the OTHER arm of the rope this element is half of.

    Returns:
        (n_elements,) array of elongations; NaN where the rest length is not
        positive.
    """
    nodes = np.asarray(struc_nodes, dtype=float)
    conn = np.asarray(connectivity, dtype=int)
    l0 = np.asarray(rest_lengths, dtype=float).reshape(-1)
    n_elements = min(len(conn), len(l0))
    if n_elements == 0:
        return np.zeros(0, dtype=float)

    pairs = conn[:n_elements, :2]
    lengths = np.linalg.norm(nodes[pairs[:, 1]] - nodes[pairs[:, 0]], axis=1)
    elongations = np.full(n_elements, np.nan, dtype=float)
    positive = l0[:n_elements] > 0.0
    elongations[positive] = lengths[positive] / l0[:n_elements][positive] - 1.0

    for key, value in (pulley_pairs or {}).items():
        index = int(key)
        if not 0 <= index < n_elements:
            continue
        other = np.asarray(value, dtype=float).reshape(-1)
        node_a, node_b = int(other[0]), int(other[1])
        l0_other, l0_self = float(other[2]), float(other[3])
        l0_rope = l0_self + l0_other
        if l0_rope <= 0.0:
            continue
        l_other = float(np.linalg.norm(nodes[node_b] - nodes[node_a]))
        elongations[index] = (lengths[index] + l_other) / l0_rope - 1.0

    return elongations


def max_element_elongation(
    struc_nodes,
    connectivity,
    rest_lengths,
    *,
    pulley_pairs=None,
):
    """Largest element elongation [-], NaN-safe. See element_elongations."""
    elongations = element_elongations(
        struc_nodes, connectivity, rest_lengths, pulley_pairs=pulley_pairs
    )
    if elongations.size == 0 or not np.any(np.isfinite(elongations)):
        return float("nan")
    return float(np.nanmax(elongations))


def compute_adaptive_dt(
    residual_norm_history,
    dt_initial: float,
    dt_max: float,
    residual_tol: float,
) -> float:
    """Increase PSS timestep as the residual approaches convergence."""
    if len(residual_norm_history) < 1:
        return dt_initial

    current_residual = residual_norm_history[-1]
    if (
        residual_tol is None
        or residual_tol <= 0
        or not np.isfinite(residual_tol)
        or not np.isfinite(current_residual)
        or current_residual < 0
    ):
        return dt_initial

    ratio = np.clip(residual_tol / max(current_residual, 1e-12), 0.0, 1.0)
    dt_adaptive = dt_initial + (dt_max - dt_initial) * ratio
    return float(np.clip(dt_adaptive, min(dt_initial, dt_max), max(dt_initial, dt_max)))


def check_convergence(
    *,
    iteration: int,
    residual: np.ndarray,
    residual_norm_history,
    aero_forces_vsm_format: np.ndarray,
    solver_config: dict,
    is_run_only_1_time_step: bool,
    stagnation_check_start: int = 0,
    force_reference: float | None = None,
) -> tuple[bool, bool, bool]:
    """Return convergence, break, and stagnation flags for the coupling loop.

    ``residual_norm_history`` must already be in the ACTIVE measure (see
    ``resolve_residual_tolerances``); ``force_reference`` is the resultant
    tether force [N] and is required by the relative criterion.
    """
    is_convergence = False
    should_break = False
    is_stagnated = False

    residual_tol, stagnation_tol, is_relative = resolve_residual_tolerances(
        solver_config
    )

    residual_norm_absolute = float(np.linalg.norm(residual))
    if is_relative:
        if force_reference is None:
            raise ValueError(
                "residual_tol_relative is configured but no force_reference "
                "was passed to check_convergence"
            )
        residual_norm = relative_residual_norm(residual, force_reference)
    else:
        residual_norm = residual_norm_absolute
    n_stag = solver_config["n_max_constant_residual_force"]
    iters_since_start = iteration - stagnation_check_start

    if np.isnan(residual_norm_absolute):
        logging.info("Classic PS diverged - residual force is NaN")
        should_break = True
    elif not np.isfinite(residual_norm):
        # The residual itself is finite, so this is a broken NORMALISATION
        # (zero or NaN resultant external force). Do not claim convergence and
        # do not kill the run; the aero-NaN and max-iteration guards remain.
        logging.warning(
            "Coupled convergence undefined - resultant tether force is %s N; "
            "residual %.3f N left unnormalised this iteration",
            force_reference,
            residual_norm_absolute,
        )
    elif residual_norm <= residual_tol:
        is_convergence = True
    elif iters_since_start >= n_stag and n_stag > 0:
        window_vals = np.asarray(
            residual_norm_history[iteration - n_stag : iteration + 1], dtype=float
        )
        if window_vals.size > 0 and np.isfinite(window_vals).all():
            residual_span = float(np.max(window_vals) - np.min(window_vals))
            if residual_span < stagnation_tol:
                is_stagnated = True
    elif iteration > solver_config["max_iter"]:
        logging.info(
            "Classic PS non-converging - more than max (%s) iterations needed",
            solver_config["max_iter"],
        )
        should_break = True
    elif is_run_only_1_time_step:
        should_break = True
    elif np.isnan(np.sum(np.asarray(aero_forces_vsm_format)[:, 1])):
        logging.info("Classic PS non-converging - aero forces are NaN")
        should_break = True

    return is_convergence, should_break, is_stagnated


__all__ = [
    "DEFAULT_RESIDUAL_TOL_RELATIVE",
    "DEFAULT_STAGNATION_TOL_RELATIVE",
    "check_convergence",
    "compute_adaptive_dt",
    "element_elongations",
    "max_element_elongation",
    "relative_residual_norm",
    "resolve_fallback_tolerance",
    "resolve_residual_tolerances",
    "resultant_tether_force",
]

def remaining_drift(values, n_ratios=3):
    """Estimated change still to come in a converging sequence [its unit].

    Models the tail as geometric: with successive changes ``d_j`` and ratio
    ``r = |d_j / d_(j-1)|``, what remains after the last value is
    ``|d_last| r / (1 - r)``. The ratio is the LARGEST of the last
    ``n_ratios``, because two modes decaying at different rates fool a single
    ratio: a fast transient dying away onto a slow tail reads as r ~ 0 for one
    iteration (measured on the steered Billow kite: a 0.0007 rad/s course-rate
    change with 0.02 rad/s still to come). A growing or non-decaying tail
    (r >= 1), or too short a history, is infinitely far from settled.
    """
    values = np.asarray(values, dtype=float)
    if values.size < n_ratios + 2 or not np.all(np.isfinite(values)):
        return float("inf")
    changes = np.abs(np.diff(values[-(n_ratios + 2):]))
    if changes[-1] == 0.0:
        return 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = np.where(changes[:-1] > 0.0, changes[1:] / changes[:-1], np.inf)
    ratio = float(ratios.max())
    if ratio >= 1.0:
        return float("inf")
    return float(changes[-1] * ratio / (1.0 - ratio))

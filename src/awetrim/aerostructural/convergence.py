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

and the solve is converged once the Euclidean norm of that residual,
NORMALISED BY THE RESULTANT TETHER FORCE, falls below
``residual_tol_relative`` (default 1e-4). The criterion is therefore
dimensionless: the same threshold means the same thing at 2 kN and at 8 kN of
tether load, unlike the absolute force tolerance [N] the loop used before
2026-08-27.

``resultant_tether_force`` is the single place the normalising force is
defined; ``relative_residual_norm`` the single place the ratio is formed.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping

import numpy as np

# Dimensionless defaults for the coupled loop (see module docstring).
DEFAULT_RESIDUAL_TOL_RELATIVE = 1.0e-4
DEFAULT_STAGNATION_TOL_RELATIVE = 4.0e-5

# Absolute force tolerances [N] the loop used before the criterion was
# normalised. They are rejected on sight rather than reinterpreted: a
# ``tol: 5`` read as a RELATIVE tolerance would "converge" on iteration 1.
_LEGACY_TOLERANCE_KEYS = {
    "tol": "residual_tol_relative",
    "stagnation_tol": "stagnation_tol_relative",
}

# Below this the normalising force carries no information (an unloaded or
# NaN-poisoned external force field), so the ratio is undefined.
_MIN_FORCE_REFERENCE_N = 1.0e-9


def resolve_residual_tolerances(
    solver_config: Mapping[str, Any],
) -> tuple[float, float]:
    """Return the (residual, stagnation) RELATIVE tolerances of the coupled loop.

    Raises ValueError if the config still carries the legacy absolute-force
    keys, which would otherwise be silently read as dimensionless ratios.
    """
    legacy = sorted(key for key in _LEGACY_TOLERANCE_KEYS if key in solver_config)
    if legacy:
        replacements = ", ".join(
            f"`{key}` -> `{_LEGACY_TOLERANCE_KEYS[key]}`" for key in legacy
        )
        raise ValueError(
            "aero_structural_solver still carries the legacy ABSOLUTE force "
            f"tolerance(s) {legacy} [N]. The coupled loop now converges on the "
            "nodal force residual ||f_int + f_ext|| normalised by the resultant "
            "tether force, so these keys are dimensionless ratios: "
            f"{replacements} (e.g. `tol: 5` -> `residual_tol_relative: 1.0e-4`, "
            "`stagnation_tol: 2` -> `stagnation_tol_relative: 4.0e-5`)."
        )
    tol = float(
        solver_config.get("residual_tol_relative", DEFAULT_RESIDUAL_TOL_RELATIVE)
    )
    stagnation_tol = float(
        solver_config.get("stagnation_tol_relative", DEFAULT_STAGNATION_TOL_RELATIVE)
    )
    return tol, stagnation_tol


def resultant_tether_force(external_force: np.ndarray) -> float:
    """Resultant tether force [N] the shape hangs from.

    The spring forces are internal and pairwise self-equilibrated, so summing
    the nodal balance ``f_int + f_ext + f_reaction = 0`` over every node leaves
    ``|f_reaction| = |sum f_ext|``: the whole applied load (aerodynamic,
    gravity, inertial) is carried by the single constrained bridle/KCU node,
    i.e. by the tether. This resultant is the reference the nodal force
    residual is normalised by.

    Args:
        external_force: nodal external forces, ``(n_nodes, 3)`` or flattened.

    Returns:
        Euclidean norm of the resultant external force [N].
    """
    forces = np.asarray(external_force, dtype=float).reshape(-1, 3)
    return float(np.linalg.norm(forces.sum(axis=0)))


def relative_residual_norm(residual: np.ndarray, force_reference: float) -> float:
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


def element_elongations(
    nodes: np.ndarray,
    connectivity: np.ndarray,
    rest_lengths: np.ndarray,
    *,
    pulley_pairs: Mapping[Any, Any] | None = None,
) -> np.ndarray:
    """Elongation ``(l - l0) / l0`` [-] of every structural element.

    This is the ONLY place that strain measure is formed: the coupled loop
    reports it and the PSM stiffness update acts on it, so both must read the
    same number.

    A PULLEY arm's own rest length is not an unstretched material length -- the
    two arms trade rope through the pulley, so an arm can lengthen with no
    strain in the rope. Given ``pulley_pairs`` (the driver's
    ``pulley_line_to_other_node_pair_dict``: ``{element_index: [node_a, node_b,
    l0_other, l0_own, node_ci]}``), both arms of a pulley get the elongation of
    the ROPE, ``(l_own + l_other - l0_own - l0_other) / (l0_own + l0_other)``.

    Args:
        nodes: node positions ``(n_nodes, 3)``.
        connectivity: rows ``(node_i, node_j, ...)`` in element order.
        rest_lengths: element rest lengths ``l0`` in the same order.
        pulley_pairs: optional pulley element map (see above).

    Returns:
        Elongations per element; NaN where ``l0`` is not positive.
    """
    positions = np.asarray(nodes, dtype=float).reshape(-1, 3)
    # Rows carry (node_i, node_j, ...) with trailing material/type entries, so
    # take the two indices per row rather than casting the whole table.
    pairs = np.array(
        [(int(row[0]), int(row[1])) for row in connectivity], dtype=int
    ).reshape(-1, 2)
    l0 = np.asarray(rest_lengths, dtype=float).reshape(-1)
    n_elements = min(len(pairs), len(l0))
    if n_elements == 0:
        return np.empty(0, dtype=float)
    pairs = pairs[:n_elements]
    l0 = l0[:n_elements]

    lengths = np.linalg.norm(positions[pairs[:, 0]] - positions[pairs[:, 1]], axis=1)
    length_total = lengths.copy()
    l0_total = l0.copy()
    for key, value in (pulley_pairs or {}).items():
        index = int(key)
        if not 0 <= index < n_elements:
            continue
        node_a, node_b = int(value[0]), int(value[1])
        l0_other = float(value[2])
        length_total[index] = lengths[index] + float(
            np.linalg.norm(positions[node_a] - positions[node_b])
        )
        l0_total[index] = l0[index] + l0_other

    elongations = np.full(n_elements, np.nan, dtype=float)
    valid = l0_total > 0.0
    elongations[valid] = (length_total[valid] - l0_total[valid]) / l0_total[valid]
    return elongations


def max_element_elongation(
    nodes: np.ndarray,
    connectivity: np.ndarray,
    rest_lengths: np.ndarray,
    *,
    pulley_pairs: Mapping[Any, Any] | None = None,
) -> float:
    """Largest element elongation ``(l - l0) / l0`` [-] of the current shape.

    The element stiffnesses are chosen so that every elongation stays below 1%
    (following Poland & Schmehl); this reports what the shape actually reached,
    so that assumption stays auditable per solve.
    """
    elongations = element_elongations(
        nodes, connectivity, rest_lengths, pulley_pairs=pulley_pairs
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
    """Increase PSS timestep as the residual approaches convergence.

    ``residual_norm_history`` and ``residual_tol`` are both the RELATIVE
    residual measure (see module docstring); mixing an absolute history with a
    relative tolerance would pin dt at ``dt_max`` from the first iteration.
    """
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
    force_reference: float,
    aero_forces_vsm_format: np.ndarray,
    solver_config: dict,
    is_run_only_1_time_step: bool,
    stagnation_check_start: int = 0,
) -> tuple[bool, bool, bool]:
    """Return convergence, break, and stagnation flags for the coupling loop.

    Args:
        iteration: current coupled iteration index.
        residual: nodal force residual ``f_int + f_ext`` (fixed nodes zeroed).
        residual_norm_history: RELATIVE residual norms of every iteration so far.
        force_reference: resultant tether force [N], from
            ``resultant_tether_force``.
        aero_forces_vsm_format: panel forces, checked for NaN.
        solver_config: the ``aero_structural_solver`` block; its tolerances
            ``residual_tol_relative`` / ``stagnation_tol_relative`` are
            dimensionless.
        is_run_only_1_time_step: stop after a single coupled iteration.
        stagnation_check_start: iteration the current phase started at.
    """
    is_convergence = False
    should_break = False
    is_stagnated = False

    residual_tol, stagnation_tol = resolve_residual_tolerances(solver_config)

    residual_norm_absolute = float(
        np.linalg.norm(np.atleast_1d(np.asarray(residual, dtype=float)))
    )
    residual_norm = relative_residual_norm(residual, force_reference)
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
    "resolve_residual_tolerances",
    "resultant_tether_force",
]

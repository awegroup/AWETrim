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

"""Convergence and timestep helpers for aerostructural coupling."""

from __future__ import annotations

import logging

import numpy as np


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
) -> tuple[bool, bool, bool]:
    """Return convergence, break, and stagnation flags for the coupling loop."""
    is_convergence = False
    should_break = False
    is_stagnated = False

    residual_norm = np.linalg.norm(residual)
    n_stag = solver_config["n_max_constant_residual_force"]
    iters_since_start = iteration - stagnation_check_start

    if residual_norm <= solver_config["tol"]:
        is_convergence = True
    elif np.isnan(residual_norm):
        logging.info("Classic PS diverged - residual force is NaN")
        should_break = True
    elif iters_since_start >= n_stag and n_stag > 0:
        window_vals = np.asarray(
            residual_norm_history[iteration - n_stag : iteration + 1], dtype=float
        )
        if window_vals.size > 0 and np.isfinite(window_vals).all():
            residual_span = float(np.max(window_vals) - np.min(window_vals))
            if residual_span < solver_config["stagnation_tol"]:
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


__all__ = ["check_convergence", "compute_adaptive_dt"]

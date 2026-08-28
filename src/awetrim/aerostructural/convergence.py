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

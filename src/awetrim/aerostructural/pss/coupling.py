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

"""Protocol-oriented PSS/QSM aero-structural coupling."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from ..mapping import (
    BilinearAeroToStructuralLoadMapper,
    LinearStructuralToAeroMapper,
    check_moment_preservation,
)
from ..protocols import (
    AeroToStructuralLoadMapper,
    PssStructuralSolver,
    QsmCouplingRequest,
    QsmCouplingResult,
    QsmIterationRecord,
    StructuralToAeroMapper,
    TapeActuationState,
)


class PssKineticDampingSolver:
    """Placeholder adapter name for the package API.

    The production scripts still call the legacy PSS functions directly. This
    class reserves the public adapter name used by the protocol-level tests.
    """


class PssQsmCoupler:
    """Fixed-point coupler between a PSS structural solver and QSM trim solver."""

    def __init__(
        self,
        *,
        structural_solver: PssStructuralSolver,
        trim_solver: Callable[..., tuple[dict[str, Any], Any]],
        structural_to_aero_mapper: StructuralToAeroMapper | None = None,
        aero_to_structural_load_mapper: AeroToStructuralLoadMapper | None = None,
    ) -> None:
        self.structural_solver = structural_solver
        self.trim_solver = trim_solver
        self.structural_to_aero_mapper = (
            structural_to_aero_mapper or LinearStructuralToAeroMapper()
        )
        self.aero_to_structural_load_mapper = (
            aero_to_structural_load_mapper or BilinearAeroToStructuralLoadMapper()
        )

    def solve(self, request: QsmCouplingRequest) -> QsmCouplingResult:
        geometry = request.structural_geometry
        settings = request.settings
        system = self.structural_solver.instantiate(
            geometry, settings.pss_settings or {}
        )

        nodes = np.asarray(geometry.nodes, dtype=float).copy()
        mapping = self.aero_to_structural_load_mapper.initialize(
            request.body_aero.panels,
            nodes,
            geometry.le_node_indices,
            geometry.te_node_indices,
        )

        final_residual = np.full(nodes.size, np.nan)
        final_nodal_forces = np.zeros_like(nodes)
        trim_result: dict[str, Any] = {}
        iteration_records: list[QsmIterationRecord] = []

        for iteration in range(settings.max_iter):
            if request.actuation is not None:
                self._apply_actuation(
                    system,
                    request.actuation,
                    iteration=iteration,
                    steering_interval=settings.steering_actuation_interval_iters,
                    power_interval=settings.power_tape_actuation_interval_iters,
                )

            aero_update = self.structural_to_aero_mapper.map(
                nodes,
                geometry.le_node_indices,
                geometry.te_node_indices,
                settings.n_aero_panels_per_structural_section,
            )
            request.body_aero.update_from_points(
                aero_update.leading_edge_points,
                aero_update.trailing_edge_points,
                aero_input_type=settings.aero_input_type,
                initial_polar_data=request.initial_polar_data,
            )

            trim_result, body_aero = self.trim_solver(
                system_model=request.system_model,
                body_aero=request.body_aero,
                vsm_solver=request.vsm_solver,
                center_of_gravity=request.center_of_gravity,
                reference_point=request.reference_point,
                x_guess=request.x_guess,
                bounds_lower=request.bounds_lower,
                bounds_upper=request.bounds_upper,
                moment_tolerance=settings.moment_tolerance,
            )

            panel_forces = np.asarray(trim_result.get("F_distribution", []), dtype=float)
            panel_points = np.asarray(
                trim_result.get("panel_cp_locations", []), dtype=float
            )
            if panel_forces.size == 0:
                panel_forces = np.zeros((len(body_aero.panels), 3))
            if panel_points.size == 0:
                panel_points = np.asarray(
                    [panel.aerodynamic_center for panel in body_aero.panels],
                    dtype=float,
                )

            nodal_forces = self.aero_to_structural_load_mapper.map_loads(
                panel_forces, panel_points, nodes, mapping
            )
            external_force = nodal_forces.reshape(-1)
            system, structural_converged, nodes, internal_force = (
                self.structural_solver.solve(system, external_force)
            )

            residual = np.asarray(internal_force, dtype=float).reshape(-1) + external_force
            residual = self._residual_without_fixed_nodes(
                residual, geometry.fixed_node_indices
            )
            final_residual = residual
            final_nodal_forces = nodal_forces
            residual_norm = float(np.linalg.norm(residual))

            record = QsmIterationRecord(
                iteration=iteration,
                residual_norm=residual_norm,
                structural_converged=bool(structural_converged),
                trim_success=bool(trim_result.get("success", False)),
                trim_success_physical=bool(
                    trim_result.get("success_physical", False)
                ),
                opt_x=np.asarray(trim_result.get("opt_x", []), dtype=float),
                total_aero_force=np.sum(nodal_forces, axis=0),
                total_inertial_force=np.asarray(
                    trim_result.get("inertial_force", np.zeros(3)), dtype=float
                ),
                total_gravity_force=np.asarray(
                    trim_result.get("gravity_force", np.zeros(3)), dtype=float
                ),
                max_node_displacement=0.0,
            )
            iteration_records.append(record)

            if residual_norm <= settings.residual_tolerance:
                break

        moment_report = check_moment_preservation(
            panel_forces, panel_points, final_nodal_forces, nodes, request.reference_point
        )
        metadata = {
            **trim_result,
            "moment_preservation": moment_report,
        }

        return QsmCouplingResult(
            converged=float(np.linalg.norm(final_residual)) <= settings.residual_tolerance,
            final_nodes=np.asarray(nodes, dtype=float),
            final_rest_lengths=np.asarray(system.extract_rest_length, dtype=float),
            final_nodal_forces=final_nodal_forces,
            final_residual=final_residual,
            trim_result=trim_result,
            iteration_records=iteration_records,
            metadata=metadata,
        )

    @staticmethod
    def _residual_without_fixed_nodes(
        residual: np.ndarray, fixed_node_indices: list[int] | tuple[int, ...]
    ) -> np.ndarray:
        cleaned = np.asarray(residual, dtype=float).copy()
        for node_idx in fixed_node_indices:
            start = 3 * int(node_idx)
            cleaned[start : start + 3] = 0.0
        return cleaned

    @staticmethod
    def _apply_actuation(
        system: Any,
        actuation: TapeActuationState,
        *,
        iteration: int,
        steering_interval: int,
        power_interval: int,
    ) -> None:
        rest_lengths = np.asarray(system.extract_rest_length, dtype=float)

        if (
            actuation.power_tape_index is not None
            and actuation.initial_power_tape_length is not None
            and power_interval > 0
            and iteration % power_interval == 0
        ):
            extension = _bounded_increment(
                actuation.power_tape_final_extension,
                actuation.power_tape_extension_step,
                iteration,
                power_interval,
            )
            idx = int(actuation.power_tape_index)
            target = float(actuation.initial_power_tape_length) + extension
            system.update_rest_length(idx, target - float(rest_lengths[idx]))

        if (
            actuation.steering_tape_indices is not None
            and actuation.initial_steering_left_length is not None
            and actuation.initial_steering_right_length is not None
            and steering_interval > 0
            and iteration % steering_interval == 0
        ):
            extension = _bounded_increment(
                actuation.steering_tape_final_extension,
                actuation.steering_tape_extension_step,
                iteration,
                steering_interval,
            )
            left_idx, right_idx = [int(idx) for idx in actuation.steering_tape_indices]
            left_target = float(actuation.initial_steering_left_length) - extension
            right_target = float(actuation.initial_steering_right_length) + extension
            rest_lengths = np.asarray(system.extract_rest_length, dtype=float)
            system.update_rest_length(left_idx, left_target - float(rest_lengths[left_idx]))
            rest_lengths = np.asarray(system.extract_rest_length, dtype=float)
            system.update_rest_length(
                right_idx, right_target - float(rest_lengths[right_idx])
            )


def _bounded_increment(final_extension: float, step: float, iteration: int, interval: int) -> float:
    final_extension = float(final_extension)
    step = float(step)
    if abs(final_extension) <= 1e-15:
        return 0.0
    if abs(step) <= 1e-15:
        return final_extension

    n_applied = iteration // interval + 1
    magnitude = min(abs(final_extension), n_applied * abs(step))
    return float(np.sign(final_extension) * magnitude)


__all__ = ["PssKineticDampingSolver", "PssQsmCoupler"]

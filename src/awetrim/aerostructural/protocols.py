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

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class StructuralGeometry:
    """Structural point-mass geometry and element properties."""

    nodes: Array
    masses: Array
    connectivity: Array
    rest_lengths: Array
    stiffness: Array
    damping: Array
    link_types: Sequence[str]
    le_node_indices: Array
    te_node_indices: Array
    fixed_node_indices: Sequence[int]
    pulley_line_indices: Sequence[int]
    pulley_line_to_other_node_pair: Mapping[str, Sequence[float]]


@dataclass(frozen=True)
class AerodynamicGeometryUpdate:
    """Aerodynamic leading/trailing-edge geometry derived from structural nodes."""

    leading_edge_points: Array
    trailing_edge_points: Array


@dataclass(frozen=True)
class AeroToStructureMap:
    """Mapping from aerodynamic panel control points to structural corner nodes."""

    panel_corner_map: Array


@dataclass(frozen=True)
class TapeActuationState:
    """Rest-length targets for depower and steering tape actuation."""

    power_tape_index: int | None
    steering_tape_indices: tuple[int, int] | None
    initial_power_tape_length: float | None
    initial_steering_left_length: float | None
    initial_steering_right_length: float | None
    power_tape_final_extension: float = 0.0
    power_tape_extension_step: float = 0.0
    steering_tape_final_extension: float = 0.0
    steering_tape_extension_step: float = 0.0


@dataclass(frozen=True)
class QsmCouplingSettings:
    """Numerical settings for the outer aero-structural coupling loop."""

    max_iter: int
    residual_tolerance: float
    residual_stagnation_window: int
    residual_stagnation_tolerance: float
    relaxation_factor: float
    use_aitken_relaxation: bool
    n_aero_panels_per_structural_section: int
    include_gravity: bool = False
    include_aero_bridle: bool = False
    steering_actuation_interval_iters: int = 1
    power_tape_actuation_interval_iters: int = 1
    qs_state_stagnation_decimals: int = 3
    qs_state_stagnation_n_iter: int = 0
    pss_settings: Mapping[str, Any] | None = None
    moment_tolerance: float = 1e-2
    aero_input_type: str = "reuse_initial_polar_data"


@dataclass(frozen=True)
class QsmCouplingRequest:
    """Inputs needed to run one PSS-QSM aero-structural solve."""

    structural_geometry: StructuralGeometry
    system_model: Any
    body_aero: Any
    vsm_solver: Any
    center_of_gravity: Array | None
    reference_point: Array
    x_guess: Array
    bounds_lower: Array
    bounds_upper: Array
    settings: QsmCouplingSettings
    actuation: TapeActuationState | None = None
    initial_polar_data: Sequence[Any] | None = None


@dataclass(frozen=True)
class QsmIterationRecord:
    """Diagnostics from one outer aero-structural iteration."""

    iteration: int
    residual_norm: float
    structural_converged: bool
    trim_success: bool
    trim_success_physical: bool
    opt_x: Array
    total_aero_force: Array
    total_inertial_force: Array
    total_gravity_force: Array
    max_node_displacement: float


@dataclass(frozen=True)
class QsmCouplingResult:
    """Final state and diagnostics from a PSS-QSM coupling solve."""

    converged: bool
    final_nodes: Array
    final_rest_lengths: Array
    final_nodal_forces: Array
    final_residual: Array
    trim_result: Mapping[str, Any]
    iteration_records: Sequence[QsmIterationRecord]
    metadata: Mapping[str, Any]


class PssSystem(Protocol):
    """Minimal Particle System interface required by AWETrim."""

    @property
    def particles(self) -> Sequence[Any]:
        """Particle objects exposing current positions."""

    @property
    def extract_rest_length(self) -> Array:
        """Current element rest lengths."""

    @property
    def f_int(self) -> Array:
        """Current flattened internal force vector."""

    @property
    def x_v_current(self) -> tuple[Array, Array]:
        """Current flattened positions and velocities."""

    def update_rest_length(self, element_index: int, delta_length: float) -> None:
        """Increment one element rest length by `delta_length`."""

    def kin_damp_sim(self, external_force: Array) -> None:
        """Advance the PSS kinetic damping solve with flattened external forces."""


class PssStructuralSolver(Protocol):
    """Adapter that owns PSS instantiation and one structural relaxation call."""

    def instantiate(
        self, geometry: StructuralGeometry, settings: Mapping[str, Any]
    ) -> PssSystem:
        """Create a PSS system from AWETrim structural geometry."""

    def solve(
        self, system: PssSystem, external_force: Array
    ) -> tuple[PssSystem, bool, Array, Array]:
        """Return updated system, convergence flag, nodes, and flattened internal force."""


class DeformableAeroBody(Protocol):
    """VSM-compatible body whose wing mesh can be replaced by LE/TE points."""

    panels: Sequence[Any]

    def update_from_points(
        self,
        leading_edge_points: Array,
        trailing_edge_points: Array,
        *,
        aero_input_type: str,
        initial_polar_data: Sequence[Any] | None,
    ) -> None:
        """Update the aerodynamic mesh from structural leading/trailing edges."""


class StructuralToAeroMapper(Protocol):
    """Build VSM aerodynamic geometry from structural geometry."""

    def map(
        self,
        nodes: Array,
        le_node_indices: Array,
        te_node_indices: Array,
        n_panels_per_section: int,
    ) -> AerodynamicGeometryUpdate:
        """Return interpolated leading/trailing-edge points for VSM."""


class AeroToStructuralLoadMapper(Protocol):
    """Distribute VSM panel loads onto structural nodes."""

    def initialize(
        self,
        panels: Sequence[Any],
        nodes: Array,
        le_node_indices: Array,
        te_node_indices: Array,
    ) -> AeroToStructureMap:
        """Build a reusable panel-to-corner-node map."""

    def map_loads(
        self,
        panel_forces: Array,
        panel_points: Array,
        nodes: Array,
        mapping: AeroToStructureMap,
    ) -> Array:
        """Return nodal aerodynamic forces with total force preserved."""


class QsmAerostructuralCoupler(Protocol):
    """Run the PSS-QSM fixed-point aero-structural solve."""

    def solve(self, request: QsmCouplingRequest) -> QsmCouplingResult:
        """Iterate structure, VSM QSM trim, load mapping, actuation, and convergence."""


__all__ = [
    "AeroToStructuralLoadMapper",
    "AeroToStructureMap",
    "AerodynamicGeometryUpdate",
    "Array",
    "DeformableAeroBody",
    "PssStructuralSolver",
    "PssSystem",
    "QsmAerostructuralCoupler",
    "QsmCouplingRequest",
    "QsmCouplingResult",
    "QsmCouplingSettings",
    "QsmIterationRecord",
    "StructuralGeometry",
    "StructuralToAeroMapper",
    "TapeActuationState",
]

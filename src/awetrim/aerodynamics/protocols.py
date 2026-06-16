# Copyright (c) 2023-2026 Oriol Cayon, Delft University of Technology
#
# Licensed under the EUPL, Version 1.2 or - as soon they will be approved by
# the European Commission - subsequent versions of the EUPL (the "Licence");
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at:
#
#     https://joinup.ec.europa.eu/software/page/eupl
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the Licence is distributed on an "AS IS" basis,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the Licence for the specific language governing permissions and
# limitations under the Licence.
#
# SPDX-License-Identifier: EUPL-1.2

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class AxisDefinition:
    """Course-frame basis used by the VSM aerodynamic trim adapter."""

    course: Array
    normal: Array
    radial: Array


@dataclass(frozen=True)
class VsmTrimBounds:
    """Lower and upper bounds for the five VSM trim unknowns."""

    lower: Array
    upper: Array


@dataclass(frozen=True)
class VsmTrimState:
    """Ordered VSM trim state used by AWETrim aerodynamic APIs."""

    speed_tangential: float
    angle_roll_body_deg: float
    angle_pitch_body_deg: float
    angle_yaw_body_deg: float
    timeder_angle_course_body: float


@dataclass(frozen=True)
class VsmTrimRequest:
    """Inputs required for a single aerodynamic VSM quasi-steady trim solve."""

    body_aero: VsmBodyAerodynamics
    center_of_gravity: Array
    reference_point: Array
    system_model: AWETrimSystemModel
    x_guess: Array
    bounds: VsmTrimBounds
    axes: AxisDefinition
    transformation_c_from_vsm: Array
    include_gravity: bool
    moment_tolerance: float


@dataclass(frozen=True)
class VsmTrimResult:
    """Aerodynamic trim solution and post-processed VSM outputs."""

    opt_x: Array
    cm: Array
    cfx: float
    cfy: float
    angle_of_attack_deg: float
    side_slip_deg: float
    aero_roll_deg: float
    lift_coefficient: float
    drag_coefficient: float
    total_aero_force_vec: Array
    success: bool
    success_physical: bool
    diagnostics: Mapping[str, Any]


@dataclass(frozen=True)
class VsmSweepRequest:
    """Inputs required for a principal/secondary VSM aerodynamic trim sweep."""

    build_body: Callable[[Mapping[str, float]], VsmBodyAerodynamics]
    system_model: AWETrimSystemModel
    center_of_gravity: Array
    reference_point: Array
    x_guess: Array
    principal_axis: str
    secondary_axis: str
    sweep_values: Mapping[str, Sequence[float] | float]


@dataclass(frozen=True)
class VsmStabilityResult:
    """Aerodynamic stability derivative output around a VSM trim state."""

    j_longitudinal: Array
    j_lateral: Array
    a_longitudinal: Array
    a_lateral: Array
    eigenvalues_longitudinal: Array
    eigenvalues_lateral: Array
    stable_longitudinal: bool
    stable_lateral: bool
    diagnostics: Mapping[str, Any]


class VsmBodyAerodynamics(Protocol):
    """Structural interface required from a VSM aerodynamic body."""

    wings: Sequence[Any]
    panels: Sequence[Any]
    geometry_rotation: Array

    def va_initialize(
        self,
        *,
        Umag: float,
        angle_of_attack: float,
        side_slip: float,
        body_rates: float,
        body_axis: Array,
        reference_point: Array,
        rates_in_body_frame: bool,
    ) -> None:
        """Set apparent-flow and body-rate inputs before a VSM solve."""


class VsmSolver(Protocol):
    """Structural interface required from a VSM aerodynamic solver."""

    rho: float

    def solve(self, body: VsmBodyAerodynamics) -> Mapping[str, Any]:
        """Return VSM aerodynamic coefficients, forces, and panel diagnostics."""


class AWETrimSystemModel(Protocol):
    """Numerical AWETrim system data required by the VSM trim adapter."""

    speed_tangential: float
    timeder_angle_course: float
    acceleration: Array
    velocity_kite: Array
    velocity_apparent_wind: Array

    @property
    def wind(self) -> Any:
        """Wind model exposing `velocity_wind(system_model)`."""


class VsmQuasiSteadyAerodynamicSolver(Protocol):
    """Public service interface for VSM aerodynamic quasi-steady functionality."""

    def solve_trim(
        self, request: VsmTrimRequest
    ) -> tuple[VsmTrimResult, VsmBodyAerodynamics]:
        """Solve one rigid-geometry VSM aerodynamic quasi-steady trim state."""

    def compute_stability_derivatives(
        self,
        trim_request: VsmTrimRequest,
        trim_result: VsmTrimResult,
        *,
        mass: float,
        inertia_xx: float,
        inertia_yy: float,
        inertia_zz: float,
    ) -> VsmStabilityResult:
        """Linearise aerodynamic forces and moments around a trim state."""

    def run_sweep(self, request: VsmSweepRequest) -> Sequence[Mapping[str, Any]]:
        """Run a warm-started principal/secondary aerodynamic trim sweep."""

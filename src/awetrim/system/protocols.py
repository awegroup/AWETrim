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

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Protocol, Union

import casadi as ca

Symbolic = Union[ca.MX, ca.SX, ca.DM]


@dataclass
class FlightCondition:
    """Inputs that fully specify a quasi-steady trim condition.

    Passed to ``KiteAeroModel.solve_quasi_steady``; the solver fills in all
    derived quantities and returns them in a ``State``.
    """

    distance_radial: float          # tether length / radial distance [m]
    angle_elevation: float          # elevation angle β [rad]
    angle_course: float             # course angle χ [rad]
    speed_radial: float             # reelout speed vr [m/s]
    wind_speed: float               # wind speed at reference height [m/s]
    input_depower: float = 0.0      # depower setting [-]
    angle_azimuth: float = 0.0      # azimuth angle φ [rad]
    input_steering: float = 0.0         # steering input [-]


class KiteAeroModel(Protocol):
    """Shared interface for ROM and VSM/aerostructural aerodynamic models.

    Both ``SystemModel`` (ROM) and ``VSMAeroModelAdapter`` (VSM/PSS) must
    satisfy this protocol so that the identification pipeline and validation
    scripts can use either backend interchangeably.
    """

    def solve_quasi_steady(self, condition: FlightCondition) -> "State": ...

    def get_aero_coefficients(self, state: "State") -> Dict[str, float]:
        """Return at minimum {'CL': ..., 'CD': ..., 'CS': ...}."""
        ...

    def compute_forces(self, state: "State") -> Dict[str, Any]:
        """Return tether tension and aerodynamic force vector in course frame."""
        ...


class KiteModel(Protocol):
    """Interface required by SystemModel for kite component equations."""

    mass_wing: float
    mass_kcu: float
    input_steering: Symbolic
    input_depower: Symbolic
    g: float
    rho: float
    steering_control: str

    def force_aerodynamic(self, model: "SystemModelProtocol") -> Symbolic: ...

    def force_gravity_for(self, model: "SystemModelProtocol") -> Symbolic: ...

    def force_gravity_wing_for(self, model: "SystemModelProtocol") -> Symbolic: ...

    def force_gravity_kcu_for(self, model: "SystemModelProtocol") -> Symbolic: ...

    def angle_of_attack_for(self, model: "SystemModelProtocol") -> Symbolic: ...

    def pitch_bridle_for(self, model: "SystemModelProtocol") -> Symbolic: ...

    def roll_bridle_for(self, model: "SystemModelProtocol") -> Symbolic: ...

    def angle_roll_aerodynamic_for(
        self, model: "SystemModelProtocol"
    ) -> Symbolic: ...

    def lift_coefficient_for(self, model: "SystemModelProtocol") -> Symbolic: ...

    def drag_coefficient_for(self, model: "SystemModelProtocol") -> Symbolic: ...


class TetherModel(Protocol):
    """Interface required by SystemModel for tether component equations."""

    is_tether_rigid: bool

    def mass_tether_for(self, model: "SystemModelProtocol") -> Symbolic: ...

    def force_tether_at_kite_for(self, model: "SystemModelProtocol") -> Symbolic: ...

    def tension_kite_for(self, model: "SystemModelProtocol") -> Symbolic: ...

    def drag_tether_at_kite_for(self, model: "SystemModelProtocol") -> Symbolic: ...

    def force_gravity_tether_at_kite_for(
        self, model: "SystemModelProtocol"
    ) -> Symbolic: ...


class WindModel(Protocol):
    """Interface required by SystemModel for wind component equations."""

    speed_wind_ref: Symbolic

    def velocity_wind(self, model: "SystemModelProtocol") -> Symbolic: ...

    def velocity_wind_at_height(
        self, model: "SystemModelProtocol", height: Symbolic
    ) -> Symbolic: ...


class SystemModelProtocol(Protocol):
    """Structural protocol for the symbolic model context passed to components."""

    kite: KiteModel
    tether: TetherModel
    wind: WindModel
    distance_radial: Symbolic
    angle_elevation: Symbolic
    angle_azimuth: Symbolic
    angle_course: Symbolic
    speed_tangential: Symbolic
    speed_radial: Symbolic
    input_steering: Symbolic
    input_depower: Symbolic
    tension_tether_ground: Symbolic
    velocity_apparent_wind: Symbolic
    acceleration: Symbolic
    force_gravity_kcu: Symbolic
    g: float
    rho: float

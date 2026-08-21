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

"""Bluff-body drag of the kite control unit (KCU).

This module owns a dependency-light (numpy only, no VSM, no CasADi) drag model
for the KCU, treated as a finite cylinder hanging on the tether. It is THE
single place the coefficients and the force law are written (root ``AGENTS.md``
single-source rule); the VSM trims, their stability linearisations and any
script that needs a KCU drag call in here rather than restating the formula.

Force law
---------
With the apparent wind ``va`` split about the KCU long axis ``e`` into an
axial part ``v_ax = (va . e) e`` and a crossflow part ``v_perp = va - v_ax``::

    F = 0.5 rho ( cd_area_axial |v_ax| v_ax + cd_area_broadside |v_perp| v_perp )

each component quadratic in its own velocity, so the force reverses with the
flow and vanishes with it. The two drag areas are constants of a given KCU::

    cd_area_axial     = cdt(L/D) * pi (d/2)^2      (flow ALONG the axis)
    cd_area_broadside = cdp(L/D) * d L             (flow ACROSS the axis)

Coefficients
------------
``cdt`` (axial flow, referenced to the frontal area) and ``cdp`` (crossflow,
referenced to the broadside area) are the finite-cylinder tables of the
*Applied Fluid Dynamics Handbook* (Blevins), interpolated linearly in the
fineness ratio L/D. They depend only on the KCU's geometry, so they collapse to
two floats per kite and never enter a symbolic graph -- which is why this
module needs neither the ``xp`` math-namespace pattern of
``environment/profile_laws.py`` nor ``ca.interpolant``. Linear interpolation
differs from the cubic spline of the reference implementation by 0.23 % (cdt)
and 0.08 % (cdp) at the LEI-V3 fineness -- 0.23 %/0.16 % at the LEI-V9 one, and
at worst 1.4 %/1.9 % anywhere in the tables' interpolation range -- well below
the uncertainty of a handbook bluff-body coefficient.

Pairing note (2026-08-21)
-------------------------
The reference implementation this model is compared against --
``awes_ekf/setup/kcu.py`` + ``awes_ekf/setup/tether.py`` -- applies the
CROSSFLOW pair ``(cdp, Ap)`` to the AXIAL velocity component and the AXIAL pair
``(cdt, At)`` to the crossflow component (``tether.py``: ``vajp`` is
``dot(vaj, ej) ej``, i.e. parallel to the tether, yet is multiplied by
``cdp * Ap``; the source even carries a ``# TODO: check whether to use vajn``).
The handbook tables identify the pairs unambiguously: ``cdt(L/D -> 0) = 1.15``
is a disc facing the flow (axial, frontal area) and ``cdp(L/D -> inf) = 1.2``
is infinite-cylinder crossflow (planform area). This module uses the physical
pairing. Consequence: in crosswind flight -- where the flow is mostly ACROSS a
tether-aligned KCU -- it predicts roughly 1.5x the EKF's published
``kcu_drag_coefficient`` channel. That gap is the pairing, not a modelling
error; do not "calibrate" it away. Swapping the two areas at the call site
reproduces the EKF convention exactly, should that ever be wanted.

Axis and station
----------------
Callers pass the KCU long axis. In the trims it is the outward radial
(tether) direction, which is exact only for a radially hanging tether; the real
tether tilts toward the apparent wind, which lowers the crossflow share, so the
radial assumption slightly OVER-estimates the drag (about 13 % of the KCU term,
1.5 % of total CD, for a 10 deg tilt). The force is applied at the bridle point
where the KCU hangs, which in the shipped configurations is the trim
``reference_point`` -- so it produces no moment there, but it does about any
other pivot (the CG form).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

#: Fineness ratio L/D of the tabulated AXIAL-flow drag coefficients.
FINENESS_AXIAL = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0])
#: Axial-flow drag coefficient of a finite cylinder, on the FRONTAL area
#: ``pi (d/2)^2`` (Applied Fluid Dynamics Handbook).
CD_AXIAL = np.array([1.15, 1.10, 0.93, 0.85, 0.83, 0.85, 0.85, 0.85])

#: Fineness ratio L/D of the tabulated CROSSFLOW drag coefficients.
FINENESS_BROADSIDE = np.array([1.0, 1.98, 2.96, 5.0, 10.0, 20.0, 40.0, 1.0e6])
#: Crossflow drag coefficient of a finite cylinder, on the BROADSIDE area
#: ``d L`` (Applied Fluid Dynamics Handbook).
CD_BROADSIDE = np.array([0.64, 0.68, 0.74, 0.74, 0.82, 0.91, 0.98, 1.20])

#: Label recorded in trim results for the KCU-axis assumption in use.
AXIS_MODEL_RADIAL = "radial"
#: Label recorded when no KCU drag was applied.
AXIS_MODEL_NONE = "none"


def _fineness(length_kcu: float, diameter_kcu: float) -> float:
    """L/D of the KCU, or 0.0 when either dimension is missing."""
    length = float(length_kcu or 0.0)
    diameter = float(diameter_kcu or 0.0)
    if length <= 0.0 or diameter <= 0.0:
        return 0.0
    return length / diameter


def cd_area_axial_kcu(length_kcu: float, diameter_kcu: float) -> float:
    """Drag area ``cdt(L/D) * pi (d/2)^2`` [m^2] for flow ALONG the KCU axis.

    Zero when either dimension is missing, so a system without KCU geometry
    simply carries no KCU drag. Outside the tabulated fineness range the
    endpoint coefficients are held (``np.interp`` clamps; the reference
    implementation extrapolates its spline instead).
    """
    fineness = _fineness(length_kcu, diameter_kcu)
    if fineness <= 0.0:
        return 0.0
    area_frontal = np.pi * (float(diameter_kcu) / 2.0) ** 2
    return float(np.interp(fineness, FINENESS_AXIAL, CD_AXIAL) * area_frontal)


def cd_area_broadside_kcu(length_kcu: float, diameter_kcu: float) -> float:
    """Drag area ``cdp(L/D) * d L`` [m^2] for flow ACROSS the KCU axis.

    Zero when either dimension is missing; endpoints held outside the table.
    """
    fineness = _fineness(length_kcu, diameter_kcu)
    if fineness <= 0.0:
        return 0.0
    area_broadside = float(diameter_kcu) * float(length_kcu)
    return float(np.interp(fineness, FINENESS_BROADSIDE, CD_BROADSIDE) * area_broadside)


def force_drag_kcu(
    velocity_apparent: Any,
    axis_kcu: Any,
    density_air: float,
    cd_area_axial: float,
    cd_area_broadside: float,
) -> np.ndarray:
    """KCU drag force [N], in whatever frame the inputs are given in.

    ``axis_kcu`` need not be a unit vector and its sign is irrelevant (the law
    is even in the axis), so callers never have to fix an orientation
    convention. Returns zeros for a vanishing apparent wind, a degenerate axis
    or zero drag areas.
    """
    va = np.asarray(velocity_apparent, dtype=float).ravel()
    axis = np.asarray(axis_kcu, dtype=float).ravel()
    if va.size != 3 or axis.size != 3:
        raise ValueError("velocity_apparent and axis_kcu must be 3-vectors.")
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm < 1e-12 or float(np.linalg.norm(va)) < 1e-12:
        return np.zeros(3)
    unit = axis / axis_norm
    velocity_axial = float(np.dot(va, unit)) * unit
    velocity_crossflow = va - velocity_axial
    return (
        0.5
        * float(density_air)
        * (
            float(cd_area_axial)
            * float(np.linalg.norm(velocity_axial))
            * velocity_axial
            + float(cd_area_broadside)
            * float(np.linalg.norm(velocity_crossflow))
            * velocity_crossflow
        )
    )


@dataclass(frozen=True)
class KcuDragModel:
    """The two drag areas of one KCU, plus the geometry they came from.

    Construct through the ``from_*`` classmethods: every one of them returns
    ``None`` when the geometry is missing or degenerate, so "no KCU drag" is a
    single sentinel the call sites test for rather than an exception path.
    """

    length: float
    diameter: float
    cd_area_axial: float
    cd_area_broadside: float

    # ------------------------------------------------------------ constructors
    @classmethod
    def from_dimensions(
        cls, length_kcu: float, diameter_kcu: float
    ) -> "KcuDragModel | None":
        """Model from KCU length and diameter [m]; ``None`` if either is unset."""
        cd_area_axial = cd_area_axial_kcu(length_kcu, diameter_kcu)
        cd_area_broadside = cd_area_broadside_kcu(length_kcu, diameter_kcu)
        if cd_area_axial <= 0.0 or cd_area_broadside <= 0.0:
            return None
        return cls(
            length=float(length_kcu),
            diameter=float(diameter_kcu),
            cd_area_axial=cd_area_axial,
            cd_area_broadside=cd_area_broadside,
        )

    @classmethod
    def from_system_model(cls, system_model: Any) -> "KcuDragModel | None":
        """Model from ``system_model.kite.{length_kcu, diameter_kcu}``.

        Same attribute lookup the trims use for ``mass_kcu``, so a system model
        built from a ``system.yaml`` without a ``control_system.structure``
        block (or a bare test double) yields ``None`` -- no KCU drag, no crash.
        """
        kite = getattr(system_model, "kite", system_model)
        return cls.from_dimensions(
            getattr(kite, "length_kcu", 0.0), getattr(kite, "diameter_kcu", 0.0)
        )

    @classmethod
    def from_cd_areas(
        cls, cd_area_axial: float, cd_area_broadside: float
    ) -> "KcuDragModel | None":
        """Model straight from the two drag areas [m^2] (geometry unknown)."""
        if float(cd_area_axial or 0.0) <= 0.0 or float(cd_area_broadside or 0.0) <= 0.0:
            return None
        return cls(
            length=0.0,
            diameter=0.0,
            cd_area_axial=float(cd_area_axial),
            cd_area_broadside=float(cd_area_broadside),
        )

    @classmethod
    def from_trim_result(cls, trim_result: Mapping[str, Any]) -> "KcuDragModel | None":
        """Model a trim was solved with, recovered from its own result dict.

        Lets the stability linearisation default to exactly the model of the
        trim it linearises about, instead of being told separately -- a
        mismatch there would be absorbed by the baseline anchor and show up
        only as missing damping in the Jacobian.
        """
        if not trim_result:
            return None
        return cls.from_cd_areas(
            trim_result.get("kcu_cd_area_axial_m2", 0.0),
            trim_result.get("kcu_cd_area_broadside_m2", 0.0),
        )

    # ----------------------------------------------------------------- physics
    def force(
        self, velocity_apparent: Any, axis_kcu: Any, density_air: float
    ) -> np.ndarray:
        """KCU drag force [N] (see :func:`force_drag_kcu`)."""
        return force_drag_kcu(
            velocity_apparent,
            axis_kcu,
            density_air,
            self.cd_area_axial,
            self.cd_area_broadside,
        )

    def drag_coefficient(
        self, velocity_apparent: Any, axis_kcu: Any, area_reference: float
    ) -> float:
        """Additive CD contribution ``(F . va_hat) / (q S)`` [-].

        This is the number to ADD to a wing drag coefficient referenced to the
        same area: it is the streamwise share of the KCU force. Closed form,
        with theta the angle between the apparent wind and the KCU axis::

            (cd_area_axial |cos theta|^3 + cd_area_broadside sin^3 theta) / S
        """
        va = np.asarray(velocity_apparent, dtype=float).ravel()
        speed = float(np.linalg.norm(va))
        if speed < 1e-12 or float(area_reference) <= 0.0:
            return 0.0
        force = self.force(va, axis_kcu, 1.0)  # rho factored out with q below
        return float(np.dot(force, va / speed)) / (
            0.5 * speed**2 * float(area_reference)
        )

    def force_coefficient(
        self, velocity_apparent: Any, axis_kcu: Any, area_reference: float
    ) -> float:
        """Force-magnitude coefficient ``|F| / (q S)`` [-].

        The normalisation the EKF's ``kcu_drag_coefficient`` channel uses
        (``D_kcu = |dp + dt|``), so this is the one to compare against flight
        -- keeping the pairing caveat in the module docstring in mind.
        """
        va = np.asarray(velocity_apparent, dtype=float).ravel()
        speed = float(np.linalg.norm(va))
        if speed < 1e-12 or float(area_reference) <= 0.0:
            return 0.0
        force = self.force(va, axis_kcu, 1.0)
        return float(np.linalg.norm(force)) / (0.5 * speed**2 * float(area_reference))

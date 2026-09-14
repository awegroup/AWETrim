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

"""Apparent wind at a point of a rigidly rotating body.

THE single place this relation is written for the kite-body family (root
``AGENTS.md`` single-source rule)::

    va(r) = va_free - omega x (r - r0)

``va_free`` is the apparent wind at the reference point ``r0``, ``omega`` the
body's angular velocity, and ``r`` the station the inflow is wanted at. The
sign and the convention are the VSM's own -- ``BodyAerodynamics``'s ``va``
setter builds its panel distribution as ``va_distribution += -cross(omega,
control_points - r0)`` -- so anything charged alongside the wing (bridle
segments, the KCU) must use this and the same ``(omega, r0)`` the trim was
solved with, or it is charged at the wrong station.

Why this is worth a function
----------------------------
Not the arithmetic. It is that ``r0`` and ``omega`` have to be NAMED at every
call site. Two bugs in this repository were exactly a missing or misplaced
``r - r0``:

* the bridle segments were charged ``va_ref_vector``, the VSM's area-weighted
  mean PANEL inflow -- the rotational term evaluated at the WING, on a body
  that is not the wing;
* the KCU's drag took ``va_free`` while the moment arm beside it was written
  ``-r0``, i.e. the two halves of one force disagreed about where the KCU is.
  They agreed only because ``r0`` happens to be the geometry origin in every
  shipped configuration.

Both vanish when ``omega`` is zero, so unsteered baselines never show them.

What does NOT belong here
-------------------------
The tether. ``system/williams_tether.py`` rotates about the GROUND ANCHOR, not
the kite's reference point, and evaluates the wind at each node's own height
rather than carrying one freestream -- a different pivot and a different wind,
already written explicitly there. The lumped/rigid closed form in
``system/tether.py`` is an INTEGRAL along the tether reduced to one equivalent
force at the kite, not a point evaluation, so it cannot call a point kernel at
all. Both are CasADi; this is numpy, like ``kcu_drag``.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np


class RigidInflowState(NamedTuple):
    """The triple that defines a rigidly rotating body's inflow field.

    ``velocity_apparent_free`` is ``None`` when the body carries a full
    per-panel distribution: there is then no single freestream to extrapolate
    from, and the caller must decide what to charge rather than be handed an
    invented one.
    """

    velocity_apparent_free: np.ndarray | None
    velocity_rotation: np.ndarray
    reference_point: np.ndarray


def apparent_wind_at(
    velocity_apparent_free: Any,
    velocity_rotation: Any,
    position: Any,
    reference_point: Any = None,
    xp: Any = np,
) -> Any:
    """Apparent wind [m/s] at ``position``, in the frame of the inputs.

    Args:
        velocity_apparent_free: ``va_free``, the apparent wind at
            ``reference_point`` (3-vector).
        velocity_rotation: ``omega``, the body's angular velocity (3-vector).
            Zero reduces the result to ``va_free``.
        position: ``r``, the station the inflow is wanted at (3-vector).
        reference_point: ``r0``; ``None`` means the origin.
        xp: math namespace, ``numpy`` (default) or ``casadi``. With CasADi
            the two velocities may be symbolic 3-vectors; ``position`` and
            ``reference_point`` stay numeric.

    Returns:
        ``va_free - omega x (r - r0)`` as a fresh ``(3,)`` array (or the
        CasADi expression of it).
    """
    if xp is not np:
        station = np.asarray(position, dtype=float).ravel()
        origin = np.zeros(3) if reference_point is None else np.asarray(reference_point, dtype=float).ravel()
        arm = station - origin
        omega = velocity_rotation
        return velocity_apparent_free - xp.vertcat(
            omega[1] * arm[2] - omega[2] * arm[1],
            omega[2] * arm[0] - omega[0] * arm[2],
            omega[0] * arm[1] - omega[1] * arm[0],
        )
    va_free = np.asarray(velocity_apparent_free, dtype=float).ravel()
    omega = np.asarray(velocity_rotation, dtype=float).ravel()
    station = np.asarray(position, dtype=float).ravel()
    origin = (
        np.zeros(3)
        if reference_point is None
        else np.asarray(reference_point, dtype=float).ravel()
    )
    for name, vec in (
        ("velocity_apparent_free", va_free),
        ("velocity_rotation", omega),
        ("position", station),
        ("reference_point", origin),
    ):
        if vec.size != 3:
            raise ValueError(f"{name} must be a 3-vector; got size {vec.size}.")
    return va_free - np.cross(omega, station - origin)


def inflow_state_of(body_aero: Any) -> RigidInflowState:
    """Read ``(va_free, omega, r0)`` off a VSM ``BodyAerodynamics``.

    One place for the attribute archaeology, because getting it wrong is
    invisible: every shipped configuration puts ``r0`` at the origin, so a
    consumer that silently defaults it to zeros looks correct forever.

    ``body_aero.va`` is the freestream exactly as handed to the VSM's ``va``
    setter -- the rotational term lives on ``panel.va``, never on ``_va``. A
    caller who supplied a per-panel field instead gets ``None`` here: the
    freestream is genuinely not recoverable from it (reconstructing one by
    ADDING ``omega x (r - r0)`` back onto ``_va`` double-counts, since ``_va``
    never had it subtracted).

    ``reference_point`` needs a VSM that stores it; older builds fall back to
    the origin, which is what they effectively assumed anyway.
    """
    omega = np.asarray(
        getattr(body_aero, "body_rates", None)
        if getattr(body_aero, "body_rates", None) is not None
        else np.zeros(3),
        dtype=float,
    ).ravel()
    if omega.size != 3:
        omega = np.zeros(3)

    origin = getattr(body_aero, "reference_point", None)
    if origin is None:
        origin = getattr(body_aero, "_reference_point", None)
    origin = np.asarray(np.zeros(3) if origin is None else origin, dtype=float).ravel()
    if origin.size != 3:
        origin = np.zeros(3)

    va_free = getattr(body_aero, "va", None)
    va_free = None if va_free is None else np.asarray(va_free, dtype=float)
    if va_free is not None and va_free.shape != (3,):
        va_free = None  # per-panel distribution: no single freestream
    if va_free is not None and not np.all(np.isfinite(va_free)):
        va_free = None

    return RigidInflowState(va_free, omega, origin)

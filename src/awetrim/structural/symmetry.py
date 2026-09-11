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

"""Mirror symmetry: node pairing, symmetry constraints, frame consistency.

A structure that is mirror-symmetric in every input has a mirror-symmetric
equilibrium, but that need not be the one a minimum-energy solve returns: from a
slack, nearly singular start the solve can settle on either side of a flat
valley, and roundoff decides which. Constraining the solve to the symmetric
subspace, then releasing it, separates the two questions "does a symmetric
equilibrium exist" and "is it the one the unconstrained solve finds".

With the reflection ``M`` (default ``diag(1, -1, 1)``, the plane ``y = 0``) and
``p(i)`` the mirror partner of node ``i``:

* positions are vectors, ``x_p(i) = M x_i``;
* rotation increments are PSEUDOVECTORS, ``psi_p(i) = -M psi_i``, equivalently
  ``C_p(i) = M C_i M`` for the rotation matrices -- a reflection reverses the
  sense of every rotation about an in-plane axis and keeps the one about the
  plane normal.

On the plane itself (``p(i) = i``) this leaves a node free to move in the plane
and to rotate about the plane normal only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import casadi as ca
import numpy as np

from .model import DofLayout

Array = np.ndarray

#: Reflection in the plane ``y = 0``.
REFLECTION_Y = np.diag([1.0, -1.0, 1.0])


def _check_reflection(reflection: Array) -> tuple[Array, Array, Array]:
    """``(M, normal, in_plane)``: the reflection, its -1 and +1 eigenvectors."""
    reflection = np.asarray(reflection, dtype=float).reshape(3, 3)
    if not (
        np.allclose(reflection, reflection.T)
        and np.allclose(reflection @ reflection, np.eye(3))
        and np.isclose(np.linalg.det(reflection), -1.0)
    ):
        raise ValueError("the mirror must be a reflection (symmetric, orthogonal, det -1)")
    values, vectors = np.linalg.eigh(reflection)
    return reflection, vectors[:, values < 0].T, vectors[:, values > 0].T


@dataclass(frozen=True)
class LinearEqualities:
    """Homogeneous linear equalities ``C X = 0`` on the unknown vector, as triplets."""

    rows: Array
    columns: Array
    values: Array
    n_rows: int

    def casadi_matrix(self, n_columns: int) -> ca.DM:
        """``C`` as a sparse CasADi matrix -- never densified, so it scales."""
        return ca.DM.triplet(
            [int(r) for r in self.rows],
            [int(c) for c in self.columns],
            ca.DM([float(v) for v in self.values]),
            int(self.n_rows),
            int(n_columns),
        )

    def dense(self, n_columns: int) -> Array:
        matrix = np.zeros((int(self.n_rows), int(n_columns)))
        np.add.at(matrix, (self.rows, self.columns), self.values)
        return matrix

    def residual(self, unknowns: Array) -> Array:
        """``C X`` -- zero exactly when ``X`` satisfies the equalities."""
        unknowns = np.asarray(unknowns, dtype=float).reshape(-1)
        out = np.zeros(int(self.n_rows))
        np.add.at(out, self.rows, self.values * unknowns[self.columns])
        return out


def mirror_partners(
    nodes: Array, reflection: Array = REFLECTION_Y, tolerance: float = 1e-9
) -> Array:
    """Index of each node's mirror partner in an exactly symmetric node set.

    Raises if any node has no partner within ``tolerance`` [m] -- pairing a set
    that is only approximately symmetric would silently constrain the wrong
    nodes together.
    """
    reflection, _, _ = _check_reflection(reflection)
    nodes = np.asarray(nodes, dtype=float).reshape(-1, 3)
    mirrored = nodes @ reflection.T
    distances = np.linalg.norm(nodes[None, :, :] - mirrored[:, None, :], axis=2)
    partner = distances.argmin(axis=1)
    worst = float(distances[np.arange(len(nodes)), partner].max(initial=0.0))
    if worst > tolerance:
        raise ValueError(
            f"the node set is not mirror-symmetric: worst unpaired node is "
            f"{worst:.3e} m from any mirror image (tolerance {tolerance:.1e} m)"
        )
    if np.any(partner[partner] != np.arange(len(nodes))):
        raise ValueError("mirror pairing is not an involution; nodes coincide")
    return partner


def mirror_equalities(
    layout: DofLayout,
    partner: Array,
    reflection: Array = REFLECTION_Y,
    pinned_nodes: Sequence[int] = (),
) -> LinearEqualities:
    """The equalities that restrict ``X`` to mirror-symmetric configurations.

    One row per independent condition, so the constraint Jacobian has full row
    rank: three per mirror pair of nodes, one (the out-of-plane component) per
    node on the plane; the same for rotational slots with the pseudovector rule,
    two per slot on the plane. Rows whose every DOF is pinned are dropped -- they
    would reach IPOPT as empty rows once it removes fixed variables.
    """
    reflection, normal, in_plane = _check_reflection(reflection)
    partner = np.asarray(partner, dtype=int)
    pinned = set(int(n) for n in pinned_nodes)
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    n_rows = 0

    def add_row(entries):
        nonlocal n_rows
        for column, value in entries:
            if value != 0.0:
                rows.append(n_rows)
                columns.append(int(column))
                values.append(float(value))
        n_rows += 1

    def pair_rows(dof_a, dof_b, sign):
        # x_a - sign * M x_b = 0, component by component
        for r in range(3):
            add_row([(dof_a[r], 1.0)] + [(dof_b[c], -sign * reflection[r, c]) for c in range(3)])

    def plane_rows(dof, directions):
        for direction in directions:
            add_row([(dof[c], direction[c]) for c in range(3)])

    for node in range(layout.n_nodes):
        other = int(partner[node])
        if other < node:
            continue
        both_pinned = node in pinned and other in pinned
        if both_pinned:
            continue
        dof = layout.translation_dof(np.array([node]))[0]
        if other == node:
            plane_rows(dof, normal)                    # n . x = 0
        else:
            pair_rows(dof, layout.translation_dof(np.array([other]))[0], +1.0)

    for node in layout.rotational_nodes:
        node = int(node)
        other = int(partner[node])
        if other < node:
            continue
        if layout.rotation_slot(other) < 0:
            raise ValueError(
                f"node {node} carries rotational DOF but its mirror partner {other} "
                "does not; the element layout is not mirror-symmetric"
            )
        dof = layout.rotation_dof(np.array([node]))[0]
        if other == node:
            plane_rows(dof, in_plane)                  # only the normal spin survives
        else:
            pair_rows(dof, layout.rotation_dof(np.array([other]))[0], -1.0)

    return LinearEqualities(
        rows=np.asarray(rows, dtype=int),
        columns=np.asarray(columns, dtype=int),
        values=np.asarray(values, dtype=float),
        n_rows=int(n_rows),
    )


#: Director signs of the mirror frame convention ``R_p = M R S``: ``d1`` flips.
#: Right for any member that CROSSES the plane (a leading edge), whose chain
#: direction the reflection reverses, so its tangent maps as ``-M t``.
MIRROR_SIGNS = (-1.0, 1.0, 1.0)


def mirror_frames(
    frames: Array,
    nodes: Array,
    partner: Array,
    frame_nodes: Sequence[int],
    reflection: Array = REFLECTION_Y,
    signs: Sequence[float] = MIRROR_SIGNS,
    tolerance: float = 1e-9,
) -> Array:
    """Make a nodal frame field exactly mirror-consistent: ``R_p(i) = M R_i S``.

    Frames on the positive side of the mirror plane are kept; each partner on
    the negative side gets the mirror image of its frame. With ONE sign matrix
    ``S`` for every node, every beam element's strain maps to a sign-flipped
    copy of its mirror element's, so the energy of a mirror-symmetric model is
    exactly mirror-symmetric (the kernels are even in each strain component).

    Why this is needed at all: minimal-rotation transport over a beam network is
    NOT mirror-equivariant when the network mixes members whose tangents map
    with opposite signs -- a leading edge crossing the plane (``t -> -M t``) and
    struts beside it (``t -> +M t``). Every hop between the two at a junction
    rotates the roll the wrong way on one side, so frames transported from one
    tip arrive at the other rolled by tens of degrees. The positions stay
    exactly symmetric, so no position check can see it; the energy does.

    A node ON the plane must already satisfy ``R = M R S``: with the default
    ``S`` that means ``d1`` along the plane normal, which a leading-edge node at
    the centre does. A member lying IN the plane (a centre strut) cannot share
    one ``S`` with a leading edge crossing it, and raises.
    """
    reflection, normal, _ = _check_reflection(reflection)
    frames = np.array(frames, dtype=float, copy=True).reshape(-1, 3, 3)
    nodes = np.asarray(nodes, dtype=float).reshape(-1, 3)
    partner = np.asarray(partner, dtype=int)
    sign_matrix = np.diag(np.asarray(signs, dtype=float))
    if not np.isclose(np.linalg.det(sign_matrix), -1.0):
        raise ValueError("the director signs must have determinant -1")
    side = nodes @ normal[0]
    for node in (int(n) for n in frame_nodes):
        other = int(partner[node])
        if other == node:
            if np.abs(reflection @ frames[node] @ sign_matrix - frames[node]).max() > 1e-6:
                raise ValueError(
                    f"beam node {node} lies on the mirror plane but its frame is "
                    "not its own mirror image; a member in the plane cannot share "
                    "the director signs of one that crosses it"
                )
            continue
        if side[node] > tolerance:
            frames[other] = reflection @ frames[node] @ sign_matrix
    return frames


def frame_mirror_mismatch(
    frames: Array, layout: DofLayout, partner: Array, reflection: Array = REFLECTION_Y
) -> Array:
    """Per rotational slot, how far the frame is from the mirror of its partner's [rad].

    A mirror-consistent pair satisfies ``R_p = M R_i S`` with ``S`` a diagonal
    sign matrix of determinant -1 (which directors flip depends on whether the
    member runs through the plane or alongside it). The angle returned is the
    rotation left over once the nearest such ``S`` is removed: zero for a
    consistent pair, and in particular it exposes a ROLL mismatch about the
    member axis, which positions alone can never show.
    """
    reflection, _, _ = _check_reflection(reflection)
    frames = np.asarray(frames, dtype=float).reshape(-1, 3, 3)
    partner = np.asarray(partner, dtype=int)
    angles = np.zeros(len(frames))
    for slot, node in enumerate(layout.rotational_nodes):
        other = layout.rotation_slot(int(partner[int(node)]))
        signed = frames[slot].T @ reflection @ frames[other]
        signs = np.where(np.diag(signed) >= 0.0, 1.0, -1.0)
        if np.prod(signs) > 0:
            angles[slot] = np.pi
            continue
        proper = np.diag(signs) @ signed
        # atan2 of the skew and trace parts: arccos of the trace alone cannot
        # resolve angles below ~4e-8 rad, which would read as a floor.
        skew_part = 0.5 * np.linalg.norm([proper[2, 1] - proper[1, 2],
                                          proper[0, 2] - proper[2, 0],
                                          proper[1, 0] - proper[0, 1]])
        angles[slot] = float(np.arctan2(skew_part, 0.5 * (np.trace(proper) - 1.0)))
    return angles

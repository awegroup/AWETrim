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

"""Geometrically exact shear-flexible (Timoshenko) beam element.

Two-node Simo-Reissner beam with one-point (mid-element) integration. Each node
carries a position and an orthonormal material frame ``R = [d1 d2 d3]`` whose
columns are the directors: ``d1`` along the beam axis, ``d2``/``d3`` the
cross-section principal axes. Twelve DOF per element -- three translations and
three incremental rotations per node.

Strain measures at the element midpoint, with ``psi`` the relative rotation
between the two node frames and ``Rm`` the exact half-way frame:

    Gamma = Rm^T (x2 - x1) / L0        axial stretch + two transverse shears
    Omega = Rm^T psi / L0              torsion + two bending curvatures

    U = L0/2 [ dG^T diag(EA, GA2, GA3) dG + dO^T diag(GJ, EI2, EI3) dO ]

with ``dG = Gamma - Gamma0`` and ``dO = Omega - Omega0`` measured against the
reference configuration, so an initially curved or pre-twisted member (a
leading-edge tube, a strut) is stress-free as built.

Three properties are what make this element usable inside an NLP objective:

**Objective.** Under a superposed rigid rotation ``Q``, both ``psi`` and
``x2 - x1`` map to ``Q(.)`` while ``Rm -> Q Rm``, so ``Gamma`` and ``Omega`` are
invariant and rigid-body motion costs exactly zero energy. A slack member that
swings freely is not spuriously stiffened.

**Shear-locking free.** The single midpoint sample is the standard reduced
integration cure (Simo and Vu-Quoc 1986): a slender element bends without the
parasitic shear energy full integration would add. Kite battens and LE tubes are
slender, so this is not optional.

**Branch-free.** ``psi`` is a Rodrigues (Cayley) vector, so every map in the
kernel is rational -- see :mod:`awetrim.structural.rotations`. The element
strain measure is ``Omega_exact + O(phi^3)`` per element, the same order as the
one-point quadrature error and vanishing under mesh refinement. Paid
deliberately: the alternative, the exponential map and its ``sin(phi)/phi``,
needs an ``if_else`` at ``phi = 0`` that would put a kink in the Hessian IPOPT
differentiates.

**Inflatable members:** :meth:`BeamSection.from_tube` returns the thin-walled
*structural* stiffness of the laminate. The bending stiffness of a real
inflated LE tube is pressure-dependent and collapses past a critical moment
once the compressed side wrinkles; treat ``from_tube`` as a starting point to
calibrate, not as a constitutive law for the tube.

Reference: Simo and Vu-Quoc (1986) *Comput. Methods Appl. Mech. Engrg.* 58,
79-116.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import casadi as ca
import numpy as np

from ..rotations import (
    unit_vector,
    cayley,
    cayley_vector,
    half_vector,
    minimal_rotation,
)
from .base import ElementSet, element_frame, element_rotation, element_translations

Array = np.ndarray

PARAM_NAMES: tuple[str, ...] = (
    "rest_length",
    "ea",
    "ga_2",
    "ga_3",
    "gj",
    "ei_2",
    "ei_3",
    "gamma_0_1",
    "gamma_0_2",
    "gamma_0_3",
    "omega_0_1",
    "omega_0_2",
    "omega_0_3",
)


def beam_strains(x_1, x_2, rotation_1, rotation_2, rest_length, xp):
    """Midpoint ``(Gamma, Omega)`` of a two-node geometrically exact beam.

    The single place these strain measures are written: the CasADi kernel and
    the NumPy computation of the reference strains both call it.
    """
    relative = rotation_2 @ rotation_1.T
    psi = cayley_vector(relative, xp)
    frame_mid = cayley(half_vector(psi, xp), xp) @ rotation_1
    gamma = (frame_mid.T @ (x_2 - x_1)) / rest_length
    omega = (frame_mid.T @ psi) / rest_length
    return gamma, omega


@dataclass(frozen=True)
class TimoshenkoBeamKernel:
    """Strain energy of one two-node geometrically exact beam element."""

    nodes_per_element: int = 2
    rotational_nodes: tuple[int, ...] = (0, 1)
    param_names: tuple[str, ...] = PARAM_NAMES

    def energy(self, q, frames, p):
        x_1 = element_translations(q, 0)
        x_2 = element_translations(q, 1)
        rotation_1 = cayley(element_rotation(q, 2, 0), ca) @ element_frame(frames, 0, ca)
        rotation_2 = cayley(element_rotation(q, 2, 1), ca) @ element_frame(frames, 1, ca)

        rest_length = p[0]
        gamma, omega = beam_strains(x_1, x_2, rotation_1, rotation_2, rest_length, ca)
        d_gamma = gamma - ca.vertcat(p[7], p[8], p[9])
        d_omega = omega - ca.vertcat(p[10], p[11], p[12])

        axial = ca.vertcat(p[1], p[2], p[3])
        bending = ca.vertcat(p[4], p[5], p[6])
        return 0.5 * rest_length * (
            ca.dot(axial * d_gamma, d_gamma) + ca.dot(bending * d_omega, d_omega)
        )


@dataclass(frozen=True)
class BeamSection:
    """Cross-section stiffnesses of a beam member, in material axes.

    ``ea`` axial, ``ga_2``/``ga_3`` transverse shear, ``gj`` torsion,
    ``ei_2``/``ei_3`` bending about the two cross-section principal axes.
    """

    ea: float
    ga_2: float
    ga_3: float
    gj: float
    ei_2: float
    ei_3: float

    @classmethod
    def from_tube(
        cls,
        diameter: float,
        wall_thickness: float,
        youngs_modulus: float,
        shear_modulus: float,
        *,
        shear_correction: float = 0.5,
    ) -> "BeamSection":
        """Thin-walled circular tube -- LE tubes and struts before inflation.

        ``shear_correction`` is the Timoshenko shear coefficient kappa; 0.5 is
        the standard value for a thin-walled circular section.
        """
        radius = 0.5 * diameter
        area = 2.0 * np.pi * radius * wall_thickness
        second_moment = np.pi * radius ** 3 * wall_thickness
        polar = 2.0 * second_moment
        return cls(
            ea=youngs_modulus * area,
            ga_2=shear_correction * shear_modulus * area,
            ga_3=shear_correction * shear_modulus * area,
            gj=shear_modulus * polar,
            ei_2=youngs_modulus * second_moment,
            ei_3=youngs_modulus * second_moment,
        )

    def as_row(self) -> Array:
        return np.array(
            [self.ea, self.ga_2, self.ga_3, self.gj, self.ei_2, self.ei_3],
            dtype=float,
        )


def initial_frames_from_polyline(
    points: Array, reference_normal: Sequence[float] = (0.0, 0.0, 1.0)
) -> Array:
    """Parallel-transported material frames along an open polyline.

    Returns ``(n, 3, 3)`` with columns ``[d1 d2 d3]``: ``d1`` tangent to the
    polyline, ``d2``/``d3`` carried along by minimal rotation so the frame field
    has no artificial twist. ``reference_normal`` only seeds ``d2`` at the first
    node.
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2:
        raise ValueError("points must be an (n >= 2, 3) polyline")

    segments = np.diff(points, axis=0)
    tangents = np.empty_like(points)
    tangents[0] = segments[0]
    tangents[-1] = segments[-1]
    if len(points) > 2:
        tangents[1:-1] = segments[:-1] + segments[1:]

    d1 = unit_vector(tangents[0])
    seed = np.asarray(reference_normal, dtype=float)
    seed = seed - np.dot(seed, d1) * d1
    if np.linalg.norm(seed) < 1e-8:
        seed = np.cross(d1, [1.0, 0.0, 0.0])
        if np.linalg.norm(seed) < 1e-8:
            seed = np.cross(d1, [0.0, 1.0, 0.0])
    d2 = unit_vector(seed)

    frames = np.empty((len(points), 3, 3))
    frames[0] = np.column_stack([d1, d2, np.cross(d1, d2)])
    for index in range(1, len(points)):
        transport = minimal_rotation(tangents[index - 1], tangents[index])
        frames[index] = transport @ frames[index - 1]
    return frames


def build_beam_elements(
    nodes: Array,
    connectivity: Array,
    sections: Sequence[BeamSection] | BeamSection,
    frames: Array,
    *,
    name: str = "beams",
) -> ElementSet:
    """Assemble a beam set, taking its reference strains from ``nodes``/``frames``.

    ``frames`` is the full ``(n_nodes, 3, 3)`` frame field (nodes with no beam
    attached are ignored). Reference strains are computed with
    :func:`beam_strains` in NumPy, so the built configuration is stress-free by
    construction even when a member is curved or pre-twisted.
    """
    nodes = np.asarray(nodes, dtype=float)
    connectivity = np.atleast_2d(np.asarray(connectivity, dtype=int))
    frames = np.asarray(frames, dtype=float)
    n_elements = len(connectivity)

    if isinstance(sections, BeamSection):
        sections = [sections] * n_elements
    if len(sections) != n_elements:
        raise ValueError(f"{n_elements} beam elements but {len(sections)} sections")

    params = np.empty((n_elements, len(PARAM_NAMES)))
    for element, (node_a, node_b) in enumerate(connectivity):
        rest_length = float(np.linalg.norm(nodes[node_b] - nodes[node_a]))
        gamma_0, omega_0 = beam_strains(
            nodes[node_a],
            nodes[node_b],
            frames[node_a],
            frames[node_b],
            rest_length,
            np,
        )
        params[element] = np.concatenate(
            [[rest_length], sections[element].as_row(), gamma_0, omega_0]
        )
    return ElementSet(name, TimoshenkoBeamKernel(), connectivity, params)

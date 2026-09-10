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

"""Constant-strain triangular membrane for the canopy fabric.

Three-node CST with a Saint-Venant-Kirchhoff plane-stress law, written in the
finite-strain (Green-Lagrange) measure so a canopy panel can rotate and billow
arbitrarily without spurious stiffening. Translation-only: no rotational DOF,
because a membrane carries no bending.

The reference triangle is flattened once into its own in-plane basis and stored
as the inverse edge matrix ``D0inv``, so the kernel only ever sees a 2x2 map:

    F = [a b] D0inv          (3x2 deformation gradient, a/b current edges)
    C = F^T F,   E = (C - I)/2
    U = A0 t * E/(2(1-nu^2)) [ (trE)^2 - 2(1-nu)(E11 E22 - E12^2) ]

Wrinkling
---------
A membrane cannot carry compression: left alone, the SVK law would let the
canopy mesh buckle into whatever compressive mode the discretisation offers,
and the result depends on mesh density rather than on physics. ``wrinkling``
switches on the relaxed (tension-field) energy in principal Green strains
``e1 >= e2``:

    taut       e2 + nu e1 >= 0    full SVK energy
    wrinkled   e1 > 0, otherwise  uniaxial: U = A0 t E/2 e1^2
    slack      e1 <= 0            U = 0

This is the classical Pipkin relaxed energy. It is exactly what a
minimum-energy formulation wants: the relaxed functional is the quasiconvex
envelope, so IPOPT converges to the *wrinkled* state directly instead of
chasing the near-singular tangent stiffness that a Newton solve on the
unrelaxed equations has to fight through. Solving the fabric by energy
minimisation is not a workaround here -- it is the better-posed formulation.

The three branches join C1, so the Hessian steps across the taut/wrinkled and
wrinkled/slack boundaries. ``regularization`` [-] rounds the principal-strain
square root (which is also where equal principal strains would be
non-differentiable); raise it if IPOPT stalls on a canopy where many elements
sit on a mode boundary at once.

``slack_stiffness_ratio`` is not cosmetic. A region that goes fully slack
stores *exactly* zero energy, so its block of the Hessian is exactly zero and
the Newton step is undefined -- IPOPT reports ``Error_In_Step_Computation`` on
a canopy clamped into a frame smaller than itself. Blending a small fraction
of the unrelaxed law back in,

    U = (1 - r) U_relaxed + r U_taut,

leaves the taut region untouched (there the two coincide) while giving slack
fabric a residual stiffness that keeps the tangent definite. It is also not a
lie: real fabric has some compressive and bending resistance. The default
``r = 1e-4`` is far below anything that shows in a load path.

Reference: Pipkin (1986) *IMA J. Appl. Math.* 36, 85-99; Roddeman et al. (1987)
*J. Appl. Mech.* 54, 884-892.
"""

from __future__ import annotations

from dataclasses import dataclass

import casadi as ca
import numpy as np

from .base import ElementSet, element_translations

Array = np.ndarray

PARAM_NAMES: tuple[str, ...] = (
    "area",
    "thickness",
    "youngs_modulus",
    "poisson_ratio",
    "d0inv_11",
    "d0inv_21",
    "d0inv_12",
    "d0inv_22",
)


def green_strain(edge_a, edge_b, d0_inverse, xp):
    """In-plane Green-Lagrange strain ``(E11, E22, E12)`` of a CST triangle.

    The single place this measure is written: the CasADi kernel and the NumPy
    wrinkling diagnostic both call it.
    """
    columns = (
        np.column_stack([edge_a, edge_b]) if xp is np else xp.horzcat(edge_a, edge_b)
    )
    deformation = columns @ d0_inverse
    cauchy_green = deformation.T @ deformation
    return (
        0.5 * (cauchy_green[0, 0] - 1.0),
        0.5 * (cauchy_green[1, 1] - 1.0),
        0.5 * cauchy_green[0, 1],
    )


#: Regime codes returned by :func:`membrane_regimes`.
SLACK, WRINKLED, TAUT = 0, 1, 2


@dataclass(frozen=True)
class MembraneKernel:
    """Strain energy of one CST membrane triangle."""

    wrinkling: bool = True
    regularization: float = 1e-12
    slack_stiffness_ratio: float = 1e-4

    nodes_per_element: int = 3
    rotational_nodes: tuple[int, ...] = ()
    param_names: tuple[str, ...] = PARAM_NAMES

    def energy(self, q, frames, p):
        x_1 = element_translations(q, 0)
        edge_a = element_translations(q, 1) - x_1
        edge_b = element_translations(q, 2) - x_1

        d0_inverse = ca.vertcat(ca.horzcat(p[4], p[6]), ca.horzcat(p[5], p[7]))
        strain_11, strain_22, strain_12 = green_strain(
            edge_a, edge_b, d0_inverse, ca
        )

        area, thickness = p[0], p[1]
        modulus, poisson = p[2], p[3]
        volume = area * thickness

        trace = strain_11 + strain_22
        taut = (modulus / (2.0 * (1.0 - poisson ** 2))) * (
            trace ** 2
            - 2.0 * (1.0 - poisson) * (strain_11 * strain_22 - strain_12 ** 2)
        )
        if not self.wrinkling:
            return volume * taut

        # Principal Green strains. The regularisation only rounds the square
        # root, which is where equal principal strains would be
        # non-differentiable; the taut branch above stays exact, so the bulk of
        # a biaxially loaded canopy carries no regularisation error at all.
        mean = 0.5 * trace
        deviator = ca.sqrt(
            (0.5 * (strain_11 - strain_22)) ** 2
            + strain_12 ** 2
            + self.regularization
        )
        principal_1 = mean + deviator
        principal_2 = mean - deviator

        wrinkled = 0.5 * modulus * principal_1 ** 2
        relaxed = ca.if_else(
            principal_2 + poisson * principal_1 >= 0.0,
            taut,
            ca.if_else(principal_1 > 0.0, wrinkled, 0.0),
        )
        ratio = self.slack_stiffness_ratio
        return volume * ((1.0 - ratio) * relaxed + ratio * taut)


def membrane_reference(nodes: Array, triangle: Array) -> tuple[float, Array]:
    """Reference area and inverse in-plane edge matrix of one triangle.

    The reference triangle is expressed in an orthonormal basis of its own
    plane, so the kernel never needs the out-of-plane direction.
    """
    node_a, node_b, node_c = (np.asarray(nodes[i], dtype=float) for i in triangle)
    edge_a, edge_b = node_b - node_a, node_c - node_a

    normal = np.cross(edge_a, edge_b)
    area = 0.5 * float(np.linalg.norm(normal))
    if area < 1e-14:
        raise ValueError(f"degenerate membrane triangle {tuple(triangle)}")

    basis_1 = edge_a / np.linalg.norm(edge_a)
    basis_2 = np.cross(normal / np.linalg.norm(normal), basis_1)
    edge_matrix = np.array(
        [
            [float(edge_a @ basis_1), float(edge_b @ basis_1)],
            [float(edge_a @ basis_2), float(edge_b @ basis_2)],
        ]
    )
    return area, np.linalg.inv(edge_matrix)


def build_membrane_elements(
    nodes: Array,
    triangles: Array,
    thickness: float | Array,
    youngs_modulus: float | Array,
    poisson_ratio: float | Array,
    *,
    name: str = "canopy",
    wrinkling: bool = True,
    regularization: float = 1e-12,
    slack_stiffness_ratio: float = 1e-4,
) -> ElementSet:
    """Assemble a canopy element set; the reference mesh is stress-free.

    ``thickness`` is the fabric thickness [m] and ``youngs_modulus`` its
    in-plane modulus [Pa]; both accept a scalar or one value per triangle, so a
    reinforced panel or a heavier leading-edge strip is just a different row.
    """
    triangles = np.atleast_2d(np.asarray(triangles, dtype=int))
    n_elements = len(triangles)

    def _column(value) -> Array:
        return np.broadcast_to(np.asarray(value, dtype=float), (n_elements,)).copy()

    params = np.empty((n_elements, len(PARAM_NAMES)))
    params[:, 1] = _column(thickness)
    params[:, 2] = _column(youngs_modulus)
    params[:, 3] = _column(poisson_ratio)
    for element, triangle in enumerate(triangles):
        area, d0_inverse = membrane_reference(nodes, triangle)
        params[element, 0] = area
        params[element, 4:8] = d0_inverse.reshape(-1, order="F")

    kernel = MembraneKernel(
        wrinkling=wrinkling,
        regularization=regularization,
        slack_stiffness_ratio=slack_stiffness_ratio,
    )
    return ElementSet(name, kernel, triangles, params)


def membrane_regimes(positions: Array, element_set: ElementSet) -> dict[str, Array]:
    """Per-triangle wrinkling diagnostic for a solved configuration.

    Returns the principal Green strains and a regime code per element
    (:data:`SLACK`, :data:`WRINKLED`, :data:`TAUT`), using the same strain
    measure and the same discriminant the energy kernel branches on -- so the
    picture it draws is the state the solver actually chose, not a
    re-derivation of it.
    """
    positions = np.asarray(positions, dtype=float).reshape(-1, 3)
    kernel = element_set.kernel
    params = element_set.params

    principal_1 = np.empty(element_set.n_elements)
    principal_2 = np.empty(element_set.n_elements)
    for element, triangle in enumerate(element_set.connectivity):
        row = params[element]
        d0_inverse = np.array([[row[4], row[6]], [row[5], row[7]]])
        strain_11, strain_22, strain_12 = green_strain(
            positions[triangle[1]] - positions[triangle[0]],
            positions[triangle[2]] - positions[triangle[0]],
            d0_inverse,
            np,
        )
        mean = 0.5 * (strain_11 + strain_22)
        deviator = np.sqrt(
            (0.5 * (strain_11 - strain_22)) ** 2
            + strain_12 ** 2
            + kernel.regularization
        )
        principal_1[element] = mean + deviator
        principal_2[element] = mean - deviator

    poisson = params[:, 3]
    regime = np.where(
        principal_2 + poisson * principal_1 >= 0.0,
        TAUT,
        np.where(principal_1 > 0.0, WRINKLED, SLACK),
    )
    return {
        "principal_1": principal_1,
        "principal_2": principal_2,
        "regime": regime,
    }

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

"""Total potential energy of a structural model, assembled for scale.

    Pi(X) = sum_sets U_set(X) - f_ext . x - m_ext . psi + anchor

The one design decision in this module: **the element loop happens once, in
SX, and is then mapped**. Each kernel is compiled to a small
``casadi.Function`` over one element and evaluated across the whole set with
``Function.map``, so the objective graph holds one mapped node per element
*type* rather than one subgraph per element. Graph construction, ``gradient``
and ``hessian`` therefore cost O(number of element types), not O(number of
elements), which is what lets the canopy go from tens of springs to tens of
thousands of triangles without the graph build becoming the bottleneck. A
Python ``for`` loop over elements building MX -- the shape of the existing
``aerostructural/pss/structural_nlp.py`` -- is what this replaces.

The Hessian keeps the sparsity of the element connectivity graph, so IPOPT's
sparse linear algebra sees a mesh-stencil matrix and scales accordingly.

Everything that can change between solves is an NLP *parameter*, not a baked
constant: reference frames, every element parameter column (rest lengths for
actuation, stiffnesses for a ramp), the external loads and the anchor. One
build serves a whole coupled run.

The anchor is a weak spring to the seed positions, there only to pin a node
that no taut element reaches and whose equilibrium position would otherwise be
indeterminate. It biases the reported force balance by at most
``anchor_stiffness * |x - x_seed|``, so the default is small enough
(1e-6 N/m, i.e. a microNewton per metre moved) to sit far below any coupling
tolerance. Raise it only if a fully slack region makes the Hessian singular
beyond what the IPOPT inertia correction handles.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import casadi as ca
import numpy as np

from .elements.base import ElementSet
from .model import StructuralModel, StructuralState
from .rotations import rotation_vector

Array = np.ndarray


def _compile_kernel(element_set: ElementSet) -> ca.Function:
    """Compile one element kernel to an SX ``Function(q, frames, p) -> U``."""
    kernel = element_set.kernel
    n_dof = 3 * kernel.nodes_per_element + 3 * len(kernel.rotational_nodes)
    n_frames = 9 * len(kernel.rotational_nodes)

    q = ca.SX.sym("q", n_dof)
    frames = ca.SX.sym("frames", n_frames)
    p = ca.SX.sym("p", len(kernel.param_names))
    return ca.Function(
        f"energy_{element_set.name}", [q, frames, p], [element_set.kernel.energy(q, frames, p)]
    )


@dataclass(frozen=True)
class ParameterLayout:
    """Slice boundaries of the packed NLP parameter vector."""

    n_frame_entries: int
    element_sizes: tuple[int, ...]
    n_translation_dof: int
    n_rotation_dof: int

    @property
    def size(self) -> int:
        return (
            self.n_frame_entries
            + sum(self.element_sizes)
            + self.n_translation_dof
            + self.n_rotation_dof
            + self.n_translation_dof
        )


class PotentialEnergy:
    """Compiled total potential energy of a :class:`StructuralModel`."""

    def __init__(
        self,
        model: StructuralModel,
        *,
        anchor_stiffness: float = 1e-6,
        parallelization: str = "serial",
        n_threads: int = 1,
    ) -> None:
        self.model = model
        self.layout = model.layout
        self.anchor_stiffness = float(anchor_stiffness)

        n_dof = self.layout.n_dof
        n_translation = self.layout.n_translation_dof
        n_rotation = 3 * self.layout.n_rotational_nodes
        n_frame_entries = 9 * self.layout.n_rotational_nodes

        unknowns = ca.MX.sym("X", n_dof)
        frames = ca.MX.sym("frames", n_frame_entries)
        forces = ca.MX.sym("f_ext", n_translation)
        moments = ca.MX.sym("m_ext", n_rotation)
        anchor = ca.MX.sym("x_anchor", n_translation)

        strain_energy = ca.MX(0)
        element_parameters: list[ca.MX] = []
        element_sizes: list[int] = []

        for element_set in model.element_sets:
            n_elements = element_set.n_elements
            n_params = len(element_set.kernel.param_names)
            parameters = ca.MX.sym(f"p_{element_set.name}", n_params, n_elements)
            element_parameters.append(parameters)
            element_sizes.append(n_params * n_elements)

            dof_map = self.layout.element_dof_indices(element_set)
            gathered_dof = ca.reshape(
                unknowns[dof_map.flatten(order="F").tolist()],
                dof_map.shape[0],
                n_elements,
            )

            frame_map = self.layout.element_frame_indices(element_set)
            if frame_map.shape[0]:
                gathered_frames = ca.reshape(
                    frames[frame_map.flatten(order="F").tolist()],
                    frame_map.shape[0],
                    n_elements,
                )
            else:
                gathered_frames = ca.MX(0, n_elements)

            mapped = _compile_kernel(element_set).map(
                n_elements, parallelization, n_threads
            ) if parallelization != "serial" else _compile_kernel(element_set).map(
                n_elements
            )
            strain_energy = strain_energy + ca.sum2(
                mapped(gathered_dof, gathered_frames, parameters)
            )

        translations = unknowns[:n_translation]
        rotations = unknowns[n_translation:]
        external_work = ca.dot(forces, translations)
        if n_rotation:
            # A moment is work-conjugate to the exponential-map rotation vector,
            # not to the Rodrigues vector the DOF carry. Using ``-M . psi``
            # directly is correct only to first order in psi, and the error is
            # visible the moment a node rotates appreciably within one solve --
            # the roll-up benchmark makes the whole answer load-step dependent.
            # NOTE: even this is exact only for a moment about a fixed axis. A
            # constant ("dead") moment in 3-D is genuinely non-conservative, so
            # it has no potential and cannot be posed as an energy minimisation
            # at all; see the module docstring.
            turned = ca.vertcat(
                *[
                    rotation_vector(rotations[3 * slot: 3 * slot + 3], ca)
                    for slot in range(self.layout.n_rotational_nodes)
                ]
            )
            external_work = external_work + ca.dot(moments, turned)
        anchor_energy = 0.5 * self.anchor_stiffness * ca.sumsqr(translations - anchor)

        self.parameter_layout = ParameterLayout(
            n_frame_entries=n_frame_entries,
            element_sizes=tuple(element_sizes),
            n_translation_dof=n_translation,
            n_rotation_dof=n_rotation,
        )
        parameter_vector = ca.vertcat(
            frames,
            *[ca.vec(block) for block in element_parameters],
            forces,
            moments,
            anchor,
        )

        self.unknowns = unknowns
        self.parameters = parameter_vector
        self.strain_energy = strain_energy
        self.objective = strain_energy - external_work + anchor_energy

        #: Internal load vector ``-dU/dX``: nodal forces then nodal moments.
        #: The anchor and the external terms are deliberately excluded, so this
        #: is the structure resisting load and is directly comparable with the
        #: applied loads to form a coupling residual.
        self.internal_load = ca.Function(
            "internal_load",
            [unknowns, parameter_vector],
            [-ca.gradient(strain_energy, unknowns)],
        )
        self.total_energy = ca.Function(
            "total_energy", [unknowns, parameter_vector], [strain_energy]
        )
        self._tangent_stiffness: ca.Function | None = None

    @property
    def nlp(self) -> dict:
        """The ``casadi.nlpsol`` problem dictionary."""
        return {"x": self.unknowns, "p": self.parameters, "f": self.objective}

    def tangent_stiffness(self) -> ca.Function:
        """``d2U/dX2`` -- built on first use, for linear buckling or modal work."""
        if self._tangent_stiffness is None:
            hessian, _ = ca.hessian(self.strain_energy, self.unknowns)
            self._tangent_stiffness = ca.Function(
                "tangent_stiffness", [self.unknowns, self.parameters], [hessian]
            )
        return self._tangent_stiffness

    def pack_unknowns(self, translations: Array, rotations: Array | None = None) -> Array:
        """Flatten positions and incremental rotations into ``X``."""
        flat = [np.asarray(translations, dtype=float).reshape(-1)]
        if self.layout.n_rotational_nodes:
            increments = (
                np.zeros(3 * self.layout.n_rotational_nodes)
                if rotations is None
                else np.asarray(rotations, dtype=float).reshape(-1)
            )
            flat.append(increments)
        return np.concatenate(flat)

    def unpack_unknowns(self, unknowns: Array) -> tuple[Array, Array]:
        """Split ``X`` back into ``(positions (n, 3), increments (r, 3))``."""
        flat = np.asarray(unknowns, dtype=float).reshape(-1)
        split = self.layout.n_translation_dof
        return flat[:split].reshape(-1, 3), flat[split:].reshape(-1, 3)

    def pack_parameters(
        self,
        state: StructuralState,
        element_sets: Sequence[ElementSet],
        forces: Array,
        moments: Array | None,
        anchor: Array,
    ) -> Array:
        """Assemble the parameter vector in the order the graph was built."""
        blocks = [np.asarray(state.frames, dtype=float).reshape(-1)]
        for element_set, expected in zip(element_sets, self.parameter_layout.element_sizes):
            block = np.asarray(element_set.params, dtype=float).reshape(-1)
            if block.size != expected:
                raise ValueError(
                    f"element set {element_set.name!r} has {block.size} parameter "
                    f"entries, the compiled graph expects {expected}"
                )
            blocks.append(block)
        blocks.append(np.asarray(forces, dtype=float).reshape(-1))
        n_rotation = self.parameter_layout.n_rotation_dof
        blocks.append(
            np.zeros(n_rotation)
            if moments is None
            else np.asarray(moments, dtype=float).reshape(-1)
        )
        blocks.append(np.asarray(anchor, dtype=float).reshape(-1))

        packed = np.concatenate(blocks)
        if packed.size != self.parameter_layout.size:
            raise ValueError(
                f"packed {packed.size} parameters, expected "
                f"{self.parameter_layout.size}"
            )
        return packed

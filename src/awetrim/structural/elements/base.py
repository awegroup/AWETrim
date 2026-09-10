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

"""Element contract for the minimum-energy structural model.

An element type is described by an :class:`ElementKernel`: the strain energy of
*one* element, written symbolically in CasADi ``SX``. The assembler
(:mod:`awetrim.structural.energy`) compiles that kernel once and evaluates it
over the whole element set with ``casadi.Function.map``, so the objective graph
stays O(1) in the number of elements. Kernels must therefore never loop over
elements, index a global array, or close over per-element data -- everything
element-specific arrives through ``params``.

Kernel signature
----------------
``energy(q, frames, p) -> SX`` (scalar), where

``q``       flat element degrees of freedom: the translations of every local
            node in connectivity order, followed by the incremental rotation
            vectors of the local nodes listed in ``rotational_nodes``.
            Length ``3 * nodes_per_element + 3 * len(rotational_nodes)``.
``frames``  row-major flattened reference frames of those same rotational
            nodes, length ``9 * len(rotational_nodes)``; empty for
            translation-only elements.
``p``       the element's parameter row, length ``len(param_names)``.

Splitting ``q`` and ``frames`` is done for the kernel by
:func:`element_translations`, :func:`element_rotation` and
:func:`element_frame`, so a kernel never hard-codes an offset.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence, runtime_checkable

import numpy as np

from ..rotations import unflatten_frame

Array = np.ndarray


@runtime_checkable
class ElementKernel(Protocol):
    """Strain energy of a single element, as a CasADi ``SX`` expression."""

    #: Nodes referenced by one element (2 for cables/beams, 3 for triangles).
    nodes_per_element: int

    #: Local node indices that carry a rotational frame; ``()`` if none.
    rotational_nodes: tuple[int, ...]

    #: Names of the columns of the element parameter table, in order.
    param_names: tuple[str, ...]

    def energy(self, q: Any, frames: Any, p: Any) -> Any:
        """Return the scalar strain energy [J] of one element."""


def element_translations(q: Any, node: int) -> Any:
    """Translation DOF of local ``node`` out of the flat element vector."""
    return q[3 * node: 3 * node + 3]


def element_rotation(q: Any, nodes_per_element: int, slot: int) -> Any:
    """Incremental rotation vector of the ``slot``-th rotational local node."""
    offset = 3 * nodes_per_element + 3 * slot
    return q[offset: offset + 3]


def element_frame(frames: Any, slot: int, xp) -> Any:
    """Reference frame of the ``slot``-th rotational local node, as a 3x3."""
    return unflatten_frame(frames[9 * slot: 9 * slot + 9], xp)


@dataclass(frozen=True)
class ElementSet:
    """One element kernel applied to a table of elements.

    ``connectivity`` is ``(n_elements, kernel.nodes_per_element)`` of global
    node indices; ``params`` is ``(n_elements, len(kernel.param_names))``.
    Parameters stay live NLP parameters rather than baked constants, so rest
    lengths, stiffness ramps and section properties can change between solves
    without rebuilding the solver.
    """

    name: str
    kernel: ElementKernel
    connectivity: Array
    params: Array

    def __post_init__(self) -> None:
        connectivity = np.atleast_2d(np.asarray(self.connectivity, dtype=int))
        params = np.atleast_2d(np.asarray(self.params, dtype=float))
        object.__setattr__(self, "connectivity", connectivity)
        object.__setattr__(self, "params", params)

        expected_nodes = self.kernel.nodes_per_element
        if connectivity.shape[1] != expected_nodes:
            raise ValueError(
                f"element set {self.name!r}: connectivity has "
                f"{connectivity.shape[1]} columns, kernel expects {expected_nodes}"
            )
        expected_params = len(self.kernel.param_names)
        if params.shape[1] != expected_params:
            raise ValueError(
                f"element set {self.name!r}: params has {params.shape[1]} columns, "
                f"kernel expects {expected_params} {self.kernel.param_names}"
            )
        if params.shape[0] != connectivity.shape[0]:
            raise ValueError(
                f"element set {self.name!r}: {connectivity.shape[0]} elements but "
                f"{params.shape[0]} parameter rows"
            )

    @property
    def n_elements(self) -> int:
        return int(self.connectivity.shape[0])

    @property
    def rotational_node_indices(self) -> Array:
        """Global indices of every node this set needs a frame for."""
        local = self.kernel.rotational_nodes
        if not local:
            return np.empty(0, dtype=int)
        return np.unique(self.connectivity[:, list(local)])

    def param_column(self, name: str) -> Array:
        """Read one named parameter column (e.g. ``"rest_length"``)."""
        return self.params[:, self.kernel.param_names.index(name)]

    def with_param_column(self, name: str, values: Sequence[float]) -> "ElementSet":
        """Copy with one parameter column replaced -- for actuation updates."""
        params = self.params.copy()
        params[:, self.kernel.param_names.index(name)] = np.asarray(
            values, dtype=float
        )
        return ElementSet(self.name, self.kernel, self.connectivity, params)

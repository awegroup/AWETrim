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

"""Degrees of freedom, configuration state, and the assembled structural model.

Layout
------
The unknown vector is

    X = [ x_0 ... x_(n-1) | psi_0 ... psi_(r-1) ]

with three translations for **every** node and three incremental rotations only
for the ``r`` nodes that some beam element needs a frame for. Cable, pulley and
membrane nodes therefore stay three-DOF; adding beams does not inflate the
canopy.

Incremental rotations
---------------------
``psi_j`` is measured against a stored reference frame ``R_ref_j`` held in
:class:`StructuralState`, so the configuration is
``R_j = cayley(psi_j) R_ref_j`` and the solve always starts from ``psi = 0``.
Two reasons, both of which matter for an NLP:

* The Cayley map is singular at ``phi = pi``. Total rotations of a flapping
  canopy batten reach that; per-solve increments do not.
* With an applied nodal moment the potential ``-M . psi`` is only work-conjugate
  to first order in ``psi``, i.e. exactly at ``psi = 0``. Absorbing the solved
  increment into ``R_ref`` and re-solving drives ``psi -> 0``, so the converged
  fixed point satisfies the true moment balance rather than a
  parameterisation-dependent one. :class:`~awetrim.structural.solver`
  does that automatically; with force-only loading a single solve is already
  exact.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

from .elements.base import ElementSet
from .rotations import cayley, orthonormalize

Array = np.ndarray


@dataclass(frozen=True)
class DofLayout:
    """Maps nodes and element sets onto slices of the unknown vector."""

    n_nodes: int
    rotational_nodes: Array

    def __post_init__(self) -> None:
        rotational = np.unique(np.asarray(self.rotational_nodes, dtype=int))
        if rotational.size and (rotational.min() < 0 or rotational.max() >= self.n_nodes):
            raise ValueError("rotational node index outside the node range")
        object.__setattr__(self, "rotational_nodes", rotational)

        slots = np.full(self.n_nodes, -1, dtype=int)
        slots[rotational] = np.arange(rotational.size)
        object.__setattr__(self, "_rotation_slots", slots)

    @classmethod
    def from_element_sets(
        cls, n_nodes: int, element_sets: Iterable[ElementSet]
    ) -> "DofLayout":
        rotational = [element_set.rotational_node_indices for element_set in element_sets]
        stacked = np.concatenate(rotational) if rotational else np.empty(0, dtype=int)
        return cls(n_nodes=int(n_nodes), rotational_nodes=stacked)

    @property
    def n_rotational_nodes(self) -> int:
        return int(self.rotational_nodes.size)

    @property
    def n_translation_dof(self) -> int:
        return 3 * self.n_nodes

    @property
    def n_dof(self) -> int:
        return 3 * self.n_nodes + 3 * self.n_rotational_nodes

    def rotation_slot(self, node: int) -> int:
        """Position of ``node`` in the rotational block, or -1 if it has none."""
        return int(self._rotation_slots[node])

    def translation_dof(self, nodes: Array) -> Array:
        """DOF indices of the translations of ``nodes``; shape ``(len, 3)``."""
        nodes = np.asarray(nodes, dtype=int)
        return 3 * nodes[..., None] + np.arange(3)

    def rotation_dof(self, nodes: Array) -> Array:
        """DOF indices of the incremental rotations of ``nodes``."""
        slots = self._rotation_slots[np.asarray(nodes, dtype=int)]
        if np.any(slots < 0):
            raise ValueError("node has no rotational DOF in this layout")
        return self.n_translation_dof + 3 * slots[..., None] + np.arange(3)

    def frame_entries(self, nodes: Array) -> Array:
        """Indices into the flat reference-frame vector; shape ``(len, 9)``."""
        slots = self._rotation_slots[np.asarray(nodes, dtype=int)]
        if np.any(slots < 0):
            raise ValueError("node has no reference frame in this layout")
        return 9 * slots[..., None] + np.arange(9)

    def element_dof_indices(self, element_set: ElementSet) -> Array:
        """``(dof_per_element, n_elements)`` gather map into ``X``.

        Column ``e`` is the flat DOF vector the kernel expects for element
        ``e``: all local translations in connectivity order, then the
        incremental rotations of the local nodes the kernel declares.
        """
        connectivity = element_set.connectivity
        blocks = [self.translation_dof(connectivity[:, local]).T
                  for local in range(element_set.kernel.nodes_per_element)]
        blocks += [self.rotation_dof(connectivity[:, local]).T
                   for local in element_set.kernel.rotational_nodes]
        return np.vstack(blocks)

    def element_frame_indices(self, element_set: ElementSet) -> Array:
        """``(9 * n_rotational_local, n_elements)`` gather map into the frames."""
        local_nodes = element_set.kernel.rotational_nodes
        if not local_nodes:
            return np.empty((0, element_set.n_elements), dtype=int)
        connectivity = element_set.connectivity
        return np.vstack(
            [self.frame_entries(connectivity[:, local]).T for local in local_nodes]
        )


@dataclass
class StructuralState:
    """A configuration: nodal positions plus the reference frames.

    ``frames`` is ordered by rotational slot, i.e. aligned with
    ``layout.rotational_nodes``, and is empty when the model has no beams.
    """

    positions: Array
    frames: Array

    def __post_init__(self) -> None:
        self.positions = np.asarray(self.positions, dtype=float).reshape(-1, 3)
        frames = np.asarray(self.frames, dtype=float)
        self.frames = frames.reshape(-1, 3, 3) if frames.size else frames.reshape(0, 3, 3)

    @property
    def n_nodes(self) -> int:
        return int(self.positions.shape[0])

    def copy(self) -> "StructuralState":
        return StructuralState(self.positions.copy(), self.frames.copy())

    def absorbed(self, translations: Array, increments: Array) -> "StructuralState":
        """New state with solved translations and the increments folded into the frames.

        Folding rather than storing ``psi`` is what keeps every solve starting
        from ``psi = 0``; the frames are re-projected onto SO(3) so repeated
        absorption cannot let them drift.
        """
        positions = np.asarray(translations, dtype=float).reshape(-1, 3)
        increments = np.asarray(increments, dtype=float).reshape(-1, 3)
        if increments.shape[0] != self.frames.shape[0]:
            raise ValueError(
                f"{increments.shape[0]} rotation increments for "
                f"{self.frames.shape[0]} frames"
            )
        frames = np.stack(
            [
                orthonormalize(cayley(increments[slot], np) @ self.frames[slot])
                for slot in range(self.frames.shape[0])
            ]
        ) if self.frames.shape[0] else self.frames.copy()
        return StructuralState(positions, frames)


@dataclass
class StructuralModel:
    """Element sets plus the reference configuration and the pinned DOF.

    This object owns the structural *definition*; :class:`StructuralState` owns
    the configuration and the solver owns the numerics. Nothing here is CasADi:
    symbolics are created inside :mod:`awetrim.structural.energy` and never
    cross this boundary.
    """

    nodes: Array
    element_sets: Sequence[ElementSet]
    node_frames: Array | None = None
    fixed_translation_nodes: Sequence[int] = field(default_factory=tuple)
    fixed_rotation_nodes: Sequence[int] = field(default_factory=tuple)
    layout: DofLayout = field(init=False)

    def __post_init__(self) -> None:
        self.nodes = np.asarray(self.nodes, dtype=float).reshape(-1, 3)
        self.element_sets = tuple(self.element_sets)
        if not self.element_sets:
            raise ValueError("a structural model needs at least one element set")

        self.layout = DofLayout.from_element_sets(len(self.nodes), self.element_sets)
        self.fixed_translation_nodes = tuple(int(i) for i in self.fixed_translation_nodes)
        self.fixed_rotation_nodes = tuple(int(i) for i in self.fixed_rotation_nodes)

        if self.layout.n_rotational_nodes and self.node_frames is None:
            raise ValueError(
                "the model has beam elements, so node_frames (n_nodes, 3, 3) "
                "is required -- build it with "
                "elements.beam.initial_frames_from_polyline"
            )
        for node in self.fixed_rotation_nodes:
            if self.layout.rotation_slot(node) < 0:
                raise ValueError(f"node {node} has no rotational DOF to fix")

    @property
    def n_nodes(self) -> int:
        return int(self.nodes.shape[0])

    def initial_state(self) -> StructuralState:
        """Reference configuration as a state, ready to seed a solve."""
        rotational = self.layout.rotational_nodes
        if rotational.size == 0:
            return StructuralState(self.nodes.copy(), np.empty((0, 3, 3)))
        frames = np.asarray(self.node_frames, dtype=float)[rotational]
        return StructuralState(self.nodes.copy(), frames.copy())

    def element_set(self, name: str) -> ElementSet:
        for element_set in self.element_sets:
            if element_set.name == name:
                return element_set
        available = ", ".join(s.name for s in self.element_sets)
        raise KeyError(f"no element set {name!r}; have {available}")

    def replaced(self, element_set: ElementSet) -> "StructuralModel":
        """Copy with the like-named element set swapped -- for actuation updates.

        Only the parameter *values* may differ; the connectivity and kernel are
        what the compiled solver graph was built from.
        """
        sets = tuple(
            element_set if existing.name == element_set.name else existing
            for existing in self.element_sets
        )
        return StructuralModel(
            self.nodes,
            sets,
            self.node_frames,
            self.fixed_translation_nodes,
            self.fixed_rotation_nodes,
        )

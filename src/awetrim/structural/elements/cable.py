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

"""Axial (cable / spring) elements and frictionless pulley ropes.

These reproduce the spring physics of the existing PSS particle system
exactly, so the new model can be validated against
``aerostructural/pss/structural_nlp.py`` on the current bridle before any beam
or membrane element is switched on:

* :class:`CableKernel` -- a linear spring, optionally tension-only.
* :class:`PulleyKernel` -- a rope running over a frictionless pulley: two arms
  ``i-j`` and ``j-k`` sharing one stretch ``|xj-xi| + |xk-xj| - l0``, which is
  what makes the pulley node free to slide along the rope.

Both are translation-only, so they carry no rotational DOF and no frames.
"""

from __future__ import annotations

from dataclasses import dataclass

import casadi as ca
import numpy as np

from .base import ElementSet, element_translations

Array = np.ndarray


def _length(q, node_a: int, node_b: int):
    return ca.norm_2(
        element_translations(q, node_b) - element_translations(q, node_a)
    )


@dataclass(frozen=True)
class CableKernel:
    """Linear axial spring: ``U = 1/2 k s^2`` with ``s`` the stretch.

    With ``tension_only`` the compressive branch is cut, ``s -> max(0, s)``,
    which models a slack line. That cut is C1 but not C2 at ``s = 0``: the
    stiffness jumps, so the Hessian is discontinuous exactly on the
    taut/slack boundary. IPOPT copes (the existing solver uses the same cut),
    but it is the reason ``slack_smoothing`` exists -- set it to a small
    positive stretch [m] to round the corner over that width and recover a
    continuous Hessian, at the cost of a tiny spurious compressive stiffness.
    """

    tension_only: bool = True
    slack_smoothing: float = 0.0

    nodes_per_element: int = 2
    rotational_nodes: tuple[int, ...] = ()
    param_names: tuple[str, ...] = ("rest_length", "stiffness")

    def energy(self, q, frames, p):
        stretch = _length(q, 0, 1) - p[0]
        return 0.5 * p[1] * _positive_part(stretch, self.tension_only,
                                           self.slack_smoothing) ** 2


@dataclass(frozen=True)
class PulleyKernel:
    """Rope over a frictionless pulley at the middle node.

    One shared stretch over both arms means the rope tension is equal either
    side of the pulley -- the defining property of a frictionless sheave -- and
    the pulley node finds its own position along the rope.
    """

    tension_only: bool = True
    slack_smoothing: float = 0.0

    nodes_per_element: int = 3
    rotational_nodes: tuple[int, ...] = ()
    param_names: tuple[str, ...] = ("rest_length", "stiffness")

    def energy(self, q, frames, p):
        stretch = _length(q, 0, 1) + _length(q, 1, 2) - p[0]
        return 0.5 * p[1] * _positive_part(stretch, self.tension_only,
                                           self.slack_smoothing) ** 2


def _positive_part(stretch, tension_only: bool, smoothing: float):
    """``max(0, s)`` for a tension-only element, optionally smoothed."""
    if not tension_only:
        return stretch
    if smoothing <= 0.0:
        return ca.fmax(0.0, stretch)
    # Smooth hinge: within smoothing/2 of max(0, s) everywhere, and smooth
    # across the taut/slack boundary instead of only C1.
    return 0.5 * (stretch + ca.sqrt(stretch ** 2 + smoothing ** 2))


def line_tensions(positions: Array, element_set: ElementSet) -> Array:
    """Tension [N] in each cable or pulley rope of a solved configuration.

    Taken from the kernel's own energy, not re-derived: the gradient at the
    element's LAST node -- a cable's far end, a pulley rope's end past the
    sheave -- is ``T e`` with ``e`` the unit vector into that node along the
    line, so ``T = dU/dx_last . e``. The slack cut and any ``slack_smoothing``
    are therefore exactly the solver's: a slack line reads 0, and a
    compression-capable element in compression reads negative.

    ``positions`` must be the configuration the solver RETURNED. A relaxed or
    interpolated shape is not in equilibrium, and on a dyneema line a few
    millimetres of spurious stretch is hundreds of newtons.
    """
    kernel = element_set.kernel
    nodes = kernel.nodes_per_element
    q = ca.SX.sym("q", 3 * nodes)
    p = ca.SX.sym("p", len(kernel.param_names))
    gradient = ca.gradient(kernel.energy(q, ca.SX(0, 1), p), q)
    last = element_translations(q, nodes - 1)
    into_last = last - element_translations(q, nodes - 2)
    tension = ca.dot(gradient[3 * (nodes - 1):], into_last / ca.norm_2(into_last))
    evaluate = ca.Function("line_tension", [q, p], [tension]).map(element_set.n_elements)

    positions = np.asarray(positions, dtype=float).reshape(-1, 3)
    element_positions = positions[element_set.connectivity].reshape(element_set.n_elements, -1)
    return np.asarray(evaluate(element_positions.T, element_set.params.T)).ravel()


def build_cable_elements(
    connectivity: Array,
    rest_lengths: Array,
    stiffness: Array,
    *,
    name: str = "cables",
    tension_only: bool = True,
    slack_smoothing: float = 0.0,
) -> ElementSet:
    """Assemble a two-node cable set from parallel arrays."""
    params = np.column_stack(
        [np.asarray(rest_lengths, dtype=float), np.asarray(stiffness, dtype=float)]
    )
    kernel = CableKernel(tension_only=tension_only, slack_smoothing=slack_smoothing)
    return ElementSet(name, kernel, np.asarray(connectivity, dtype=int), params)


def build_pulley_elements(
    connectivity: Array,
    rest_lengths: Array,
    stiffness: Array,
    *,
    name: str = "pulleys",
    tension_only: bool = True,
    slack_smoothing: float = 0.0,
) -> ElementSet:
    """Assemble a three-node pulley-rope set ``(i, pulley, k)``."""
    params = np.column_stack(
        [np.asarray(rest_lengths, dtype=float), np.asarray(stiffness, dtype=float)]
    )
    kernel = PulleyKernel(tension_only=tension_only, slack_smoothing=slack_smoothing)
    return ElementSet(name, kernel, np.asarray(connectivity, dtype=int), params)

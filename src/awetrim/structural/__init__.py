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

"""Standalone minimum-energy structural model.

A self-contained structural library: cables, frictionless pulleys, geometrically
exact Timoshenko beams and wrinkling membrane fabric, assembled into one total
potential energy and solved for static equilibrium with IPOPT.

    nodes + element sets  ->  StructuralModel
    StructuralModel       ->  MinimumEnergySolver   (compiled once)
    solver.solve(forces)  ->  StructuralSolution    (state, residual, energy)

Deliberately isolated: this package imports only NumPy and CasADi. It does not
know about PSS, VSM, the aerostructural coupling loop or the kite YAML schema,
and nothing in ``aerostructural/`` imports it yet. Coupling it in belongs in an
adapter under ``aerostructural/``, not here -- see ``AGENTS.md``.

Example
-------
>>> import numpy as np
>>> from awetrim.structural import MinimumEnergySolver, StructuralModel
>>> from awetrim.structural.elements import build_cable_elements
>>> nodes = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
>>> cables = build_cable_elements([[0, 1]], rest_lengths=[1.0], stiffness=[1e4])
>>> model = StructuralModel(nodes, [cables], fixed_translation_nodes=[0])
>>> solution = MinimumEnergySolver(model).solve(np.array([[0.0, 0, 0], [10.0, 0, 0]]))
>>> bool(solution.converged), round(solution.state.positions[1, 0], 4)
(True, 1.001)
"""

from .energy import PotentialEnergy
from .model import DofLayout, StructuralModel, StructuralState
from .solver import MinimumEnergySolver, StructuralSolution

__all__ = [
    "DofLayout",
    "StructuralState",
    "StructuralModel",
    "PotentialEnergy",
    "MinimumEnergySolver",
    "StructuralSolution",
]

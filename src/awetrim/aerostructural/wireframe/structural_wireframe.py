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
#
# Portions of this file are adapted from ASKITE
# (https://github.com/awegroup/ASKITE), licensed under the MIT License,
# Copyright (c) 2024 jellepoland (Jelle Poland, Patrick Roeleveld, TU Delft).
# See the NOTICE file at the repository root for the full MIT licence text.

"""The wireframe structural backend: Billow's line system, driven by the QSM loop.

This is the bridle-and-lines fidelity — tension-only cables and frictionless
pulleys, no bending — solved by minimising the total potential energy. It
replaces the particle system and its kinetic-damping relaxation that AWETrim
used until 2026-09-12, and the separate ``structural_nlp.py`` that had already
replaced that relaxation's *solve* while leaving it owning the *state*.

Both of those are now one thing. :class:`WireframeStructure` wraps a
:class:`billow.LineSystem`, which owns positions, rest lengths and stiffnesses
across the whole coupled run, and solves each call with the same minimum-energy
formulation the full Billow model uses. There is one cable force law in play,
in Billow, and the two fidelities call it — so the consistency that used to be
maintained by hand across three codes is now structural.

Why the interface looks like a particle system
----------------------------------------------
:class:`WireframeStructure` implements the ``WireframeSystem`` protocol in
``..protocols``: ``particles`` with ``update_pos``/``update_vel``,
``extract_rest_length``, ``update_rest_length``, ``springdampers`` with ``k``,
``f_int`` and ``x_v_current``. That is deliberate. The coupled drivers, the tape
actuation and the Aitken relaxation all speak that vocabulary already and are
correct, so keeping it means the backend swap touches the backend and nothing
else. ``kin_damp_sim`` is the one member NOT implemented — it is precisely the
kinetic relaxation being removed, and a caller reaching for it should fail
loudly rather than silently get something else.

``update_vel`` is a no-op. A minimum-energy solve has no velocity state; the
drivers zero the velocities after every relaxation step, which was always a
statement that the velocities carry no information between iterations.

The pulley rest-length convention
---------------------------------
This is the one place two otherwise-agreeing codes disagree, so it is handled
explicitly rather than inferred:

* the geometry reader (``structural_geometry_io``) stores the WHOLE rope's
  ``l0`` on both arm rows, and hands over the proportional per-arm split in
  ``pulley_line_to_other_node_pair_dict[idx][3]``;
* ``extract_rest_length`` — what the drivers and the actuation read and write —
  therefore carries the per-arm SPLIT, exactly as the particle system did;
* Billow's ``PulleyKernel`` takes the whole rope, so the two arms' split values
  are summed on the way in, and re-summed whenever either changes.

The physics is unaffected either way (the rope shares one stretch), but the
bookkeeping is not: reading the total as an arm length puts every rope into
artificial tension.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping, Sequence

import numpy as np

from billow import build_line_system
from billow.wireframe import LineSystem

Array = np.ndarray

logger = logging.getLogger(__name__)

#: Link types the geometry reader emits, normalised to lower case.
TENSION_ONLY = "noncompressive"
PULLEY = "pulley"


def _link_type_name(value: Any) -> str:
    """Normalised link-type string ('default' | 'noncompressive' | 'pulley')."""
    return str(value).lower().split(".")[-1]


class _NodeView:
    """One node, wearing the particle interface the drivers speak.

    A view, not a copy: it reads and writes the live Billow state, so a driver
    that relaxes positions and writes them back is updating the thing the next
    solve starts from.
    """

    __slots__ = ("_state", "_index")

    def __init__(self, state, index: int) -> None:
        self._state = state
        self._index = int(index)

    @property
    def x(self) -> Array:
        return self._state.positions[self._index]

    def update_pos(self, position: Array) -> None:
        self._state.positions[self._index] = np.asarray(position, dtype=float)

    def update_vel(self, velocity: Array) -> None:  # noqa: ARG002
        """No-op: a minimum-energy solve carries no velocity state."""


class _LinkView:
    """One line, wearing the spring-damper interface the drivers read ``k`` off."""

    __slots__ = ("_structure", "_index")

    def __init__(self, structure: "WireframeStructure", index: int) -> None:
        self._structure = structure
        self._index = int(index)

    @property
    def k(self) -> float:
        return float(self._structure.stiffnesses[self._index])

    @k.setter
    def k(self, value: float) -> None:
        stiffnesses = self._structure.stiffnesses.copy()
        stiffnesses[self._index] = float(value)
        self._structure.set_stiffnesses(stiffnesses)

    @property
    def linktype(self) -> str:
        return self._structure.link_types[self._index]


class WireframeStructure:
    """A Billow line system that owns the coupled run's structural state.

    Built by :func:`instantiate`. Everything addressed from outside is in the
    geometry reader's LINE ordering, which is not Billow's element ordering —
    see the module docstring on pulleys.
    """

    def __init__(
        self,
        system: LineSystem,
        *,
        link_types: Sequence[str],
        rest_lengths: Array,
        pulley_arm_pairs: Sequence[tuple[int, int]],
        masses: Array,
        fixed_node_indices: Sequence[int],
        solver_settings: Mapping[str, Any] | None = None,
    ) -> None:
        self.system = system
        self.link_types = [str(t) for t in link_types]
        #: Per-line rest lengths in the READER's convention: a pulley arm holds
        #: its own share, not the whole rope.
        self._rest_lengths = np.asarray(rest_lengths, dtype=float).copy()
        self.pulley_arm_pairs = [(int(a), int(b)) for a, b in pulley_arm_pairs]
        self.masses = np.asarray(masses, dtype=float).reshape(-1)
        self.fixed_node_indices = [int(i) for i in fixed_node_indices]
        self._last_internal_forces = np.zeros_like(self.system.positions)
        self.system.build_solver(**dict(solver_settings or {}))

    # -- the WireframeSystem protocol -------------------------------------------

    @property
    def particles(self) -> list[_NodeView]:
        return [_NodeView(self.system.state, i) for i in range(self.n_nodes)]

    @property
    def springdampers(self) -> list[_LinkView]:
        return [_LinkView(self, i) for i in range(self.n_lines)]

    @property
    def extract_rest_length(self) -> Array:
        """Per-line rest lengths [m], pulley arms split (the reader's convention)."""
        return self._rest_lengths.copy()

    def update_rest_length(self, element_index: int, delta_length: float) -> None:
        """Increment one line's rest length [m], as the particle system did.

        On a pulley arm this changes that arm's share and therefore the rope
        total, which is what Billow's element holds.
        """
        index = int(element_index)
        self._rest_lengths[index] += float(delta_length)
        self._push_rest_length(index)

    @property
    def f_int(self) -> Array:
        """Flattened internal force [N] from the last solve."""
        return self._last_internal_forces.reshape(-1)

    @property
    def x_v_current(self) -> tuple[Array, Array]:
        """Flattened positions and velocities; the velocities are always zero."""
        positions = self.system.positions.reshape(-1)
        return positions.copy(), np.zeros_like(positions)

    def kin_damp_sim(self, external_force: Array) -> None:  # noqa: ARG002
        raise NotImplementedError(
            "kinetic damping was removed with the particle system; this backend "
            "solves for equilibrium by energy minimisation. Call run_wireframe()."
        )

    # -- geometry and parameters -------------------------------------------

    @property
    def n_nodes(self) -> int:
        return int(self.system.positions.shape[0])

    @property
    def n_lines(self) -> int:
        return int(self.system.n_lines)

    @property
    def positions(self) -> Array:
        return self.system.positions

    @property
    def stiffnesses(self) -> Array:
        """Per-line stiffness [N/m] in the reader's ordering."""
        return self.system.stiffnesses

    def set_stiffnesses(self, values: Array) -> None:
        self.system.set_stiffnesses(np.asarray(values, dtype=float).reshape(-1))

    def _push_rest_length(self, index: int) -> None:
        """Write one line's rest length through to the Billow element.

        A pulley takes the sum of its two arms, because its kernel holds the
        whole rope.
        """
        for first, second in self.pulley_arm_pairs:
            if index in (first, second):
                total = float(self._rest_lengths[first] + self._rest_lengths[second])
                self.system.set_rest_length(first, total)
                return
        self.system.set_rest_length(index, float(self._rest_lengths[index]))

    def tensions(self) -> Array:
        """Per-line tension [N] at the current positions; a slack line reads 0."""
        return self.system.tensions()


def instantiate(
    config,
    struc_nodes,
    m_arr,
    kite_connectivity_arr,
    l0_arr,
    k_arr,
    c_arr,
    linktype_arr,
    pulley_line_to_other_node_pair_dict,
):
    """Build the wireframe structure from the geometry reader's arrays.

    Returns ``(structure, initial_conditions, params, struc_nodes_initial)``,
    the shape the drivers already expect. ``c_arr`` (damping) is accepted and
    ignored: a static minimum-energy solve has no damping, and the value only
    ever mattered to the kinetic relaxation.
    """
    if config.get("is_with_initial_point_velocity"):
        raise ValueError("initial point velocity has never been defined")

    settings = dict(config.get("structural_pss", {}) or {})
    connectivity = np.asarray(kite_connectivity_arr, dtype=int)
    rest_lengths = np.asarray(l0_arr, dtype=float).reshape(-1).copy()
    stiffness = np.asarray(k_arr, dtype=float).reshape(-1)
    link_types = [_link_type_name(t) for t in linktype_arr]
    fixed = [int(i) for i in settings.get("fixed_point_indices", [0])]

    # -- pulleys -----------------------------------------------------------
    # The reader appends the two arms of one rope back to back (ci-cj then
    # cj-ck), so consecutive pulley rows pair up. Checked, not assumed: a wrong
    # pairing builds a plausible-looking wrong rope.
    pulley_lines = sorted(int(i) for i in pulley_line_to_other_node_pair_dict)
    if len(pulley_lines) % 2:
        raise ValueError(
            f"pulley arms must pair up (got {len(pulley_lines)} entries)"
        )
    arm_pairs: list[tuple[int, int]] = []
    for first, second in zip(pulley_lines[0::2], pulley_lines[1::2]):
        if second != first + 1:
            raise ValueError(
                f"pulley arm indices are expected consecutive (got {first}, "
                f"{second}); the pair grouping would be wrong"
            )
        arm_pairs.append((first, second))

    # The reader stores the WHOLE rope's l0 on both arm rows and the
    # proportional per-arm split at index [3] of the pulley entry. The split is
    # what `extract_rest_length` must report, because that is what the particle
    # system reported and what the actuation is written against.
    for first, second in arm_pairs:
        rest_lengths[first] = float(pulley_line_to_other_node_pair_dict[str(first)][3])
        rest_lengths[second] = float(
            pulley_line_to_other_node_pair_dict[str(second)][3]
        )

    system = build_line_system(
        np.asarray(struc_nodes, dtype=float),
        connectivity,
        rest_lengths,
        stiffness,
        tension_only=[t == TENSION_ONLY for t in link_types],
        pulley_arm_pairs=arm_pairs,
        # Explicit rather than inferred: the two split arms sum to the rope.
        pulley_rest_lengths=[
            float(rest_lengths[first] + rest_lengths[second])
            for first, second in arm_pairs
        ],
        fixed_nodes=fixed,
    )

    structure = WireframeStructure(
        system,
        link_types=link_types,
        rest_lengths=rest_lengths,
        pulley_arm_pairs=arm_pairs,
        masses=np.asarray(m_arr, dtype=float),
        fixed_node_indices=fixed,
        solver_settings=_solver_settings(settings),
    )

    logger.info(
        "wireframe structure: %d nodes, %d lines (%d tension-only, %d pulley "
        "arms), fixed %s",
        structure.n_nodes,
        structure.n_lines,
        sum(t == TENSION_ONLY for t in link_types),
        2 * len(arm_pairs),
        fixed,
    )

    struc_nodes_initial = structure.positions.copy()
    return structure, None, settings, struc_nodes_initial


def _solver_settings(settings: Mapping[str, Any]) -> dict[str, Any]:
    """Map the ``structural_pss`` config block onto Billow solver settings.

    The key names are the ones the NLP inner solver already used, so an existing
    as_config keeps working unchanged.
    """
    return {
        "tolerance": float(settings.get("nlp_tolerance", 1e-8)),
        "max_iterations": int(settings.get("nlp_max_iterations", 1000)),
        # 1e-3 N/m, carried over from the NLP inner solver it replaces: enough
        # to pin a force-free fully slack node whose position would otherwise be
        # indeterminate, far below any coupling tolerance.
        "anchor_stiffness": float(settings.get("nlp_anchor_stiffness", 1e-3)),
    }


def run_wireframe(structure: WireframeStructure, f_ext, config_structural_pss=None):
    """Solve one equilibrium of the line system under ``f_ext``.

    Matches the call contract the coupled drivers use:
    ``(structure, is_converged, struc_nodes, f_int)`` with ``f_int`` flattened.
    The solved positions become the structure's state, so the next call warm
    starts from them.
    """
    forces = np.asarray(f_ext, dtype=float).reshape(-1, 3)
    solution = structure.system.solve(forces)

    structure._last_internal_forces = np.asarray(
        solution.internal_forces, dtype=float
    ).reshape(-1, 3)

    if not solution.converged:
        logger.warning(
            "wireframe solve did not reach force balance (residual %.4g N, "
            "IPOPT status %s); returning the best iterate",
            solution.residual_norm,
            solution.status,
        )

    return (
        structure,
        bool(solution.converged),
        structure.positions.copy(),
        structure.f_int,
    )


def get_stiffnesses(structure: WireframeStructure) -> Array:
    """Current line stiffnesses [N/m], in the reader's line order."""
    return structure.stiffnesses


def set_stiffnesses(structure: WireframeStructure, stiffnesses) -> None:
    """Write line stiffnesses [N/m] back into the structure."""
    structure.set_stiffnesses(stiffnesses)


def get_rest_lengths(structure: WireframeStructure) -> Array:
    """Current line rest lengths [m], pulley arms split (reader convention)."""
    return structure.extract_rest_length


def modulus_stiffness_ceiling(stiffnesses_initial, max_modulus_factor: float):
    """Per-element stiffness ceiling implied by a MODULUS ceiling [N/m].

    Element stiffness is ``k = E * A / l0``, so for a fixed geometry the ratio
    ``k / k_initial`` IS the ratio ``E / E_initial``. Capping every element at
    the same multiple of its own initial stiffness therefore caps every element
    at the same MODULUS -- which is the physical statement ("no line is stiffer
    than its fibre") -- while a single scalar ``max_stiffness`` in N/m caps a
    short thick line at a wholly different modulus than a long thin one.

    That distinction is not academic: under a scalar cap ``Br_main_1`` (0.61 m,
    2 mm) settled at an effective 160 GPa, well above the 109 GPa datasheet
    value of the fibre it is supposedly made of.

    Args:
        stiffnesses_initial: per-element ``k`` as built from the geometry [N/m].
        max_modulus_factor: the modulus ceiling as a multiple of the geometry's
            own modulus, e.g. 10.9 to allow 109 GPa from a 10 GPa base.

    Returns:
        Per-element ceiling [N/m].
    """
    return np.asarray(stiffnesses_initial, dtype=float).reshape(-1) * float(
        max_modulus_factor
    )


def adapt_stiffnesses(
    stiffnesses,
    elongations,
    *,
    element_indices=None,
    max_elongation: float = 0.01,
    factor: float = 1.5,
    max_stiffness=None,
):
    """Stiffen every eligible element elongating more than ``max_elongation``.

    The WING element stiffnesses are not known; following Poland & Schmehl they
    are chosen so that no element elongates by more than 1%. This enforces that
    inside the coupled solve, so a converged shape satisfies both the force
    residual and the elongation bound.

    ``max_stiffness`` is a per-element ARRAY (see modulus_stiffness_ceiling),
    not a scalar. Once an element sits at its ceiling it stops being updated,
    which is what lets the loop finish: the ceiling is a modulus the material
    genuinely has, so an element pinned there is reporting a real limit rather
    than being silently pushed past one.

    Args:
        stiffnesses: current per-element stiffness [N/m], in element order.
        elongations: per-element elongation [-], same order (pulley-aware --
            see convergence.element_elongations).
        element_indices: eligible elements (None = all).
        max_elongation: bound above which an element is stiffened [-].
        factor: multiplier applied to an offending element.
        max_stiffness: per-element ceiling [N/m], or None for uncapped.

    Returns:
        (updated stiffnesses, number stiffened, largest ELIGIBLE elongation,
        number already pinned at their ceiling).
    """
    updated = np.asarray(stiffnesses, dtype=float).reshape(-1).copy()
    values = np.asarray(elongations, dtype=float).reshape(-1)
    n_elements = min(len(updated), len(values))
    ceiling = (
        None
        if max_stiffness is None
        else np.asarray(max_stiffness, dtype=float).reshape(-1)
    )
    eligible = (
        range(n_elements)
        if element_indices is None
        else [int(i) for i in element_indices if 0 <= int(i) < n_elements]
    )

    n_updated = 0
    n_pinned = 0
    max_seen = float("nan")
    for index in eligible:
        elongation = float(values[index])
        if not np.isfinite(elongation):
            continue
        max_seen = elongation if np.isnan(max_seen) else max(max_seen, elongation)
        if elongation <= max_elongation:
            continue
        limit = float("inf") if ceiling is None else float(ceiling[index])
        if updated[index] >= limit:
            n_pinned += 1
            continue
        k_new = min(updated[index] * float(factor), limit)
        if k_new > updated[index]:
            updated[index] = k_new
            n_updated += 1

    return updated, n_updated, max_seen, n_pinned


__all__ = [
    "WireframeStructure",
    "instantiate",
    "run_wireframe",
    "get_stiffnesses",
    "set_stiffnesses",
    "get_rest_lengths",
    "modulus_stiffness_ceiling",
    "adapt_stiffnesses",
]

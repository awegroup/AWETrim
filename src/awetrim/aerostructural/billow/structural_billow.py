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

"""Billow structural backend for the aero-structural coupling.

Adapts the standalone minimum-energy library in :mod:`billow` to
the call contract the coupled drivers already use for ``kite_fem``
(``instantiate`` / ``run_billow`` / ``get_rest_lengths``), so the outer
fixed-point loop, the VSM trim and the load mapping are untouched.

The element mapping, read off the geometry the FEM reader already produces:

===========================  ===============================================
 reader element               Billow element
===========================  ===============================================
 ``inflatable_beam``          ``InflatableBeamKernel`` tube beams
 ``pulley`` (arm pairs)       ``PulleyKernel`` three-node ropes
 bridle ``noncompressive``    ``CableKernel`` tension-only cables
 canopy grid springs          **dropped** -- replaced by wrinkling membrane
                              triangles on the same structured grid
===========================  ===============================================

Dropping the canopy spring net is the point of this backend. The FEM path
models the fabric as a square net with diagonals, which is too soft by
``1 - nu`` in tension and carries shear only as a fourth-order effect of the
diagonal stretch; a relaxed (Pipkin) CST membrane carries the fabric's real
biaxial law and settles into a wrinkled state instead of mesh-scale crumple.
The bridle, the pulleys and the inflatable tubes are element-for-element the
same as the FEM model, so a Billow-vs-FEM difference is attributable to the
canopy and to the solver formulation, and to nothing else.

This module is the only place that knows about both packages:
:mod:`billow` stays free of schema, VSM and PSS knowledge.
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from billow import MinimumEnergySolver, StructuralModel, StructuralState
from billow.rotations import minimal_rotation, orthonormalize
from billow.symmetry import mirror_equalities, mirror_frames, mirror_partners
from billow.elements import (
    InflatableTubeLaw,
    build_cable_elements,
    build_inflatable_beam_elements,
    build_membrane_elements,
    build_pulley_elements,
    initial_frames_from_polyline,
)

logger = logging.getLogger(__name__)

Array = np.ndarray

#: Timoshenko shear correction for a thin-walled circular tube -- the same
#: ``8/9`` kite_fem's ``BeamElement`` uses, so the two shear stiffnesses are
#: directly comparable.
SHEAR_CORRECTION = 8.0 / 9.0

#: Element-set names, fixed so diagnostics and actuation can address them.
CABLES = "bridle"
PULLEYS = "pulleys"
TUBES = "tubes"
CANOPY = "canopy"

DEFAULTS: dict[str, Any] = {
    # -- canopy fabric --------------------------------------------------
    # E*t [N/m], the membrane stress resultant per unit strain, and the fabric
    # thickness that splits it. These are the REAL fabric, not a number chosen
    # to agree with another model: the V3 canopy is a polyester ripstop at the
    # 170 g/m2 the geometry YAML records, so the solid-equivalent thickness is
    # 0.170 / 1380 = 123 um, and at a crimp-derated in-plane modulus of 4 GPa
    # that is E*t ~ 4.9e5 N/m.
    #
    # The FEM canopy is a spring net at 5000 N/m, whose uniaxial membrane
    # equivalent is E*t = 5000 -- E = 20 MPa at any plausible thickness, i.e.
    # rubber. Matching it makes the two canopies comparable and both wrong; a
    # canopy that soft billows instead of carrying load and bends the struts
    # with it (measured: strut sagitta 4.5-6.7% of chord at 5e3, 0.6-4.1% at
    # 1e6, same load).
    #
    # The modulus is a MATERIAL ESTIMATE, not a measurement -- a tensile test
    # on the actual cloth would replace it, and the plausible range (2-12 GPa
    # fibre, more crimp derating) spans 2.5e5 to 1.5e6 N/m.
    "canopy_stiffness": 4.93e5,
    "canopy_thickness": 1.232e-4,
    "canopy_poisson_ratio": 0.3,
    "canopy_wrinkling": True,
    "canopy_slack_stiffness_ratio": 1e-4,
    # -- inflatable tubes -----------------------------------------------
    # ``tube_stiffness_factor`` scales the whole fitted moment-curvature curve,
    # initial slope included, leaving its shape alone (see
    # structural/elements/inflatable.py and the hanging-kite validation, which
    # implies a factor well below one).
    "tube_stiffness_factor": 1.0,
    "tube_axial_stiffness": None,  # [N]  None -> derived from the fitted EI_0
    "tube_shear_stiffness": None,  # [N]  None -> derived from the fitted GJ_0
    "junction_frame": "strut",
    # Mirror the tube frames of a mirror-symmetric kite from one half onto the
    # other (see symmetric_frames). Without it the frames transported from one
    # tip reach the other rolled by up to 60 degrees, the tube energy is not
    # mirror-symmetric, and an unsteered kite solves to an asymmetric shape
    # whatever the load. False reproduces results from before 2026-09-11.
    "mirror_frames": True,
    # Canopy triangulation; see canopy_mesh. "diagonal" is the historical,
    # mirror-ASYMMETRIC two-triangle split and reproduces earlier results.
    "canopy_pattern": "cross",
    # Canopy mesh refinement: subdivide every quad k x k before triangulating.
    # 1 keeps the structural grid. The coarse lattice is preserved exactly, so
    # the leading and trailing edges, the strut stations and every bridle
    # attachment keep their nodes and the tubes and cables are untouched; only
    # the fabric is refined. See refine_grid for what the added nodes do NOT get.
    "canopy_refinement": 1,
    # -- bridle relaxation ----------------------------------------------
    # The kite YAMLs store measured rest lengths against measured node
    # positions and the two do not agree; see relax_bridles.
    "relax_bridles": True,
    "relax_pull_force": -100.0,   # [N] downward pull on the KCU while settling
    "relax_settle_force": -1.0,   # [N] token load for the second pass
    "relax_move_limit": 0.25,     # [m] trust region for the relaxation solve
    # -- solver ----------------------------------------------------------
    "tolerance": 1e-8,
    # Inner acceptance for one run_billow call, as
    # max(force_tolerance, relative_force_tolerance * total aerodynamic load).
    # Relative to the TOTAL load, not the largest nodal one: the total is a
    # property of the flight state, the per-node maximum a property of the
    # discretisation, so keying off the latter would tighten the demand every
    # time the mesh is refined -- the opposite of mesh independence.
    #
    # Keep this proportionate to the OUTER coupled gate. Asking the inner solve
    # for 1e-3 N while the loop accepts 0.5 N is 500x tighter than anything
    # downstream can use: a coarse mesh happens to manage it, a 5838-DOF largely
    # slack membrane does not, and then every solve burns its whole round budget
    # and returns its best iterate anyway (measured at refinement 3: 250 s a
    # solve, still at 50 N). At the LEI-V3's ~1.3 kN this gives 1.3e-2 N, still
    # nearly forty times tighter than the gate.
    "force_tolerance": 1e-6,
    "relative_force_tolerance": 1e-5,
    "max_iterations": 1000,
    # ``move_limit`` makes one solve a trust-region step; ``max_rounds`` is
    # how many of those ``run_billow`` will take to reach force balance. With
    # ``move_limit: None`` the solve is unbounded and one round is all there is.
    "move_limit": 0.25,
    "max_rounds": 12,
    "anchor_stiffness": 1e-6,
    "max_frame_updates": 3,
    "print_info": False,
}


def resolve_config(config_structural_billow: Mapping[str, Any] | None) -> dict[str, Any]:
    """Fill a ``structural_billow`` config block with the module defaults."""
    supplied = dict(config_structural_billow or {})
    unknown = set(supplied) - set(DEFAULTS)
    if unknown:
        raise ValueError(
            f"unknown structural_billow keys: {sorted(unknown)}; "
            f"known keys are {sorted(DEFAULTS)}"
        )
    resolved = dict(DEFAULTS)
    resolved.update(supplied)
    return resolved


@dataclass
class BillowStructure:
    """Live structural state of one Billow model, plus its compiled solver.

    ``model`` carries the current element *parameters* (actuated rest lengths,
    a ramped stiffness); ``solver`` is compiled against the topology once and
    reused. ``state`` holds the current positions and reference frames and is
    what the next solve is seeded from.
    """

    model: StructuralModel
    solver: MinimumEnergySolver
    state: StructuralState
    #: The structural canopy grid -- the sections the reader produced.
    grid: Array
    #: The grid the MEMBRANE is meshed on. Equal to ``grid`` unless
    #: ``canopy_refinement`` subdivided it. Load the aerodynamics onto THIS one:
    #: mapping onto the coarse grid leaves every refined node unloaded, which is
    #: not merely a resolution question -- those nodes are also massless, so a
    #: large share of the degrees of freedom would have nothing determining them
    #: and the structural solve becomes badly conditioned (measured at
    #: refinement 3: the inner solve stalled at 9-25 N and cost 275 s against
    #: 2 s coarse, while the aerodynamic side was unchanged at 10 s).
    fine_grid: Array
    #: Row of the cable set for each reader element, ``-1`` where the element is
    #: not a Billow cable (a canopy spring, a pulley arm or a tube).
    cable_row: Array
    #: Row of the pulley set for each reader element, ``-1`` otherwise. Both
    #: arms of one rope point at the same row.
    pulley_row: Array
    #: Row of the tube set for each reader element, ``-1`` otherwise.
    tube_row: Array
    tube_laws: list = field(default_factory=list)
    #: Nodal masses, extended to match ``model.nodes`` when the canopy pattern
    #: adds quad-centre nodes. Callers must use THIS, not the reader's m_arr,
    #: or the force and position arrays will disagree in length.
    masses: Array = field(default_factory=lambda: np.empty(0))
    fixed_node_indices: tuple[int, ...] = (0,)
    last_solution: Any = None

    @property
    def struc_nodes(self) -> Array:
        """Current nodal positions ``(n_nodes, 3)``."""
        return self.state.positions


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------


def canopy_grid(canopy_sections, strut_sections) -> Array:
    """The structured spanwise x chordwise grid of wing nodes.

    The reader emits canopy sections and strut sections separately, but together
    they tile the whole wing: every section runs leading edge to trailing edge
    with the same node count, and sorting by leading-edge index puts them in
    spanwise order.
    """
    sections = list(canopy_sections) + list(strut_sections)
    widths = {len(section) for section in sections}
    if len(widths) != 1:
        raise ValueError(
            f"wing sections have mixed node counts {sorted(widths)}; the canopy "
            "is not a structured grid and cannot be meshed with membranes"
        )
    sections.sort(key=lambda section: section[0])
    return np.asarray([list(section) for section in sections], dtype=int)


def grid_edges(grid: Array) -> set[frozenset[int]]:
    """Every chordwise, spanwise and diagonal edge of the structured grid.

    These are exactly the springs the FEM reader lays over the canopy, so the
    set doubles as the classifier that tells a canopy spring from a bridle line.
    """
    edges: set[frozenset[int]] = set()
    rows, columns = grid.shape
    for i in range(rows):
        for j in range(columns - 1):
            edges.add(frozenset((int(grid[i, j]), int(grid[i, j + 1]))))
    for i in range(rows - 1):
        for j in range(columns):
            edges.add(frozenset((int(grid[i, j]), int(grid[i + 1, j]))))
        for j in range(columns - 1):
            edges.add(frozenset((int(grid[i, j]), int(grid[i + 1, j + 1]))))
            edges.add(frozenset((int(grid[i, j + 1]), int(grid[i + 1, j]))))
    return edges


def canopy_triangles(grid: Array) -> Array:
    """Split every grid quad into two triangles along a consistent diagonal.

    Kept for the ``diagonal`` pattern and for callers that only want the
    topology; :func:`canopy_mesh` is the general entry point.
    """
    rows, columns = grid.shape
    triangles = []
    for i in range(rows - 1):
        for j in range(columns - 1):
            a, b = int(grid[i, j]), int(grid[i, j + 1])
            c, d = int(grid[i + 1, j + 1]), int(grid[i + 1, j])
            triangles.append([a, b, c])
            triangles.append([a, c, d])
    return np.asarray(triangles, dtype=int)


def refine_grid(grid: Array, nodes: Array, factor: int):
    """Subdivide the canopy grid ``factor`` x ``factor``. Returns ``(grid, extra)``.

    Bilinear within each quad, which keeps the refined lattice exactly
    mirror-symmetric whenever the coarse one is: the interpolation of a
    symmetric set of corners is symmetric. The original nodes are reused in
    place, so the leading and trailing edges, the strut stations and the bridle
    attachments all keep their identity and the tube, cable and pulley elements
    need no remapping at all.

    **The added nodes carry no mass and receive no direct aerodynamic load.**
    The reader weighed the canopy onto the coarse nodes, and the
    aero-to-structure mapping targets the coarse sections, so a refined interior
    node is held by the membrane alone. That is tolerable for shape, and it is
    the same limitation the ``cross`` pattern has, but it means a refined canopy
    is not simply a better-resolved version of the same load case: the pressure
    is still applied at the coarse stations.
    """
    factor = int(factor)
    if factor < 1:
        raise ValueError(f"canopy_refinement must be >= 1, got {factor}")
    if factor == 1:
        return np.asarray(grid, dtype=int), np.empty((0, 3))

    grid = np.asarray(grid, dtype=int)
    nodes = np.asarray(nodes, dtype=float)
    rows, columns = grid.shape
    fine_rows, fine_columns = (rows - 1) * factor + 1, (columns - 1) * factor + 1

    fine = np.full((fine_rows, fine_columns), -1, dtype=int)
    fine[::factor, ::factor] = grid            # the coarse lattice, kept in place

    extra: list[Array] = []
    next_index = len(nodes)
    for i in range(fine_rows):
        for j in range(fine_columns):
            if fine[i, j] >= 0:
                continue
            # Bilinear blend of the four coarse corners bounding this station.
            i0, j0 = min(i // factor, rows - 2), min(j // factor, columns - 2)
            u, v = (i - i0 * factor) / factor, (j - j0 * factor) / factor
            corner = [nodes[grid[i0, j0]], nodes[grid[i0, j0 + 1]],
                      nodes[grid[i0 + 1, j0 + 1]], nodes[grid[i0 + 1, j0]]]
            extra.append(
                (1 - u) * ((1 - v) * corner[0] + v * corner[1])
                + u * ((1 - v) * corner[3] + v * corner[2])
            )
            fine[i, j] = next_index
            next_index += 1
    return fine, np.asarray(extra, dtype=float).reshape(-1, 3)


def canopy_mesh(grid: Array, nodes: Array, pattern: str = "cross"):
    """Triangulate the canopy grid. Returns ``(triangles, extra_nodes, scale)``.

    A constant-strain triangle mesh built by splitting every quad along the
    *same* diagonal is directionally biased, and on this canopy the quads are a
    sizeable fraction of the wing rather than a fine discretisation. Worse, the
    bias is not mirror-symmetric: under ``y -> -y`` the diagonal ``(i,j)-(i+1,j+1)``
    maps to the other diagonal, so a symmetric kite under symmetric actuation
    does not come out symmetric. Measured on the LEI-V3, the solve generated up
    to 21 mm of left-right mismatch from an exactly mirror-symmetric geometry,
    concentrated on the pure-canopy sections and suppressed on the
    strut-stiffened ones -- the signature of exactly this effect.

    ``diagonal``
        One diagonal per quad, two triangles. The historical mesh; biased.
    ``union``
        Both 2-triangle triangulations SUPERPOSED, each at half the stress
        resultant. This is an overlay, not a subdivision: the quad is covered
        twice, by ``(abc, acd)`` on one diagonal and ``(abd, bcd)`` on the
        other, and the two diagonals never meet -- they belong to two
        independent sheets, so no centre node is needed or implied.

        Two consequences follow, and both are why this is a control rather
        than a recommendation. A quad need not be planar, and on a non-planar
        quad the two triangulations describe two DIFFERENT surfaces, so the
        patch is modelled as two interpenetrating half-stiffness sheets rather
        than one surface. And it still cannot dome: every triangle has all
        three corners on the quad corners, so there is no interior freedom,
        and superposing two fold-only surfaces gives a fold-only patch.

        What it is good for is exactly one thing: it is mirror-symmetric (the
        reflection swaps the two triangulations) at zero added DOF, so it
        isolates the diagonal bias from the DOF count in an A/B.
    ``cross`` (default)
        A node at each quad centre joined to the four corners. Exactly
        mirror-symmetric, and the centre node gives the patch the one freedom a
        billowing sail most needs -- a two-triangle quad can only FOLD along its
        diagonal, it cannot bulge. Costs one node per quad.

    ``scale`` divides the stress resultant ``E*t`` between the overlapping
    sheets, so the ``union`` pattern's doubled element count does not double
    the canopy stiffness. It scales the RESULTANT and not the thickness: the
    wrinkling discriminant is a stress state, and halving ``t`` instead would
    move the slack/wrinkled/taut branch as well as the stiffness.
    """
    rows, columns = grid.shape
    nodes = np.asarray(nodes, dtype=float)
    triangles: list[list[int]] = []
    extra: list[Array] = []

    if pattern == "diagonal":
        return canopy_triangles(grid), np.empty((0, 3)), 1.0

    if pattern not in ("union", "cross"):
        raise ValueError(
            f"canopy_pattern must be 'diagonal', 'union' or 'cross', got {pattern!r}"
        )

    next_index = len(nodes)
    for i in range(rows - 1):
        for j in range(columns - 1):
            a, b = int(grid[i, j]), int(grid[i, j + 1])
            c, d = int(grid[i + 1, j + 1]), int(grid[i + 1, j])
            if pattern == "union":
                triangles += [[a, b, c], [a, c, d], [a, b, d], [b, c, d]]
            else:
                centre = next_index
                next_index += 1
                extra.append(0.25 * (nodes[a] + nodes[b] + nodes[c] + nodes[d]))
                triangles += [[a, b, centre], [b, c, centre],
                              [c, d, centre], [d, a, centre]]

    return (
        np.asarray(triangles, dtype=int),
        np.asarray(extra, dtype=float).reshape(-1, 3),
        0.5 if pattern == "union" else 1.0,
    )


def leading_edge_chain(struc_geometry: Mapping[str, Any]) -> list[int]:
    """Node path along the leading-edge tube, walked from its element list."""
    successor: dict[int, int] = {}
    predecessors: set[int] = set()
    for _name, ci, cj, _diameter in struc_geometry["leading_edge_tubes"]["data"]:
        successor[int(ci)] = int(cj)
        predecessors.add(int(cj))
    starts = [node for node in successor if node not in predecessors]
    if len(starts) != 1:
        raise ValueError(
            f"leading_edge_tubes must form one open chain; found {len(starts)} "
            f"start nodes {sorted(starts)}"
        )
    chain = [starts[0]]
    while chain[-1] in successor:
        chain.append(successor[chain[-1]])
    if len(chain) != len(successor) + 1:
        raise ValueError("leading_edge_tubes chain is broken or branches")
    return chain


def beam_chains(struc_geometry, strut_sections) -> list[list[int]]:
    """The leading-edge chain plus one chain per strut, as node paths.

    Frames live on nodes, so they have to be carried along something smooth.
    Each chain is a polyline; struts and the leading edge meet at shared nodes,
    and whichever chain is laid down last owns the frame there.
    """
    return [leading_edge_chain(struc_geometry)] + [
        list(section) for section in strut_sections
    ]


def chain_tangents(points: Array) -> Array:
    """Unit tangents along an open polyline, averaged at interior nodes.

    The same rule :func:`initial_frames_from_polyline` uses, lifted out so the
    tangent field and the roll can be chosen independently.
    """
    points = np.asarray(points, dtype=float)
    segments = np.diff(points, axis=0)
    tangents = np.empty_like(points)
    tangents[0] = segments[0]
    tangents[-1] = segments[-1]
    if len(points) > 2:
        tangents[1:-1] = segments[:-1] + segments[1:]
    return tangents / np.linalg.norm(tangents, axis=1, keepdims=True)


def build_frames(
    struc_nodes: Array,
    beam_connectivity: Array,
    struc_geometry,
    strut_sections,
    junction_frame: str = "strut",
) -> Array:
    """Nodal material frames: ``d1`` along the owning member, roll transported.

    The directors do two jobs -- they orient the diagonal section stiffness and
    they fix the reference strains. An inflated tube is isotropic in roll
    (``ga_2 == ga_3``, one bending law), so only ``d1`` is physically
    determined and the roll is free. It is spent here on keeping every
    element's relative rotation as small as possible, because the reference
    curvature is ``omega_0 = psi / L0`` with ``psi`` a Rodrigues vector: it
    grows like ``tan(theta/2)`` and is singular at ``theta = pi``. Seeding each
    chain's roll independently piles an arbitrary roll offset on top of the
    physical joint angle and took the worst element to 173 degrees on the
    LEI-V3; minimal-rotation transport over the beam network leaves the joint
    angle alone and nothing else.

    ``junction_frame`` decides who owns the node where a strut meets the
    leading edge. One nodal frame cannot be tangent to two near-perpendicular
    members, so the ~90 degree joint rotation has to land on *somebody*; it
    should land on the long members. The strut's first element is 17-97 mm
    while the leading-edge runs are 300-750 mm, so ``"strut"`` is the default
    and puts ``omega_0`` an order of magnitude lower.
    """
    if junction_frame not in ("leading_edge", "strut"):
        raise ValueError("junction_frame must be 'leading_edge' or 'strut'")

    # -- per-node tangents, the owning member having the last word ---------
    chains = beam_chains(struc_geometry, strut_sections)
    ordered = chains if junction_frame == "strut" else chains[1:] + chains[:1]
    tangents = np.zeros((len(struc_nodes), 3))
    for chain in ordered:
        tangents[chain] = chain_tangents(struc_nodes[chain])

    # -- roll, transported over the beam network --------------------------
    neighbours: dict[int, list[int]] = {}
    for node_a, node_b in np.asarray(beam_connectivity, dtype=int):
        neighbours.setdefault(int(node_a), []).append(int(node_b))
        neighbours.setdefault(int(node_b), []).append(int(node_a))

    frames = np.tile(np.eye(3), (len(struc_nodes), 1, 1))
    visited = set()
    for root in sorted(neighbours):
        if root in visited:
            continue
        # Seed one arbitrary roll per connected component; every other node in
        # it inherits by minimal rotation, so only the component's own
        # orientation is a choice and no element pays for it.
        frames[root] = initial_frames_from_polyline(
            np.array([struc_nodes[root], struc_nodes[root] + tangents[root]])
        )[0]
        visited.add(root)
        queue = [root]
        while queue:
            node = queue.pop()
            for other in neighbours[node]:
                if other in visited:
                    continue
                frames[other] = orthonormalize(
                    minimal_rotation(tangents[node], tangents[other]) @ frames[node]
                )
                visited.add(other)
                queue.append(other)
    return frames


def symmetric_frames(struc_nodes: Array, frames: Array, beam_connectivity: Array) -> Array:
    """Mirror-consistent tube frames for a mirror-symmetric kite.

    :func:`build_frames` transports the roll over the beam network from ONE
    root, and that transport is not mirror-equivariant on an LEI kite: the
    leading edge crosses the symmetry plane, so the reflection reverses its
    tangent, while the struts' tangents keep their sense -- and every hop
    between the two at a strut junction turns the roll the wrong way on one
    side. On the LEI-V3 the far half arrived rolled by up to 60 degrees against
    the mirror image of the near half (96 of 98 frames). Positions stayed
    symmetric to 4e-15 m, so every position check passed, but the geometrically
    exact tube energy depends on the frames: the internal load at a symmetric
    configuration was off by 40% of the largest moment, a symmetric load had no
    symmetric equilibrium, and the unsteered kite solved to the SAME asymmetric
    shape from any start.

    Keeping the near half (``y > 0``) and mirroring it, ``R_p = M R diag(-1, 1,
    1)``, makes every tube element exactly mirror-invariant; the near half's
    frames, and so its reference curvatures, are untouched. An asymmetric
    geometry is left as built.
    """
    try:
        partner = mirror_partners(struc_nodes)
    except ValueError as error:
        logger.info("tube frames not mirrored, the geometry is not symmetric: %s", error)
        return frames
    beam_connectivity = np.asarray(beam_connectivity, dtype=int)
    beam_nodes = np.unique(beam_connectivity)
    mirrored = mirror_frames(frames, struc_nodes, partner, beam_nodes)

    def angle(first, second):
        cosine = (np.einsum("nij,nij->n", first, second) - 1.0) / 2.0
        return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

    changed = angle(frames[beam_nodes], mirrored[beam_nodes])
    # The seam: elements crossing the plane now join a kept frame to a mirrored
    # one. Across a leading edge that is the ordinary discrete curvature; if a
    # plane-crossing element joined two strut-owned nodes, the convention would
    # reverse one d1 and the element would sit near the Cayley singularity.
    side = np.sign(struc_nodes[beam_connectivity, 1])
    seam = beam_connectivity[side[:, 0] * side[:, 1] < 0]
    seam_angle = float(angle(mirrored[seam[:, 0]], mirrored[seam[:, 1]]).max(initial=0.0))
    logger.info(
        "tube frames mirrored onto the y < 0 half: largest change %.2f deg over %d "
        "beam nodes; largest relative rotation across the plane %.2f deg",
        float(changed.max(initial=0.0)), len(beam_nodes), seam_angle,
    )
    if seam_angle > 90.0:
        logger.warning(
            "an element crossing the mirror plane joins frames %.1f deg apart; its "
            "reference curvature is near the Cayley singularity", seam_angle,
        )
    return mirrored


# --------------------------------------------------------------------------
# Model assembly
# --------------------------------------------------------------------------


def _pulley_triplets(
    kite_connectivity_arr: Array, pulley_line_indices: Sequence[int]
) -> tuple[list[tuple[int, int, int]], list[tuple[int, int]]]:
    """Group the reader's pulley arms into ``(i, pulley, k)`` ropes.

    The reader appends the two arms of one rope back to back (``ci-cj`` then
    ``cj-ck``), so consecutive entries pair up -- the same grouping
    ``pss/structural_wireframe.py`` relies on. It is checked here rather than assumed.
    """
    indices = [int(i) for i in (pulley_line_indices or [])]
    if len(indices) % 2:
        raise ValueError(f"pulley arms must pair up (got {len(indices)} entries)")

    triplets: list[tuple[int, int, int]] = []
    pairs: list[tuple[int, int]] = []
    for first, second in zip(indices[0::2], indices[1::2]):
        if second != first + 1:
            raise ValueError(
                f"pulley arm indices are expected consecutive (got {first}, "
                f"{second}); the pair grouping would be wrong"
            )
        ci, cj = (int(n) for n in kite_connectivity_arr[first])
        cj_other, ck = (int(n) for n in kite_connectivity_arr[second])
        if cj_other != cj:
            raise ValueError(
                f"pulley arms {first}/{second} do not share a node "
                f"({ci}-{cj} and {cj_other}-{ck})"
            )
        triplets.append((ci, cj, ck))
        pairs.append((first, second))
    return triplets, pairs


def _tube_stiffnesses(laws, diameters: Array, settings) -> tuple[Array, Array]:
    """Axial ``EA`` and shear ``kappa G A`` for the tube beams [N].

    The ASKITE fits cover bending and torsion only, so these two have to come
    from somewhere else. Unless the config overrides them they are derived from
    the *same* fits, reading the initial slopes as a modulus over the tube's
    thin-wall section: ``E = EI_0 / I`` and ``G = GJ_0 / J``. That keeps one
    source for the tube properties, and neither stiffness is sensitive here --
    an inflated tube is far stiffer in extension and shear than in bending, so
    both are effectively rigid at kite loads either way.
    """
    radius = 0.5 * np.asarray(diameters, dtype=float)
    area = np.pi * radius**2
    second_moment = np.pi * radius**4 / 4.0
    polar_moment = 2.0 * second_moment

    bending = np.array([law.bending_stiffness for law in laws])
    torsion = np.array([law.torsion_c1 * law.torsion_c2 for law in laws])

    axial = settings["tube_axial_stiffness"]
    shear = settings["tube_shear_stiffness"]
    axial = (
        bending / second_moment * area
        if axial is None
        else np.full(len(laws), float(axial))
    )
    shear = (
        SHEAR_CORRECTION * torsion / polar_moment * area
        if shear is None
        else np.full(len(laws), float(shear))
    )
    return axial, shear


def scale_tube_laws(laws, factor: float):
    """Multiply the whole moment-curvature curve, initial slope included.

    Scaling ``moment_max``, ``bending_stiffness`` and ``torsion_c1`` together
    leaves ``curvature_scale = M_max / EI_0`` unchanged, so the curve keeps its
    shape and only its magnitude moves -- one honest stiffness knob rather than
    a refit of the law.
    """
    if factor == 1.0:
        return list(laws)
    return [
        dataclasses.replace(
            law,
            moment_max=law.moment_max * factor,
            bending_stiffness=law.bending_stiffness * factor,
            torsion_c1=law.torsion_c1 * factor,
        )
        for law in laws
    ]


def relax_bridles(
    struc_nodes: Array,
    cables,
    pulleys,
    grid: Array,
    *,
    kcu_node: int = 0,
    pull_force: float = -100.0,
    settle_force: float = -1.0,
    move_limit: float = 0.25,
    max_rounds: int = 60,
    residual_target: float = 1e-4,
) -> tuple[Array, bool]:
    """Settle the bridle onto the built wing before the coupled model is built.

    The kite YAMLs store *measured* bridle rest lengths against *measured* node
    positions, and the two are not consistent: on the LEI-V3 FEM geometry
    ``Br_main_1`` is 11.5% long at its stored node positions, which at the real
    ``EA/l0`` is 88 kN in one line. Starting a coupled run there is starting it
    from an explosion. The FEM path hides this by calling
    ``structural_kite_fem.relaxbridles`` before the loop; this is the same step
    in Billow's own formulation.

    The wing is held rigid (every grid node fixed) so only the bridle moves and
    the canopy and tube reference configurations are untouched. The KCU is
    pulled down so the bridle settles in tension rather than folding, then
    settled again at a token load, then the whole structure is translated so
    the KCU returns to where the YAML puts it -- a rigid translation, which no
    reference strain can see.

    Returns the relaxed nodes and whether the settle solve reached equilibrium.
    """
    struc_nodes = np.asarray(struc_nodes, dtype=float)
    fixed = set(int(node) for node in grid.ravel())
    touched = {
        int(node)
        for element_set in (cables, pulleys)
        for node in element_set.connectivity.ravel()
    }
    # Anything the bridle does not touch has no load path here; pinning it
    # removes a null mode at no physical cost.
    fixed |= set(range(len(struc_nodes))) - touched
    fixed.discard(int(kcu_node))

    model = StructuralModel(
        struc_nodes, [cables, pulleys], fixed_translation_nodes=sorted(fixed)
    )
    solver = MinimumEnergySolver(
        model, tolerance=1e-8, max_iterations=2000, move_limit=move_limit
    )

    state, converged = None, False
    for load in (float(pull_force), float(settle_force)):
        forces = np.zeros_like(struc_nodes)
        forces[int(kcu_node), 2] = load
        for _ in range(max_rounds):
            solution = solver.solve(forces, state=state)
            state = solution.state
            converged = solution.residual_norm <= residual_target
            if converged:
                break

    relaxed = state.positions - state.positions[int(kcu_node)] + struc_nodes[int(kcu_node)]
    logger.info(
        "bridle relaxation: %s, %d bridle nodes moved, max displacement %.4f m",
        "settled" if converged else "NOT settled",
        len(touched - fixed),
        float(np.abs(relaxed - struc_nodes).max()),
    )
    return relaxed, converged


def build_solver(model: StructuralModel, settings, equalities=None) -> MinimumEnergySolver:
    """The minimum-energy solver for ``model`` with the ``structural_billow`` numerics.

    One place for the settings, so a constrained solve (``equalities``, e.g.
    the mirror symmetry of :func:`symmetric_equalities`) runs with exactly the
    numerics of the unconstrained one it is compared against.
    """
    return MinimumEnergySolver(
        model,
        tolerance=float(settings["tolerance"]),
        force_tolerance=float(settings["force_tolerance"]),
        relative_force_tolerance=float(settings["relative_force_tolerance"]),
        max_iterations=int(settings["max_iterations"]),
        move_limit=settings["move_limit"],
        anchor_stiffness=float(settings["anchor_stiffness"]),
        max_frame_updates=int(settings["max_frame_updates"]),
        equalities=equalities,
    )


def symmetric_equalities(structure: "BillowStructure"):
    """Mirror-symmetry constraints over EVERY node of the built model.

    Pairs are taken from the reference configuration (``model.nodes``), which
    is exactly symmetric for an unactuated kite; pairing fails loudly if it is
    not. Covering all nodes matters: the bridle knots and the quad-centre nodes
    are free DOF too, and leaving them out constrains only part of the shape.
    """
    partner = mirror_partners(structure.model.nodes)
    return mirror_equalities(
        structure.model.layout,
        partner,
        pinned_nodes=structure.model.fixed_translation_nodes,
    ), partner


def _bridle_elements(
    connectivity, rest_lengths, stiffness, link_types, pulley_line_indices,
    canopy_sections, strut_sections,
):
    """Split the reader's elements and build the cable and pulley sets.

    Returns ``(grid, is_beam, is_canopy, is_cable, cable_elements, triplets,
    arm_pairs, cables, pulleys)``. Shared by :func:`instantiate` and
    :func:`relax_bridle_nodes`, so the relaxation another structural model
    starts from is the one Billow starts from.
    """
    n_elements = len(connectivity)
    grid = canopy_grid(canopy_sections, strut_sections)
    edges = grid_edges(grid)

    is_beam = link_types == "inflatable_beam"
    triplets, arm_pairs = _pulley_triplets(connectivity, pulley_line_indices)
    is_pulley_arm = np.zeros(n_elements, dtype=bool)
    for first, second in arm_pairs:
        is_pulley_arm[[first, second]] = True

    pairs = [frozenset((int(a), int(b))) for a, b in connectivity]
    is_canopy = np.array([pair in edges for pair in pairs]) & ~is_beam & ~is_pulley_arm
    is_cable = ~(is_beam | is_pulley_arm | is_canopy)

    # A bridle line that happened to join two grid nodes would be silently
    # swallowed by the membrane, so check the split is unambiguous rather than
    # trusting it.
    on_grid = {int(node) for node in grid.ravel()}
    swallowed = [
        index
        for index, pair in enumerate(pairs)
        if is_cable[index] and set(pair) <= on_grid
    ]
    if swallowed:
        raise RuntimeError(
            f"{len(swallowed)} non-canopy elements join two canopy grid nodes "
            f"without being grid edges (first: element {swallowed[0]}, nodes "
            f"{sorted(pairs[swallowed[0]])}); the canopy split is ambiguous"
        )

    cable_elements = np.flatnonzero(is_cable)
    cables = build_cable_elements(
        connectivity[cable_elements],
        rest_lengths[cable_elements],
        stiffness[cable_elements],
        name=CABLES,
    )
    # Billow's PulleyKernel takes the rest length of the WHOLE rope, and the
    # reader stores that total on each arm (as kite_fem does, and unlike the PSS
    # reader, which splits it across the two arms). Read it off the first arm.
    first_arms = [first for first, _ in arm_pairs]
    pulleys = build_pulley_elements(
        np.asarray(triplets, dtype=int),
        rest_lengths[first_arms],
        stiffness[first_arms],
        name=PULLEYS,
    )
    return (
        grid, is_beam, is_canopy, is_cable, cable_elements, triplets, arm_pairs,
        cables, pulleys,
    )


def relax_bridle_nodes(
    config,
    struc_geometry,
    struc_nodes,
    kite_connectivity_arr,
    l0_arr,
    k_arr,
    linktype_arr,
    pulley_line_indices,
    canopy_sections,
    strut_sections,
) -> Array:
    """The node positions :func:`instantiate` starts the Billow model from.

    The same bridle relaxation, without building the tubes, the canopy or the
    solver: for a structural model reduced from the same geometry (the
    wireframe of ``wireframe.reduce_fem``), so both start from one shape.
    Returns ``struc_nodes`` unchanged when ``relax_bridles`` is off.
    """
    settings = resolve_config(config.get("structural_billow"))
    struc_nodes = np.asarray(struc_nodes, dtype=float)
    if not settings["relax_bridles"]:
        return struc_nodes.copy()
    fixed_nodes = tuple(int(i) for i in struc_geometry.get("fixed_point_indices", [0]))
    *_, cables, pulleys = _bridle_elements(
        np.asarray(kite_connectivity_arr, dtype=int),
        np.asarray(l0_arr, dtype=float),
        np.asarray(k_arr, dtype=float),
        np.asarray([str(t).lower() for t in linktype_arr]),
        pulley_line_indices,
        canopy_sections,
        strut_sections,
    )
    grid = canopy_grid(canopy_sections, strut_sections)
    relaxed, settled = relax_bridles(
        struc_nodes,
        cables,
        pulleys,
        grid,
        kcu_node=int(fixed_nodes[0]),
        pull_force=float(settings["relax_pull_force"]),
        settle_force=float(settings["relax_settle_force"]),
        move_limit=float(settings["relax_move_limit"]),
    )
    if not settled:
        logger.warning("the bridle did not settle onto the held wing")
    return relaxed


def tube_axial_stiffness(diameters, pressure, config=None) -> Array:
    """Axial stiffness ``EA`` [N] Billow gives inflatable tubes of ``diameters``.

    The same law and the same derivation :func:`instantiate` uses for the
    beams (fit at ``pressure``, scaled by ``tube_stiffness_factor``, ``EA``
    from the fitted initial slope unless ``tube_axial_stiffness`` overrides
    it), for a model that keeps only the tubes' axial stiffness.
    """
    settings = resolve_config((config or {}).get("structural_billow"))
    diameters = np.atleast_1d(np.asarray(diameters, dtype=float))
    laws = scale_tube_laws(
        [InflatableTubeLaw.from_fit(float(d), float(pressure)) for d in diameters],
        float(settings["tube_stiffness_factor"]),
    )
    axial, _shear = _tube_stiffnesses(laws, diameters, settings)
    return np.asarray(axial, dtype=float)


def instantiate(
    config,
    struc_geometry,
    struc_nodes,
    kite_connectivity_arr,
    l0_arr,
    k_arr,
    c_arr,
    m_arr,
    linktype_arr,
    pulley_line_indices,
    canopy_sections,
    strut_sections,
) -> BillowStructure:
    """Build the Billow model and compile its solver.

    Takes the arrays ``fem.read_struc_geometry_yaml.main`` already returns, so
    one geometry reader serves both structural backends. For the inflatable
    beams that reader stores the tube diameter in ``k_arr`` and the inflation
    pressure in ``c_arr``, which is how ``kite_fem`` receives them too.
    """
    settings = resolve_config(config.get("structural_billow"))

    struc_nodes = np.asarray(struc_nodes, dtype=float)
    masses = np.asarray(m_arr, dtype=float).copy()
    connectivity = np.asarray(kite_connectivity_arr, dtype=int)
    rest_lengths = np.asarray(l0_arr, dtype=float)
    stiffness = np.asarray(k_arr, dtype=float)
    damping = np.asarray(c_arr, dtype=float)
    link_types = np.asarray([str(t).lower() for t in linktype_arr])
    n_elements = len(connectivity)

    fixed_nodes = tuple(int(i) for i in struc_geometry.get("fixed_point_indices", [0]))
    (
        grid, is_beam, is_canopy, is_cable, cable_elements, triplets, arm_pairs,
        cables, pulleys,
    ) = _bridle_elements(
        connectivity, rest_lengths, stiffness, link_types, pulley_line_indices,
        canopy_sections, strut_sections,
    )

    # -- bridle relaxation -------------------------------------------------
    # Done before the tubes and the canopy are built, so their reference
    # configurations are taken on the shape the model actually starts from.
    # Only bridle nodes move (the wing is held), plus a rigid translation.
    if settings["relax_bridles"]:
        struc_nodes, settled = relax_bridles(
            struc_nodes,
            cables,
            pulleys,
            grid,
            kcu_node=int(fixed_nodes[0]),
            pull_force=float(settings["relax_pull_force"]),
            settle_force=float(settings["relax_settle_force"]),
            move_limit=float(settings["relax_move_limit"]),
        )
        if not settled:
            logger.warning(
                "the bridle did not settle; the coupled solve starts from a "
                "pre-stressed bridle and may not converge"
            )

    # -- inflatable tubes --------------------------------------------------
    beam_elements = np.flatnonzero(is_beam)
    diameters = stiffness[beam_elements]  # reader stores diameter in k_arr
    pressures = damping[beam_elements]  # ... and pressure in c_arr
    laws = scale_tube_laws(
        [
            InflatableTubeLaw.from_fit(float(diameter), float(pressure))
            for diameter, pressure in zip(diameters, pressures)
        ],
        float(settings["tube_stiffness_factor"]),
    )
    frames = build_frames(
        struc_nodes,
        connectivity[beam_elements],
        struc_geometry,
        strut_sections,
        junction_frame=str(settings["junction_frame"]),
    )
    if settings["mirror_frames"]:
        frames = symmetric_frames(struc_nodes, frames, connectivity[beam_elements])
    axial, shear = _tube_stiffnesses(laws, diameters, settings)
    tubes = build_inflatable_beam_elements(
        struc_nodes,
        connectivity[beam_elements],
        laws,
        frames,
        axial_stiffness=axial,
        shear_stiffness=shear,
        name=TUBES,
    )

    # -- canopy ------------------------------------------------------------
    # The cross pattern adds a node per quad. They are appended here, before the
    # model is built, so the node count the caller reads back from
    # ``structure.model.nodes`` already includes them -- every downstream array
    # (masses, external forces, tracking) is sized from that.
    fine_grid, refined_nodes = refine_grid(
        grid, struc_nodes, int(settings["canopy_refinement"])
    )
    if len(refined_nodes):
        struc_nodes = np.vstack([struc_nodes, refined_nodes])
        masses = np.concatenate([masses, np.zeros(len(refined_nodes))])
    triangles, extra_nodes, thickness_scale = canopy_mesh(
        fine_grid, struc_nodes, str(settings["canopy_pattern"])
    )
    if len(extra_nodes):
        struc_nodes = np.vstack([struc_nodes, extra_nodes])
        # Massless: they are a mesh refinement, not extra fabric. The canopy's
        # mass is already carried by the grid nodes the reader weighed.
        masses = np.concatenate([masses, np.zeros(len(extra_nodes))])

    # ``thickness_scale`` divides the stress resultant E*t between overlapping
    # triangles, so a pattern that lays two sets over the same quad does not
    # double the canopy stiffness. It must scale the PRODUCT, not the thickness:
    # thickness alone cancels out of E*t and would leave the mesh twice as stiff.
    thickness = float(settings["canopy_thickness"])
    stress_resultant = float(settings["canopy_stiffness"]) * thickness_scale
    canopy = build_membrane_elements(
        struc_nodes,
        triangles,
        thickness=thickness,
        youngs_modulus=stress_resultant / thickness,
        poisson_ratio=float(settings["canopy_poisson_ratio"]),
        name=CANOPY,
        wrinkling=bool(settings["canopy_wrinkling"]),
        slack_stiffness_ratio=float(settings["canopy_slack_stiffness_ratio"]),
    )

    model = StructuralModel(
        struc_nodes,
        [cables, pulleys, tubes, canopy],
        node_frames=frames,
        fixed_translation_nodes=fixed_nodes,
    )
    solver = build_solver(model, settings)

    cable_row = np.full(n_elements, -1, dtype=int)
    cable_row[cable_elements] = np.arange(len(cable_elements))
    pulley_row = np.full(n_elements, -1, dtype=int)
    for row, (first, second) in enumerate(arm_pairs):
        pulley_row[[first, second]] = row
    tube_row = np.full(n_elements, -1, dtype=int)
    tube_row[beam_elements] = np.arange(len(beam_elements))

    logger.info(
        "Billow model: %d nodes, %d DOF | %d cables, %d pulleys, %d tubes, "
        "%d membrane triangles (%d canopy springs replaced)",
        model.n_nodes,
        model.layout.n_dof,
        len(cable_elements),
        len(triplets),
        len(beam_elements),
        len(canopy.connectivity),
        int(is_canopy.sum()),
    )
    return BillowStructure(
        model=model,
        solver=solver,
        state=model.initial_state(),
        grid=grid,
        fine_grid=fine_grid,
        cable_row=cable_row,
        pulley_row=pulley_row,
        tube_row=tube_row,
        tube_laws=laws,
        masses=masses,
        fixed_node_indices=fixed_nodes,
    )


# --------------------------------------------------------------------------
# Solve
# --------------------------------------------------------------------------


def run_billow(structure: BillowStructure, f_ext_flat, config_structural_billow):
    """Solve for equilibrium under frozen nodal loads.

    Mirrors ``fem.structural_kite_fem.run_kite_fem``: returns the updated
    structure, the convergence verdict, the solved nodes and the flattened
    internal force in the convention the drivers use, ``f_res = f_int + f_ext``.
    Like ``run_kite_fem``, one call is a whole structural solve, not one step.

    With a ``move_limit`` each ``MinimumEnergySolver.solve`` is one
    trust-region step, so this walks them to force balance. The bound is what
    stops a slack-dominated configuration -- slack fabric, slack tension-only
    cable directions -- from handing IPOPT a near-zero-curvature direction
    that it answers with a step of order 1e4, overflowing the objective before
    restoration can help. It also means IPOPT's own verdict no longer implies
    equilibrium (it reports success for the *boxed* problem while sitting on
    the boundary), which is why the loop terminates on the force residual.
    """
    settings = resolve_config(config_structural_billow)
    forces = np.asarray(f_ext_flat, dtype=float).reshape(-1, 3)

    # Scale with the TOTAL load, not the largest nodal one. The total is a
    # property of the flight state; the per-node maximum is a property of the
    # discretisation, so keying off it makes the demand tighten as the mesh is
    # refined -- the opposite of what mesh independence requires.
    total_load = float(np.linalg.norm(forces.sum(axis=0)))
    peak_nodal = float(np.linalg.norm(forces, axis=1).max(initial=0.0))
    accept_at = max(
        float(settings["force_tolerance"]),
        float(settings["relative_force_tolerance"]) * max(total_load, peak_nodal),
    )
    rounds = int(settings["max_rounds"]) if settings["move_limit"] is not None else 1
    stalled = 0
    for _ in range(max(1, rounds)):
        solution = structure.solver.solve(
            forces, state=structure.state, model=structure.model
        )
        structure.state = solution.state
        if solution.residual_norm <= accept_at:
            break
        # A trust-region round that barely moves is not going to be rescued by
        # more of them; spending the whole budget on it just hides the stall.
        if solution.status == "Search_Direction_Becomes_Too_Small":
            stalled += 1
            if stalled >= 2:
                logger.info(
                    "minimum-energy solve stalled twice at residual %.3e N "
                    "(target %.3e N); accepting the best iterate",
                    solution.residual_norm,
                    accept_at,
                )
                break
        else:
            stalled = 0
    structure.last_solution = solution

    f_int = solution.internal_forces.copy()
    # At a fixed node the "internal force" is the reaction, so the residual
    # there is meaningless. Zero it out the way the FEM backend does, so the
    # driver's residual norms see only the free nodes.
    for node in structure.fixed_node_indices:
        f_int[node] = -forces[node]

    converged = solution.residual_norm <= accept_at
    if settings["print_info"]:
        logger.info(
            "Billow solve: %s in %d iterations (%d frame updates), residual "
            "%.3e N (target %.3e), strain energy %.4f J",
            solution.status,
            solution.iterations,
            solution.frame_updates,
            solution.residual_norm,
            accept_at,
            solution.strain_energy,
        )
    return (
        structure,
        bool(converged),
        solution.state.positions,
        f_int.flatten(),
    )


# --------------------------------------------------------------------------
# Rest lengths, stiffness and actuation
# --------------------------------------------------------------------------

#: Element sets whose rest length an actuator may drive, in lookup order.
_ACTUATABLE = (CABLES, PULLEYS)


def _row_of(structure: BillowStructure, element_index: int, names=_ACTUATABLE):
    """``(set_name, row)`` for a reader element, or ``(None, -1)``."""
    rows = {CABLES: structure.cable_row, PULLEYS: structure.pulley_row,
            TUBES: structure.tube_row}
    for name in names:
        row = int(rows[name][int(element_index)])
        if row >= 0:
            return name, row
    return None, -1


def get_rest_lengths(structure: BillowStructure, kite_connectivity_arr) -> Array:
    """Current rest lengths in the reader's element ordering.

    ``NaN`` where the element has no Billow counterpart -- the canopy springs
    the membrane replaced -- which keeps the array numeric and HDF5-friendly,
    the same convention the FEM backend uses for unmatched elements.
    """
    lengths = np.full(len(np.asarray(kite_connectivity_arr)), np.nan)
    for name, rows in (
        (CABLES, structure.cable_row),
        (PULLEYS, structure.pulley_row),
        (TUBES, structure.tube_row),
    ):
        column = structure.model.element_set(name).params[:, 0]
        mask = rows >= 0
        lengths[mask] = column[rows[mask]]
    return lengths


def get_rest_length(structure: BillowStructure, element_index: int) -> float:
    """One element's current rest length [m], in the reader's ordering."""
    name, row = _row_of(structure, element_index, (CABLES, PULLEYS, TUBES))
    if name is None:
        raise KeyError(f"element {element_index} has no Billow counterpart")
    return float(structure.model.element_set(name).params[row, 0])


def set_rest_length(
    structure: BillowStructure, element_index: int, rest_length: float
) -> BillowStructure:
    """Set one element's rest length [m], in the reader's element ordering.

    Rest lengths are NLP parameters, so actuating a tape rebuilds the parameter
    table but never the compiled graph. A pulley takes the length of the whole
    rope, and either of its arms addresses the same element.
    """
    name, row = _row_of(structure, element_index)
    if name is None:
        raise KeyError(
            f"element {element_index} is not an actuatable Billow cable or "
            "pulley (it is a tube, or a canopy spring the membrane replaced)"
        )
    element_set = structure.model.element_set(name)
    values = element_set.params[:, 0].copy()
    values[row] = float(rest_length)
    structure.model = structure.model.replaced(
        element_set.with_param_column("rest_length", values)
    )
    return structure


def update_rest_length(
    structure: BillowStructure, element_index: int, delta_length: float
) -> BillowStructure:
    """Increment one element's rest length by ``delta_length`` [m]."""
    current = get_rest_length(structure, element_index)
    return set_rest_length(structure, element_index, current + float(delta_length))


def get_stiffnesses(structure: BillowStructure, n_elements: int) -> Array:
    """Cable and pulley stiffnesses [N/m] in the reader's ordering, NaN elsewhere."""
    stiffness = np.full(int(n_elements), np.nan)
    for name, rows in ((CABLES, structure.cable_row), (PULLEYS, structure.pulley_row)):
        column = structure.model.element_set(name).params[:, 1]
        mask = rows >= 0
        stiffness[mask] = column[rows[mask]]
    return stiffness


def set_stiffnesses(structure: BillowStructure, values) -> BillowStructure:
    """Set the cable and pulley stiffnesses [N/m], in the reader's ordering.

    The counterpart of ``pss.structural_wireframe.set_stiffnesses``: what a stiffness
    continuation ramp writes into. Like the rest lengths it is a parameter
    update, not a rebuild. Entries with no cable or pulley counterpart are
    ignored.
    """
    values = np.asarray(values, dtype=float)
    for name, rows in ((CABLES, structure.cable_row), (PULLEYS, structure.pulley_row)):
        element_set = structure.model.element_set(name)
        column = element_set.params[:, 1].copy()
        mask = rows >= 0
        column[rows[mask]] = values[mask]
        structure.model = structure.model.replaced(
            element_set.with_param_column("stiffness", column)
        )
    return structure

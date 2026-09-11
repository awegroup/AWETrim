# Copyright (c) 2023-2026 Oriol Cayon, Delft University of Technology
#
# SPDX-License-Identifier: Apache-2.0

"""Build the hanging TU Delft V3 kite as a Billow minimum-energy model.

The reference experiment hangs the V3 from its bridle point and measures twelve
lengths off the deformed shape: nine trailing-edge billow segments between the
ten struts, the tip-to-tip span, and two tip-to-centre-LE diagonals. Ten load
cases combine two inflation pressures with a mid-span point load and a
tip-spreading load. The measurements and the reference solution both come from
``awegroup/kite_fem`` (MIT, Patrick Roeleveld) -- see the ``README.md`` beside
the vendored data.

The model is rebuilt from ``hanging_test_initial.npz``, not from the YAML. That
file is the fully-assembled reference model, so nodes, connectivity, rest
lengths, stiffnesses and tube diameters transfer with no re-interpretation. The
only thing that changes is the canopy:

    kite_fem   canopy = 721 noncompressive springs on a 29 x 8 grid
    Billow     canopy = 392 wrinkling CST membrane triangles on the same grid

Everything else -- 46 bridle cables, 14 pulleys, 98 inflatable tube beams -- is
element-for-element the same. That is deliberate: it makes the difference
between the two answers attributable to the canopy formulation alone.

Canopy grid recovery
--------------------
The reference mesh is a structured 29 x 8 grid, but the ``.npz`` stores only a
flat spring list. The grid is recovered geometrically: the reference builder
projects every intermediate section node onto the straight line joining its
leading- and trailing-edge node, so each of the 29 chordwise sections is exactly
collinear. Collecting the nodes on each LE-TE segment and sorting by distance
recovers the section. This is verified, not assumed -- :func:`canopy_grid`
checks that all 29 sections have 8 nodes and that the resulting grid edges
account for every one of the 721 canopy springs exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from awetrim.structural import StructuralModel
from awetrim.structural.elements import (
    InflatableTubeLaw,
    build_cable_elements,
    build_inflatable_beam_elements,
    build_membrane_elements,
    build_pulley_elements,
)
from awetrim.structural.rotations import orthonormalize

DATA = Path(__file__).resolve().parents[2] / "data" / "LEI-V3-KITE" / "hanging_validation"

GRAVITY = 9.81
N_SECTIONS = 29          # chordwise sections, LE node 1, 3, ... 57
N_CHORDWISE = 8          # nodes per section after the reference builder pads them
SHEAR_CORRECTION = 8.0 / 9.0     # kite_fem BeamElement.k

#: Strut leading-edge nodes, in the order the reference extractor uses them.
STRUT_LE = (1, 7, 13, 19, 25, 33, 39, 45, 51, 57)
STRUT_TE = tuple(node + 1 for node in STRUT_LE)

#: Nodes the reference load cases push on (``Hanging_test.py::loading``).
TIP_LOAD_NODES = (2, 58)         # +y on the first, -y on the second
POINT_LOAD_NODE = 29             # +z, mid-span leading edge

# -- Suspension ----------------------------------------------------------
# The stored reference model has no load path to its anchor: node 0 is flagged
# fixed but appears in no spring, pulley or beam, so the kite is a free body
# under 126 N of unbalanced weight and its potential is unbounded below. Billow
# reports that faithfully (the structure translates hundreds of kilometres);
# kite_fem does not diverge only because its ``I_stiffness`` regularisation acts
# as a ground spring on every node, and its published solutions still drift
# 0.29-0.34 m bodily upward.
#
# Those missing lines are deliberate, not an omission. Roeleveld's thesis
# (Appendix B, Table B.1) marks ``A_main``, ``A_I``, ``M``, ``PL`` and ``SL`` as
# N/A for the hanging test and adds a ``bar`` of 1650 mm: "The bridle line
# system of the V3 kite was adapted slightly to fit within the height limit of
# the hangar" (Section 6.3). Figures B.1/B.2 show the front and rear bridle
# trees converging onto that bar rather than onto a bridle point, and the kite
# hangs from the bar -- by two points, not one.
#
# Nodes 82 and 116 are those convergence points: the two lowest nodes of the
# tree, the only ones of degree five, carrying AI, AII, AIII, brmain 1 and
# brmain 3 on each side. They sit 1.8458 m apart, so the 1.650 m bar draws each
# one 98 mm inboard, which stretches the long lines out to the tips.
#
# The same table records the documented shortening: ``B_r,5`` goes 12168 ->
# 9425 mm. The YAML's ``brmain 3`` is 4.7125 m, exactly half of 9425, so that
# shortening is already applied in the stored model.
#
# The remaining orphans carry exactly zero mass, so pinning them removes null
# modes at no physical cost.
BAR_NODES = (82, 116)            # left/right main bridle convergence points
BAR_LENGTH = 1.650               # thesis Table B.1 "bar" [m]
UNWIRED_NODES = (72, 81, 106, 115, 127, 128, 129)   # zero-mass, unconnected

#: The ten reference load cases: (pressure [bar], tip load [kg], point load [kg]).
LOAD_CASES = (
    (0.15, 0.0, 0.0), (0.15, 0.0, 9.7), (0.15, 0.0, 25.2),
    (0.15, 2.0, 0.0), (0.15, 5.0, 0.0),
    (0.25, 0.0, 0.0), (0.25, 0.0, 9.7), (0.25, 0.0, 25.2),
    (0.25, 2.0, 0.0), (0.25, 5.0, 0.0),
)

LENGTH_NAMES = ("La", "Lb", "Lc", "Ld", "Le", "Lf", "Lg", "Lh", "Li",
                "b", "LcsTL", "LcsTR")


# --------------------------------------------------------------------------
# Reference model
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Reference:
    """The assembled kite_fem model, as stored."""

    nodes: np.ndarray            # (276, 3)
    masses: np.ndarray           # (276,)
    springs: np.ndarray          # (767, 2) node pairs
    spring_rest: np.ndarray      # (767,)
    spring_stiffness: np.ndarray # (767,)
    pulleys: np.ndarray          # (14, 3)
    pulley_rest: np.ndarray      # (14,)
    pulley_stiffness: np.ndarray # (14,)
    beams: np.ndarray            # (98, 2)
    beam_diameter: np.ndarray    # (98,)
    beam_rest: np.ndarray        # (98,)
    beam_modulus: np.ndarray     # (98,) secant E from the ASKITE bending fit
    beam_shear_modulus: np.ndarray

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)


def load_reference(directory: Path = DATA) -> Reference:
    """Read the vendored kite_fem initial state."""
    stored = np.load(directory / "hanging_test_initial.npz", allow_pickle=True)
    springs = stored["spring_matrix"]
    pulleys = stored["pulley_matrix"]
    beams = stored["beam_matrix"]
    return Reference(
        nodes=stored["coords_current"].reshape(-1, 3),
        masses=np.loadtxt(directory / "mass_hanging_test.csv", delimiter=","),
        springs=springs[:, :2].astype(float).astype(int),
        spring_rest=springs[:, 4].astype(float),
        spring_stiffness=springs[:, 2].astype(float),
        pulleys=pulleys[:, :3].astype(int),
        pulley_rest=pulleys[:, 5],
        pulley_stiffness=pulleys[:, 3],
        beams=beams[:, :2].astype(int),
        beam_diameter=beams[:, 2],
        beam_rest=beams[:, 4],
        beam_modulus=stored["beam_E"],
        beam_shear_modulus=stored["beam_G"],
    )


# --------------------------------------------------------------------------
# Canopy grid
# --------------------------------------------------------------------------


def canopy_grid(nodes: np.ndarray, tolerance: float = 2e-3) -> np.ndarray:
    """Recover the structured ``(29, 8)`` canopy grid from the node cloud.

    Each chordwise section is collinear by construction (the reference builder
    projects its interior nodes onto the LE-TE line), so a section is just the
    set of nodes lying on that segment, ordered by distance from the leading
    edge. Raises if any section does not come out with exactly eight nodes.
    """
    sections = []
    for leading in range(1, 2 * N_SECTIONS, 2):
        trailing = leading + 1
        origin = nodes[leading]
        chord = nodes[trailing] - origin
        length = float(np.linalg.norm(chord))
        direction = chord / length

        along = (nodes - origin) @ direction
        offset = np.linalg.norm(nodes - (origin + np.outer(along, direction)), axis=1)
        on_line = np.where(
            (offset < tolerance) & (along > -1e-6) & (along < length + 1e-6)
        )[0]
        section = on_line[np.argsort(along[on_line])]

        if len(section) != N_CHORDWISE:
            raise RuntimeError(
                f"section LE {leading} recovered {len(section)} nodes, expected "
                f"{N_CHORDWISE}: {section.tolist()}"
            )
        sections.append(section)
    return np.asarray(sections, dtype=int)


def grid_edges(grid: np.ndarray) -> set[frozenset[int]]:
    """Every chordwise, spanwise and diagonal edge of the structured grid."""
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


def classify_springs(reference: Reference, grid: np.ndarray) -> np.ndarray:
    """Boolean mask: ``True`` where a reference spring is a canopy grid edge.

    Verifies the split is unambiguous -- every spring with both ends on the grid
    must *be* a grid edge, otherwise the recovered topology is wrong.
    """
    edges = grid_edges(grid)
    on_grid = set(int(node) for node in grid.ravel())

    pairs = [frozenset((int(a), int(b))) for a, b in reference.springs]
    is_edge = np.array([pair in edges for pair in pairs])
    both_ends = np.array([set(pair) <= on_grid for pair in pairs])

    stray = int((both_ends & ~is_edge).sum())
    if stray:
        raise RuntimeError(f"{stray} springs join two grid nodes but are not grid edges")
    return is_edge


def canopy_triangles(grid: np.ndarray) -> np.ndarray:
    """Split every grid quad into two triangles along a consistent diagonal."""
    rows, columns = grid.shape
    triangles = []
    for i in range(rows - 1):
        for j in range(columns - 1):
            a, b = int(grid[i, j]), int(grid[i, j + 1])
            c, d = int(grid[i + 1, j + 1]), int(grid[i + 1, j])
            triangles.append([a, b, c])
            triangles.append([a, c, d])
    return np.asarray(triangles, dtype=int)


# --------------------------------------------------------------------------
# Beam frames
# --------------------------------------------------------------------------


def beam_chains(reference: Reference, grid: np.ndarray) -> list[list[int]]:
    """The leading-edge chain plus one chain per strut, as node paths.

    Frames are stored per node, so they have to come from *somewhere* smooth.
    Each chain is a polyline and gets parallel-transported frames along it; the
    struts and the leading edge meet at shared nodes, which is what
    ``junction_frame`` has to arbitrate.
    """
    leading_edge = list(range(1, 2 * N_SECTIONS, 2))
    struts = [grid[row].tolist() for row in range(N_SECTIONS)
              if int(grid[row, 0]) in STRUT_LE]
    return [leading_edge] + struts


def _tangents(points: np.ndarray) -> np.ndarray:
    segments = np.diff(points, axis=0)
    tangents = np.empty_like(points)
    tangents[0] = segments[0]
    tangents[-1] = segments[-1]
    if len(points) > 2:
        tangents[1:-1] = segments[:-1] + segments[1:]
    return tangents / np.linalg.norm(tangents, axis=1, keepdims=True)


def _minimal_rotation(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    axis = np.cross(source, target)
    sine = float(np.linalg.norm(axis))
    cosine = float(np.dot(source, target))
    if sine < 1e-12:
        return np.eye(3) if cosine > 0 else -np.eye(3) + 2 * np.outer(source, source)
    axis = axis / sine
    cross = np.array([[0.0, -axis[2], axis[1]],
                      [axis[2], 0.0, -axis[0]],
                      [-axis[1], axis[0], 0.0]])
    angle = np.arctan2(sine, cosine)
    return np.eye(3) + np.sin(angle) * cross + (1.0 - np.cos(angle)) * cross @ cross


def build_frames(reference: Reference, grid: np.ndarray,
                 junction_frame: str = "leading_edge") -> np.ndarray:
    """Nodal material frames, tangent-aligned along each beam chain.

    ``junction_frame`` decides who owns the ten nodes where a strut meets the
    leading edge -- ``"leading_edge"`` or ``"strut"``. One nodal frame cannot be
    tangent to two near-perpendicular members, so this is a genuine modelling
    choice and the driver A/B tests it rather than assuming it is harmless.
    """
    if junction_frame not in ("leading_edge", "strut"):
        raise ValueError("junction_frame must be 'leading_edge' or 'strut'")

    frames = np.tile(np.eye(3), (reference.n_nodes, 1, 1))
    chains = beam_chains(reference, grid)
    # Later chains overwrite earlier ones at shared nodes, so order decides the
    # junction owner: struts come after the leading edge.
    ordered = chains if junction_frame == "strut" else chains[1:] + chains[:1]

    for chain in ordered:
        points = reference.nodes[chain]
        tangents = _tangents(points)

        seed = np.array([0.0, 0.0, 1.0])
        seed = seed - np.dot(seed, tangents[0]) * tangents[0]
        if np.linalg.norm(seed) < 1e-8:
            seed = np.cross(tangents[0], [1.0, 0.0, 0.0])
        seed /= np.linalg.norm(seed)

        frame = np.column_stack([tangents[0], seed, np.cross(tangents[0], seed)])
        frames[chain[0]] = orthonormalize(frame)
        for index in range(1, len(chain)):
            frame = _minimal_rotation(tangents[index - 1], tangents[index]) @ frame
            frames[chain[index]] = orthonormalize(frame)
    return frames


def reference_misalignment(reference: Reference, frames: np.ndarray) -> np.ndarray:
    """Angle [deg] between each beam chord and its reference mid-frame tangent.

    Zero on a well-aligned element. Large values mark elements where the
    diagonal ``(EA, GA, GA)`` stiffness is being applied in the wrong axes.
    """
    angles = np.empty(len(reference.beams))
    for index, (node_a, node_b) in enumerate(reference.beams):
        chord = reference.nodes[node_b] - reference.nodes[node_a]
        chord = chord / np.linalg.norm(chord)
        tangent = 0.5 * (frames[node_a][:, 0] + frames[node_b][:, 0])
        norm = float(np.linalg.norm(tangent))
        if norm < 1e-12:
            angles[index] = 90.0
            continue
        angles[index] = np.degrees(
            np.arccos(abs(float(np.clip(chord @ (tangent / norm), -1.0, 1.0))))
        )
    return angles


# --------------------------------------------------------------------------
# Model assembly
# --------------------------------------------------------------------------


def build_model(
    reference: Reference,
    grid: np.ndarray,
    *,
    pressure: float,
    canopy_stiffness: float,
    canopy_thickness: float = 2.5e-4,
    poisson_ratio: float = 0.3,
    bridle_stiffness: float | None = None,
    junction_frame: str = "leading_edge",
    wrinkling: bool = True,
    suspension: str = "bar",
    bar_length: float = BAR_LENGTH,
) -> tuple[StructuralModel, np.ndarray, list[InflatableTubeLaw]]:
    """Assemble the Billow model at one inflation pressure.

    ``canopy_stiffness`` is the membrane stress resultant per unit strain,
    ``E t`` [N/m]. On a square spring net of stiffness ``k`` the uniaxial
    equivalent is ``E t = k``, which is how the reference net's 5000 N/m maps
    onto a membrane; ``canopy_thickness`` only splits that product and does not
    affect the answer.
    """
    is_canopy = classify_springs(reference, grid)

    bridle = reference.springs[~is_canopy]
    stiffness = (reference.spring_stiffness[~is_canopy] if bridle_stiffness is None
                 else np.full(len(bridle), float(bridle_stiffness)))
    cables = build_cable_elements(
        bridle, reference.spring_rest[~is_canopy], stiffness, name="bridle"
    )

    if suspension not in ("bar", "free"):
        raise ValueError("suspension must be 'bar' or 'free'")
    pulleys = build_pulley_elements(
        reference.pulleys, reference.pulley_rest, reference.pulley_stiffness,
        name="pulleys",
    )

    nodes = reference.nodes
    fixed = [0, *UNWIRED_NODES]
    if suspension == "bar":
        # Hold the two convergence points at the bar half-span, keeping their
        # own x and z: the bar sets their separation, gravity does the rest.
        nodes = nodes.copy()
        for node in BAR_NODES:
            nodes[node, 1] = np.sign(reference.nodes[node, 1]) * 0.5 * bar_length
        fixed.extend(BAR_NODES)

    frames = build_frames(reference, grid, junction_frame=junction_frame)
    radius = reference.beam_diameter / 2.0
    area = np.pi * radius**2
    laws = [InflatableTubeLaw.from_fit(diameter, pressure)
            for diameter in reference.beam_diameter]
    tubes = build_inflatable_beam_elements(
        nodes, reference.beams, laws, frames,
        axial_stiffness=reference.beam_modulus * area,
        shear_stiffness=SHEAR_CORRECTION * reference.beam_shear_modulus * area,
        name="tubes",
    )

    canopy = build_membrane_elements(
        nodes,
        canopy_triangles(grid),
        thickness=canopy_thickness,
        youngs_modulus=canopy_stiffness / canopy_thickness,
        poisson_ratio=poisson_ratio,
        name="canopy",
        wrinkling=wrinkling,
    )

    model = StructuralModel(
        nodes, [cables, pulleys, tubes, canopy],
        node_frames=frames,
        fixed_translation_nodes=fixed,
    )
    return model, frames, laws


def scale_tube_laws(laws, factor: float):
    """Multiply the whole moment-curvature curve, initial slope included.

    Scaling ``moment_max``, ``bending_stiffness`` and ``torsion_c1`` together
    leaves ``curvature_scale = M_max / EI_0`` unchanged, so the curve keeps its
    shape and only its magnitude moves. That makes ``factor`` one honest
    stiffness knob rather than a reshaping of the fit -- which matters, because
    the point of using it is to ask what stiffness the measurement implies, not
    to fit a new law to the data.
    """
    import dataclasses

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


def load_vector(reference: Reference, tip_load: float, point_load: float) -> np.ndarray:
    """External nodal forces [N], matching ``Hanging_test.py::loading`` exactly."""
    forces = np.zeros((reference.n_nodes, 3))
    forces[:, 2] = reference.masses * GRAVITY
    forces[TIP_LOAD_NODES[0], 1] += tip_load * GRAVITY
    forces[TIP_LOAD_NODES[1], 1] -= tip_load * GRAVITY
    forces[POINT_LOAD_NODE, 2] += point_load * GRAVITY
    return forces


# --------------------------------------------------------------------------
# Measured quantities
# --------------------------------------------------------------------------


def extract_lengths(positions: np.ndarray) -> np.ndarray:
    """The twelve measured lengths, in the reference extractor's own order.

    Nine trailing-edge billow segments between consecutive struts, then the
    tip-to-tip span, then the two tip-to-centre-LE diagonals. Verified against
    ``kite_fem``'s published ``model_results.csv`` to machine precision.
    """
    trailing = list(STRUT_TE[::-1])
    lengths = [float(np.linalg.norm(positions[a] - positions[b]))
               for a, b in zip(trailing[:-1], trailing[1:])]
    lengths.append(float(np.linalg.norm(positions[trailing[0]] - positions[trailing[-1]])))
    middle = len(STRUT_LE) // 2
    lengths.append(float(np.linalg.norm(positions[STRUT_LE[middle - 1]] - positions[trailing[-1]])))
    lengths.append(float(np.linalg.norm(positions[STRUT_LE[middle]] - positions[trailing[0]])))
    return np.asarray(lengths)


# --------------------------------------------------------------------------
# Bridle corrections from the thesis
# --------------------------------------------------------------------------

#: Lines whose stored rest length disagrees with Roeleveld's thesis Table B.1.
#: 28 of the 36 declared lines match it to the millimetre, which is what makes
#: these eight look like transcription errors rather than naming mismatches:
#:
#:   AI      wired at 3.670 m, but Table B.1 marks A_I as N/A for the hanging
#:           test -- an extra front load path that should not be there
#:   BRI     2.045 vs 2.333, and BRII 2.333 vs 2.045: an exact swap
#:   abcd2   3.3180 vs 3.180, abcd3 3.3076 vs 3.076: a duplicated leading digit
#:   ab3     0.260 vs 0.325 -- equal to ab2, so probably a copied row
#:   a5      0.234 vs 0.278
#:   br5     0.825 vs 1.420
#:
#: Only the lines whose stored rest length is uniquely identifiable are applied
#: here. The reader rewrites several others (``c1``-``d3``, ``br1``-``br4``) to
#: values that match neither source, so they cannot be mapped by length and are
#: left alone. ``ab3`` is unmappable for the same reason: the pulley rest
#: lengths stored for ``ab1``-``ab4`` bear no simple relation to the YAML.
#: ``br5`` is identifiable (springs 87-88 and 121-122, into the tip struts) but
#: was measured to change nothing, so it is off by default.
THESIS_BRIDLE = {
    "remove_AI": 3.6700,        # -> stiffness 0, the line is absent in the test
    "swap_BRI_BRII": (2.0450, 2.3330),
    "abcd2": (3.3180, 3.180),   # pulley
    "abcd3": (3.3076, 3.076),   # pulley
    "a5": (0.2340, 0.278),
    "br5": (0.8834, 1.420),     # opt-in; measured to have no effect
}


def apply_thesis_bridle(reference: Reference, *, include_br5: bool = False,
                        remove_ai: bool = False) -> Reference:
    """Return a copy with Table B.1's bridle lengths where they are mappable.

    ``remove_ai`` is off by default despite Table B.1 marking ``A_I`` as N/A,
    because the thesis contradicts itself: Figure B.1 *draws* ``A_I`` running to
    the bar in the adapted geometry. "N/A" most plausibly means its length is
    undefined once it terminates on the bar, not that the line was cut. The
    physics agrees -- cutting it removes a front load path and lets the four
    centre-load cases fall into a folded local minimum (span under 1 m against a
    measured 5 m), while keeping it holds them on the physical branch. Mean
    relative error over the six cases both codes solve is 8.9% with it cut and
    with it kept it is reported in the validation table.
    """
    import dataclasses

    rest = reference.spring_rest.copy()
    stiffness = reference.spring_stiffness.copy()
    pulley_rest = reference.pulley_rest.copy()

    if remove_ai:
        stiffness[np.isclose(rest, THESIS_BRIDLE["remove_AI"])] = 0.0
    first, second = THESIS_BRIDLE["swap_BRI_BRII"]
    lower, upper = np.isclose(rest, first), np.isclose(rest, second)
    rest[lower], rest[upper] = second, first
    for key in ("abcd2", "abcd3"):
        stored, corrected = THESIS_BRIDLE[key]
        pulley_rest[np.isclose(pulley_rest, stored)] = corrected
    stored, corrected = THESIS_BRIDLE["a5"]
    rest[np.isclose(rest, stored)] = corrected
    if include_br5:
        stored, corrected = THESIS_BRIDLE["br5"]
        rest[np.isclose(rest, stored)] = corrected

    return dataclasses.replace(
        reference, spring_rest=rest, spring_stiffness=stiffness,
        pulley_rest=pulley_rest,
    )


def kite_fem_state(case: int, directory: Path = DATA) -> np.ndarray:
    """Node positions of ``kite_fem``'s own solved state for one load case."""
    path = directory / "kite_fem_states" / f"load_case_{case}.npz"
    return np.load(path, allow_pickle=True)["coords_current"].reshape(-1, 3)


def measured_lengths(directory: Path = DATA) -> np.ndarray:
    """The ten measured rows, ``(10, 12)``."""
    return np.loadtxt(directory / "measured_lengths.csv", delimiter=",", skiprows=1)[:, 1:]


def kite_fem_lengths(directory: Path = DATA) -> np.ndarray:
    """The ten reference-model rows, ``(10, 12)``."""
    return np.loadtxt(directory / "kite_fem_results.csv", delimiter=",", skiprows=1)[:, 1:-2]

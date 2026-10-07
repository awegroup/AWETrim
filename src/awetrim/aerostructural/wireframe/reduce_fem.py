"""Reduce the full FEM geometry to a wireframe (PSM-format) geometry.

``struc_geometry_FEM_full.yaml`` describes the kite at the resolution Billow's
full model needs: 28 canopy sections, strut tubes with intermediate chord
nodes, and the complete bridle down to the individual a/b/c/d lines that fan
onto each strut. A wireframe has none of that. It needs one rib per strut with
two wing nodes -- where the front and the rear bridle attach -- and the lines
between them, which is what ``struc_geometry_PSM_reduced.yaml`` is. That file
is an exact reduction of the full one, and this module performs the same
reduction from the full geometry, so a wireframe and a Billow run start from
one input file.

**Bridle.** Rules read off the two files, not invented:

* A line from a wing attachment straight to a pulley sheave that carries
  nothing else is absorbed into the pulley: the sheave moves onto the wing and
  the rope grows by twice the line (``AIII`` + ``a5_equiv`` -> a pulley over the
  wing tip, 11.42 + 2 x 0.254 = 11.928 m; ``Br_5`` + ``br_5_equiv`` likewise).
* The attachments of one strut on one side of its chord (front: x/c < 0.5,
  rear: beyond) form a group. Each group's lines are followed toward the KCU
  until they meet another group's: everything below that knot is the group's
  FAN. A fan of more than one element (the a/b pulley, c, d, the ab_cd pulley
  and A_i on the LE-V3) becomes ONE equivalent line from the group's front-most
  (or rear-most) attachment to the knot, with rest length equal to the built
  distance -- ``A1_equiv`` = 3.708 m, as in the reduced file. A fan of one line
  (``br_i``) is kept as it is.
* Everything above the fans -- the cascade, the main lines, the pulleys, the
  tapes -- is copied unchanged.

**Wing.** Per rib bay, the reduced file's topology (``le``, ``strut``, ``te``
and an X of diagonals), with stiffnesses from the SAME material the Billow
model uses, so the wing deforms little:

* ``le`` / ``strut``: compressive axial springs, ``k = EA / L`` with ``EA`` from
  Billow's inflatable-tube law at the file's tube diameters and pressure
  (``structural_billow.tube_axial_stiffness``);
* ``te``: tension-only, the canopy strip of half the chord, ``k = E t (b/2) / a``;
* diagonals: tension-only, the canopy's in-plane shear, ``k = G t d^2 / (2 a b)``
  with ``G = E / (2 (1 + nu))`` -- an X-brace storing the energy the membrane
  stores in pure shear.

``E t`` and ``nu`` are the membrane's (``structural_billow.canopy_stiffness``,
``canopy_poisson_ratio``); ``a`` is the bay span, ``b`` its chord, ``d`` the
diagonal.

**Mass.** The full model's nodal masses (reader masses, canopy and tubes
included) are lumped onto the nearest reduced wing node; the bridle carries the
mass the PSM reader computes for the reduced lines, and the wing absorbs the
difference, so the total matches the full model.
"""

from __future__ import annotations

import copy
from collections import deque
from typing import Any, Mapping

import numpy as np

#: Membrane values used when the config does not carry a structural_billow block.
_DEFAULT_CANOPY_STIFFNESS = 4.93e5  # E*t [N/m]
_DEFAULT_CANOPY_POISSON = 0.3


def _positions(fem: Mapping[str, Any], nodes=None) -> dict[int, np.ndarray]:
    """YAML node id -> position; ``nodes`` (indexed by id) overrides the YAML."""
    positions = {0: np.asarray(fem.get("bridle_point_node", [0, 0, 0]), dtype=float)}
    for key in ("wing_particles", "bridle_particles"):
        for row in fem[key]["data"]:
            positions[int(row[0])] = np.asarray(row[1:4], dtype=float)
    if nodes is not None:
        nodes = np.asarray(nodes, dtype=float)
        for node in positions:
            positions[node] = nodes[node].copy()
    return positions


def _ribs(fem, positions):
    """Strut ribs as ``(name, chord node chain LE->TE, strut row)``, +y first."""
    rows = fem["strut_tubes"]["data"]
    headers = fem["strut_tubes"]["headers"]
    ribs = [dict(zip(headers, row)) for row in rows]
    ribs.sort(key=lambda rib: -positions[int(rib["node_indices"][0])][1])
    return ribs


def _chord_fraction(point, le, te):
    chord = te - le
    return float(np.dot(point - le, chord) / np.dot(chord, chord))


class _Bridle:
    """The bridle as a list of elements: lines ``[a, b]`` and pulleys ``[a, sheave, b]``."""

    def __init__(self, fem):
        lines = {row[0]: row for row in fem["bridle_lines"]["data"]}
        self.headers = list(fem["bridle_lines"]["headers"])
        self.elements = []
        for row in fem["bridle_connections"]["data"]:
            props = dict(zip(self.headers, lines[row[0]]))
            self.elements.append(
                {"name": row[0], "nodes": [int(n) for n in row[1:]], **props}
            )

    def edges(self):
        for index, element in enumerate(self.elements):
            nodes = element["nodes"]
            for first, second in zip(nodes[:-1], nodes[1:]):
                yield index, first, second

    def adjacency(self):
        graph: dict[int, set[int]] = {}
        for _, a, b in self.edges():
            graph.setdefault(a, set()).add(b)
            graph.setdefault(b, set()).add(a)
        return graph

    def incident(self, node):
        return [i for i, e in enumerate(self.elements) if node in e["nodes"]]


def _path_to(graph, start, target=0):
    """Shortest node path ``start -> target`` (BFS, deterministic order)."""
    previous = {start: None}
    queue = deque([start])
    while queue:
        node = queue.popleft()
        if node == target:
            break
        for neighbour in sorted(graph.get(node, ())):
            if neighbour not in previous:
                previous[neighbour] = node
                queue.append(neighbour)
    if target not in previous:
        raise ValueError(f"bridle node {start} is not connected to the KCU")
    path, node = [], target
    while node is not None:
        path.append(node)
        node = previous[node]
    return path[::-1]


def reduce_bridle(fem, positions, wing_attachment_rib):
    """Apply the bridle rules; returns ``(elements, groups)``.

    ``wing_attachment_rib`` maps every node a bridle line may attach to on the
    wing to ``(rib index, x/c)``. ``groups`` maps ``(rib, "front"|"rear")`` to
    the attachment node the reduced wing keeps for it.
    """
    bridle = _Bridle(fem)

    # -- 1. lines into a bare pulley sheave are absorbed into the pulley ----
    for index, element in list(enumerate(bridle.elements)):
        if len(element["nodes"]) != 2:
            continue
        a, b = element["nodes"]
        for attachment, sheave in ((a, b), (b, a)):
            if attachment not in wing_attachment_rib:
                continue
            around = bridle.incident(sheave)
            pulleys = [
                i for i in around
                if len(bridle.elements[i]["nodes"]) == 3
                and bridle.elements[i]["nodes"][1] == sheave
            ]
            if len(pulleys) == 1 and len(around) == 2:
                pulley = bridle.elements[pulleys[0]]
                length = float(np.linalg.norm(positions[a] - positions[b]))
                pulley["nodes"][1] = attachment
                pulley["l0"] = float(pulley["l0"]) + 2.0 * float(element["l0"])
                pulley["name"] = element["name"]  # e.g. a5_equiv, as reduced
                element["absorbed"] = True
                del length
                break
    bridle.elements = [e for e in bridle.elements if not e.get("absorbed")]

    # -- 2. group the attachments per rib and side -------------------------
    attached = {n for e in bridle.elements for n in e["nodes"]} & set(wing_attachment_rib)
    groups: dict[tuple[int, str], list[int]] = {}
    for node in attached:
        rib, fraction = wing_attachment_rib[node]
        side = "front" if fraction < 0.5 else "rear"
        groups.setdefault((rib, side), []).append(node)
    keep: dict[tuple[int, str], int] = {}
    for key, nodes in groups.items():
        nodes.sort(key=lambda n: wing_attachment_rib[n][1])
        keep[key] = nodes[0] if key[1] == "front" else nodes[-1]

    # -- 3. collapse each group's fan into one equivalent line --------------
    graph = bridle.adjacency()
    paths = {key: [_path_to(graph, n) for n in nodes] for key, nodes in groups.items()}
    reached = {key: {n for p in ps for n in p} for key, ps in paths.items()}
    removed: set[int] = set()
    added = []
    for key, nodes in groups.items():
        others = set().union(*(r for k, r in reached.items() if k != key))
        knot = next(n for n in paths[key][0] if n in others or n == 0)
        fan_nodes = set()
        for path in paths[key]:
            fan_nodes.update(path[: path.index(knot)] if knot in path else path)
        fan = [
            i for i, e in enumerate(bridle.elements)
            if set(e["nodes"]) & fan_nodes and set(e["nodes"]) <= fan_nodes | {knot}
        ]
        # Only a self-contained fan is collapsed: every element touching a fan
        # node must lie inside it (a tip pulley shared with the cascade is not).
        touching = {i for i, e in enumerate(bridle.elements) if set(e["nodes"]) & fan_nodes}
        if len(fan) <= 1 or touching != set(fan):
            continue
        top = next(
            (bridle.elements[i] for i in fan if knot in bridle.elements[i]["nodes"]),
            bridle.elements[fan[-1]],
        )
        node = keep[key]
        added.append({
            **{h: top.get(h) for h in bridle.headers},
            "name": f"{top['name']}_equiv",
            "nodes": [node, knot],
            "l0": round(float(np.linalg.norm(positions[node] - positions[knot])), 6),
            "linktype": "noncompressive",
        })
        removed.update(fan)
    elements = [e for i, e in enumerate(bridle.elements) if i not in removed] + added
    return elements, bridle.headers, keep


def reduce_fem_geometry(
    fem: Mapping[str, Any],
    *,
    nodes=None,
    masses=None,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The wireframe (PSM-format) geometry of a full FEM geometry.

    ``nodes`` / ``masses``: the full model's node positions and masses as the
    coupled reader returns them (``coupled.read_struc_geometry_yaml.main``,
    ids 0..N-1, the YAML's ids first), ideally after Billow's bridle
    relaxation (``structural_billow.relax_bridle_nodes``) so both models start
    from one shape. Without ``nodes`` the YAML positions are used; without
    ``masses`` the wing gets the file's ``mass_without_bridles``. ``config``
    supplies the ``structural_billow`` material block.

    The result reads with ``wireframe.structural_geometry_io.main``: wing nodes
    1..2R (odd = front attachment, even = rear, rib order +y to -y), bridle
    nodes after them, the KCU at 0.
    """
    from scipy.optimize import nnls

    from awetrim.aerostructural.billow.structural_billow import (
        resolve_config,
        tube_axial_stiffness,
    )

    config = dict(config or {})
    positions = _positions(fem, nodes)
    ribs = _ribs(fem, positions)

    # -- which wing node each bridle attachment belongs to -----------------
    attachment_rib: dict[int, tuple[int, float]] = {}
    wing_ids = {int(row[0]) for row in fem["wing_particles"]["data"]}
    for r, rib in enumerate(ribs):
        chain = [int(n) for n in rib["node_indices"]]
        le, te = positions[chain[0]], positions[chain[-1]]
        for node in chain:
            attachment_rib[node] = (r, _chord_fraction(positions[node], le, te))
    for node in wing_ids - set(attachment_rib):
        # a wing node off the strut ribs (a leading-edge run node at the tip):
        # the rib nearest in span, on its leading-edge side when it is a LE node
        r = int(np.argmin([abs(positions[node][1] - positions[int(rib["node_indices"][0])][1])
                           for rib in ribs]))
        chain = [int(n) for n in ribs[r]["node_indices"]]
        attachment_rib[node] = (
            r, _chord_fraction(positions[node], positions[chain[0]], positions[chain[-1]])
        )

    elements, headers, keep = reduce_bridle(fem, positions, attachment_rib)

    n_ribs = len(ribs)
    missing = [(r, s) for r in range(n_ribs) for s in ("front", "rear") if (r, s) not in keep]
    if missing:
        raise ValueError(f"ribs without a bridle attachment: {missing}")

    # -- renumber: KCU 0, wing 1..2R, bridle after ------------------------
    new_id = {0: 0}
    for r in range(n_ribs):
        new_id[keep[(r, "front")]] = 2 * r + 1
        new_id[keep[(r, "rear")]] = 2 * r + 2
    next_id = 2 * n_ribs + 1
    for element in elements:
        for node in element["nodes"]:
            if node not in new_id:
                new_id[node] = next_id
                next_id += 1
    old_of = {new: old for old, new in new_id.items()}

    # -- wing members ---------------------------------------------------------
    billow = resolve_config(config.get("structural_billow"))
    et = float(billow.get("canopy_stiffness", _DEFAULT_CANOPY_STIFFNESS))
    nu = float(billow.get("canopy_poisson_ratio", _DEFAULT_CANOPY_POISSON))
    pressure = float(fem.get("pressure", 0.3))
    front = [keep[(r, "front")] for r in range(n_ribs)]
    rear = [keep[(r, "rear")] for r in range(n_ribs)]

    def dist(a, b):
        return float(np.linalg.norm(positions[a] - positions[b]))

    def strut_diameter(rib):
        return 0.5 * (float(rib["strut_diam_le"]) + float(rib["strut_diam_te"]))

    ea_strut = tube_axial_stiffness([strut_diameter(rib) for rib in ribs], pressure, config)
    ea_le = tube_axial_stiffness([float(rib["le_diameter"]) for rib in ribs], pressure, config)

    wing_conn, wing_elem = [], []

    def member(name, a, b, k, linktype):
        length = dist(a, b)
        wing_conn.append([name, new_id[a], new_id[b]])
        wing_elem.append([name, round(length, 6), float(k), 0, 0.0, linktype])

    for r in range(n_ribs):
        member(f"strut_{r + 1}", front[r], rear[r], ea_strut[r] / dist(front[r], rear[r]), "default")
    for r in range(n_ribs - 1):
        f0, f1, r0, r1 = front[r], front[r + 1], rear[r], rear[r + 1]
        span = 0.5 * (dist(f0, f1) + dist(r0, r1))
        chord = 0.5 * (dist(f0, r0) + dist(f1, r1))
        shear = et / (2.0 * (1.0 + nu))
        ea_bay = 0.5 * (ea_le[r] + ea_le[r + 1])
        member(f"le_{r + 1}", f0, f1, ea_bay / dist(f0, f1), "default")
        member(f"te_{r + 1}", r0, r1, et * 0.5 * chord / dist(r0, r1), "noncompressive")
        for tag, a, b in (("a", f0, r1), ("b", r0, f1)):
            d = dist(a, b)
            member(f"dia_{r + 1}{tag}", a, b, shear * d**2 / (2.0 * span * chord), "noncompressive")

    # -- bridle tables -----------------------------------------------------
    bridle_lines, bridle_conn, seen = [], [], {}
    for element in elements:
        row = [element.get(h) for h in headers]
        row[0] = element["name"]
        row[headers.index("l0")] = float(element["l0"])
        key = (element["name"], row[headers.index("l0")])
        name = element["name"]
        if name in seen and seen[name] != key:
            name = f"{name}_{len(seen)}"
            row[0] = name
        seen.setdefault(name, key)
        if name not in {line[0] for line in bridle_lines}:
            bridle_lines.append(row)
        bridle_conn.append([name, *[new_id[n] for n in element["nodes"]]])

    reduced = {
        "bridle_point_node": [float(v) for v in positions[0]],
        "pulley_mass": fem.get("pulley_mass", 0.0),
        "dyneema": copy.deepcopy(fem["dyneema"]),
        "fixed_point_indices": [0],
        "wing_particles": {
            "headers": ["id", "x", "y", "z"],
            "data": [[n, *map(float, positions[old_of[n]])] for n in range(1, 2 * n_ribs + 1)],
        },
        "wing_connections": {"headers": ["name", "ci", "cj"], "data": wing_conn},
        "wing_elements": {
            "headers": ["name", "l0", "k", "c", "m", "linktype"],
            "data": wing_elem,
        },
        "bridle_particles": {
            "headers": ["id", "x", "y", "z"],
            "data": [[n, *map(float, positions[old_of[n]])] for n in range(2 * n_ribs + 1, next_id)],
        },
        "bridle_connections": {"headers": ["name", "ci", "cj", "ck"], "data": bridle_conn},
        "bridle_lines": {"headers": headers, "data": bridle_lines},
    }

    # -- masses: full-model wing mass lumped onto the reduced wing nodes ------
    wing_new = list(range(1, 2 * n_ribs + 1))
    target = np.zeros(len(wing_new))
    bridle_mass = _reduced_bridle_mass(reduced)
    if masses is not None:
        full_nodes = np.asarray(nodes if nodes is not None else [positions[i] for i in sorted(positions)])
        masses = np.asarray(masses, dtype=float)
        anchors = np.array([positions[old_of[n]] for n in wing_new])
        for node, mass in enumerate(masses[1:], start=1):  # node 0 is the KCU
            nearest = int(np.argmin(np.linalg.norm(anchors - full_nodes[node], axis=1)))
            target[nearest] += mass
        wing_total = float(masses[1:].sum()) - bridle_mass
    else:
        wing_total = float(fem.get("mass_without_bridles", 0.0))
        target[:] = 1.0
    target *= wing_total / target.sum()
    incidence = np.zeros((len(wing_new), len(wing_elem)))
    for e, (_, ci, cj) in enumerate(wing_conn):
        incidence[ci - 1, e] += 0.5
        incidence[cj - 1, e] += 0.5
    element_mass, _ = nnls(incidence, target)
    for row, m in zip(wing_elem, element_mass):
        row[4] = float(m)
    reduced["reduction"] = {
        "source": "struc_geometry_FEM_full (wireframe.reduce_fem)",
        "wing_mass": float(element_mass.sum()),
        "bridle_mass": bridle_mass,
        "original_node_ids": {int(n): int(o) for n, o in old_of.items()},
    }
    return reduced


def _reduced_bridle_mass(reduced) -> float:
    """Bridle mass the PSM reader will keep: lines (rho A l0) + pulleys.

    The reader puts half of each line's mass on either end and then OVERWRITES
    node 0 with the KCU mass, so the half of a line landing on the KCU is
    dropped -- by the full-geometry reader too, so leaving it out here keeps
    the two totals equal.
    """
    headers = reduced["bridle_lines"]["headers"]
    lines = {row[0]: dict(zip(headers, row)) for row in reduced["bridle_lines"]["data"]}
    total = 0.0
    for row in reduced["bridle_connections"]["data"]:
        line = lines[row[0]]
        area = np.pi * (float(line["d"]) / 2.0) ** 2
        mass = reduced[line["material"]]["density"] * area * float(line["l0"])
        total += 0.5 * mass if 0 in (int(row[1]), int(row[2])) else mass
        if len(row) == 4:
            total += float(reduced["pulley_mass"])
    return float(total)

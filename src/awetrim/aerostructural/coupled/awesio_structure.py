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

"""The structural geometry as an awesIO structure document.

awesIO's ``structure_schema.yml`` wants the model FULLY RESOLVED: every node
and every element written out, nothing left for a reader to generate. Our
geometry YAML is an authoring format -- the reader pads the struts, projects
their nodes onto the strut line, generates the canopy chord nodes and spreads
``mass_without_bridles`` over the tubes -- so the document is written from the
reader's OUTPUT (:func:`read_struc_geometry_yaml.main`), never from the YAML.

How our model maps onto the schema:

- every node is a ``points`` row named by its reader index ("0" is the KCU);
- every node a tube touches is also a rigid ``bodies`` row (the tube's end has
  a rotation, which a point has not), and its point rides on it. The body
  frames are the wing's: Billow builds the tube frames itself;
- a straight bridle line is a ``segments`` row; a pulley line is two segments
  and a ``pulleys`` row. awesIO stores each ARM's rest length (their sum is the
  rope), split here in proportion to the arm lengths as the reader does;
- strut and leading-edge beam elements are ``tubes`` rows, one diameter each
  (the strut taper is the chain of elements), pressure in Pa, all of material
  ``breukels2011`` with Billow's constants;
- the canopy is ``canopy_faces``: one quadrilateral per pair of adjacent
  chordwise sections, which are the ``spanwise_stations``. The spring net the
  wireframe lays over it, and Billow's triangulation, are readers' models;
- masses follow awesIO's rule: a reader derives the lines (half of each
  segment to its ends, from ``diameter``, ``rest_length`` and density) and the
  fabric (an equal share of area x areal density to each corner), and the
  document states the rest as extra mass -- the KCU and the pulleys on their
  points, the tubes on their bodies. The totals are the reader's; how the
  line and fabric mass is shared among nodes is awesIO's, which differs from
  the reader's by grams (a pulley line's quarter-half-quarter, and canopy
  quads only between non-strut sections).

What awesIO leaves to a tool is carried as columns of our own on ``segments``:
``line`` (the ``bridle_lines`` row, which also marks the depower and steering
tapes), ``linktype`` (what the line does in compression) and ``tape_width``
(the drag width of a flat tape; ``diameter`` stays the area-equivalent one), and
as the top-level ``awetrim`` block (the KCU point).
"""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import numpy as np
import yaml

from awetrim.aerostructural.coupled import read_struc_geometry_yaml

AWESIO_VERSION = "0.1.0"
SCHEMA = "structure_schema.yml"
WING = "wing"
CANOPY_MATERIAL = "canopy"
TUBE_MATERIAL = "inflatable_tube"
BAR_TO_PA = 1e5
#: Billow's canopy E*t [N/m] (``structural_billow.DEFAULTS["canopy_stiffness"]``).
CANOPY_MEMBRANE_STIFFNESS = 4.93e5


class _Row(list):
    """A list YAML writes in flow style: one table row on one line."""


def _row_representer(dumper, data):
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)


class _Dumper(yaml.SafeDumper):
    pass


_Dumper.add_representer(_Row, _row_representer)


def _table(headers, units, rows):
    return {
        "headers": _Row(headers),
        "units": _Row(units),
        "data": [_Row(row) for row in rows],
    }


def _vec(v):
    return _Row(float(x) for x in v)


def _name(index) -> str:
    return str(int(index))


def connectivity_preimage(document: dict) -> str:
    """The ASCII string ``metadata.connectivity_sha`` hashes (schema text)."""
    points = {row[0]: i + 1 for i, row in enumerate(document["points"]["data"])}
    bodies = {
        row[0]: i + 1 for i, row in enumerate(document.get("bodies", {}).get("data", []))
    }
    parts = [f"{len(points)};"]
    for row in document["segments"]["data"]:
        parts.append(",".join(str(points[p]) for p in row[1]) + ";")
    parts.append(f"{len(bodies)};")
    for row in document.get("tubes", {}).get("data", []):
        parts.append(",".join(str(bodies[b]) for b in row[1]) + ";")
    faces = document.get("canopy_faces", {}).get("data", [])
    parts.append(f"{len(faces)};")
    for row in faces:
        parts.append(",".join(str(points[p]) for p in row[2]) + ";")
    return "".join(parts)


def connectivity_sha(document: dict) -> str:
    return hashlib.sha256(connectivity_preimage(document).encode("ascii")).hexdigest()


def _quad_area(corners) -> float:
    """awesIO's face area: half the norm of the diagonals' cross product."""
    if len(corners) == 3:
        return 0.5 * float(np.linalg.norm(np.cross(corners[1] - corners[0], corners[2] - corners[0])))
    return 0.5 * float(np.linalg.norm(np.cross(corners[2] - corners[0], corners[3] - corners[1])))


def _tube_material():
    from billow.elements.inflatable import (
        BENDING_COEFFICIENTS,
        COLLAPSE_COEFFICIENTS,
        TORSION_COEFFICIENTS,
    )

    constants = {**BENDING_COEFFICIENTS, **COLLAPSE_COEFFICIENTS, **TORSION_COEFFICIENTS}
    return {key: float(constants[key]) for key in sorted(constants, key=lambda k: int(k[1:]))}


def to_awesio(
    struc_geometry: dict,
    *,
    kcu_mass: float = 0.0,
    strut_padding: str = "bisect_longest",
    height_kcu: float = 0.0,
    name: str = "kite",
    description: str = "",
    note: str = "",
) -> dict:
    """The awesIO structure document of ``struc_geometry`` as the reader resolves it.

    Args:
        struc_geometry: A loaded file in the retired ``struc_geometry_FEM_full`` format (not modified).
        kcu_mass: Extra mass on the KCU point [kg]. ``system.yaml`` is its
            single source (``utils.system_config.get_kcu_mass``); 0 leaves it
            out of the document.
        strut_padding: The reader's strut padding, which fixes the mesh.
        height_kcu: Height of the KCU above the tether exit point [m]. The
            schema places the kite straight above it, so positions are the
            wing's KA coordinates shifted up by this.
        name, description, note: ``metadata`` strings.
    """
    geometry = copy.deepcopy(struc_geometry)
    (
        struc_nodes,
        m_arr,
        _le,
        _te,
        _power_tape_index,
        _steering_tape_indices,
        _pulley_node_indices,
        canopy_sections,
        strut_sections,
        _simplified,
        kite_connectivity,
        bridle_connectivity,
        bridle_diameter,
        l0_arr,
        k_arr,
        c_arr,
        linktype_arr,
        _pulley_line_indices,
        _pulley_dict,
    ) = read_struc_geometry_yaml.main(
        geometry, config={"kcu": {"mass": float(kcu_mass)}, "strut_padding": strut_padding}
    )
    struc_nodes = np.asarray(struc_nodes, dtype=float)
    origin = np.array([0.0, 0.0, float(height_kcu)])
    n_nodes = len(struc_nodes)

    lines = {
        row[0]: dict(zip(geometry["bridle_lines"]["headers"][1:], row[1:]))
        for row in geometry["bridle_lines"]["data"]
    }
    materials = {line["material"] for line in lines.values()}
    for material in sorted(materials):
        if float(geometry[material].get("damping_per_stiffness", 0.0)) != 0.0:
            raise ValueError(
                f"material {material!r} has damping, which awesIO's linear law "
                "cannot carry"
            )

    # -- tubes: strut elements in strut_tubes order, then the leading edge ----
    tube_rows = []
    element = 0
    for row in geometry["strut_tubes"]["data"]:
        strut_name, node_indices = row[0], row[-1]
        for k, (a, b) in enumerate(zip(node_indices[:-1], node_indices[1:])):
            assert linktype_arr[element] == "inflatable_beam"
            assert list(kite_connectivity[element]) == [a, b]
            tube_rows.append((f"{strut_name}_{k + 1}", a, b, k_arr[element], c_arr[element]))
            element += 1
    for le_name, a, b, _diameter in geometry["leading_edge_tubes"]["data"]:
        assert linktype_arr[element] == "inflatable_beam"
        tube_rows.append((le_name, a, b, k_arr[element], c_arr[element]))
        element += 1
    # The canopy spring net follows; the bridle starts after it.
    first_bridle = len(kite_connectivity) - len(bridle_connectivity)

    tube_nodes = sorted({n for _, a, b, _, _ in tube_rows for n in (a, b)})
    body_of = {n: f"b{n}" for n in tube_nodes}

    # -- segments and pulleys, walking the connections as the reader does ----
    segment_rows = []
    pulley_rows = []
    element = first_bridle
    for conn_row, conn in enumerate(geometry["bridle_connections"]["data"]):
        line_name, nodes = conn[0], [int(n) for n in conn[1:]]
        line = lines[line_name]
        width = line.get("w")
        extra = [line_name, line["linktype"], None if width is None else float(width)]
        if len(nodes) == 2:
            ci, cj = nodes
            assert list(bridle_connectivity[element - first_bridle]) == [ci, cj]
            segment_rows.append(
                [f"{line_name}_{conn_row}", _Row([_name(ci), _name(cj)]),
                 float(l0_arr[element]), float(line["d"]), line["material"], *extra]
            )
            element += 1
            continue
        ci, cj, ck = nodes
        rope = float(l0_arr[element])
        arm_a = np.linalg.norm(struc_nodes[ci] - struc_nodes[cj])
        arm_b = np.linalg.norm(struc_nodes[cj] - struc_nodes[ck])
        names = (f"{line_name}_{conn_row}_a", f"{line_name}_{conn_row}_b")
        for seg_name, (p, q), length in zip(
            names, ((ci, cj), (cj, ck)), (rope * arm_a / (arm_a + arm_b), rope * arm_b / (arm_a + arm_b))
        ):
            segment_rows.append(
                [seg_name, _Row([_name(p), _name(q)]), float(length), float(line["d"]),
                 line["material"], *extra]
            )
        pulley_rows.append([f"{line_name}_{conn_row}", _Row(names)])
        element += 2
    assert element == len(kite_connectivity)

    # -- stations and faces ----------------------------------------------------
    strut_name_of = {row[-1][0]: row[0] for row in geometry["strut_tubes"]["data"]}
    # The leading edge's diameter AT each strut, as the authoring file states
    # it: the tube elements do not determine it (the LEI-V3's is up to 7 mm
    # above the elements meeting the strut), and the wireframe reduction sizes
    # its leading-edge members with it.
    strut_headers = geometry["strut_tubes"]["headers"]
    le_diameter_of = {
        row[-1][0]: float(dict(zip(strut_headers, row))["le_diameter"])
        for row in geometry["strut_tubes"]["data"]
    }
    sections = sorted(list(canopy_sections) + list(strut_sections), key=lambda s: s[0])
    station_rows = []
    for k, section in enumerate(sections):
        station = strut_name_of.get(section[0], f"section_{k + 1}")
        station_rows.append(
            [station, WING, _Row(_name(n) for n in section), le_diameter_of.get(section[0])]
        )
    face_rows = []
    for k, (s1, s2) in enumerate(zip(sections[:-1], sections[1:])):
        for j in range(len(s1) - 1):
            corners = [s1[j], s1[j + 1], s2[j + 1], s2[j]]
            face_rows.append([f"face_{k + 1}_{j + 1}", WING, _Row(_name(n) for n in corners)])

    # -- extra masses ----------------------------------------------------------
    # awesIO derives the line and fabric mass itself (half of each segment to
    # its ends, an equal share of each face to its corners), so the document
    # states only what it cannot derive: the KCU and the pulleys on their
    # points, and the tubes -- mass_without_bridles less the canopy, spread by
    # element length as the reader spreads it -- on the tube bodies.
    areal_density = float(geometry["canopy_density"]) / 1000.0
    mass_canopy = sum(
        _quad_area(struc_nodes[[int(p) for p in row[2]]]) * areal_density for row in face_rows
    )
    mass_tubes = float(geometry["mass_without_bridles"]) - mass_canopy
    if mass_tubes < 0.0:
        raise ValueError(
            f"the canopy ({mass_canopy:.3f} kg) outweighs mass_without_bridles "
            f"({geometry['mass_without_bridles']} kg)"
        )
    tube_lengths = np.array(
        [np.linalg.norm(struc_nodes[a] - struc_nodes[b]) for _, a, b, _, _ in tube_rows]
    )
    body_mass = dict.fromkeys(tube_nodes, 0.0)
    for (_, a, b, _, _), length in zip(tube_rows, tube_lengths):
        share = mass_tubes * length / tube_lengths.sum()
        body_mass[a] += share / 2.0
        body_mass[b] += share / 2.0
    point_mass = np.zeros(n_nodes)
    point_mass[0] += float(kcu_mass)
    for conn in geometry["bridle_connections"]["data"]:
        if len(conn) == 4:
            point_mass[int(conn[2])] += float(geometry["pulley_mass"])

    fixed = {int(i) for i in geometry.get("fixed_point_indices", [0])}
    point_rows = [
        [_name(i), i in fixed, None if i in fixed else body_of.get(i),
         _vec(struc_nodes[i] + origin), float(point_mass[i])]
        for i in range(n_nodes)
    ]
    identity = _Row([1.0, 0.0, 0.0, 0.0])
    body_rows = [
        [body_of[n], _vec(struc_nodes[n] + origin), identity, float(body_mass[n])]
        for n in tube_nodes
    ]
    tube_table = [
        [tube_name, _Row([body_of[a], body_of[b]]), float(d), float(p_bar) * BAR_TO_PA, TUBE_MATERIAL]
        for tube_name, a, b, d, p_bar in tube_rows
    ]

    material_rows = [
        [m, "linear", {"density": float(geometry[m]["density"]),
                       "youngs_modulus": float(geometry[m]["youngs_modulus"])}]
        for m in sorted(materials)
    ]
    material_rows.append([CANOPY_MATERIAL, "membrane",
                          {"areal_density": areal_density,
                           "membrane_stiffness": CANOPY_MEMBRANE_STIFFNESS}])
    material_rows.append([TUBE_MATERIAL, "breukels2011", _tube_material()])

    document = {
        "metadata": {
            "name": name,
            "description": description,
            "note": note,
            "awesIO_version": AWESIO_VERSION,
            "schema": SCHEMA,
            "n_points": n_nodes,
            "connectivity_sha": "",
        },
        "materials": _table(["name", "law", "parameters"], ["-", "-", "-"], material_rows),
        "points": _table(
            ["name", "fixed", "body", "pos_ENU", "extra_mass"], ["-", "-", "-", "m", "kg"], point_rows
        ),
        "segments": _table(
            ["name", "points", "rest_length", "diameter", "material", "line", "linktype", "tape_width"],
            ["-", "-", "m", "m", "-", "-", "-", "m"],
            segment_rows,
        ),
        "spanwise_stations": _table(
            ["name", "wing", "points", "le_diameter"], ["-", "-", "-", "m"], station_rows
        ),
        "pulleys": _table(["name", "segments"], ["-", "-"], pulley_rows),
        "wings": _table(
            ["name", "pos_ENU", "Q_KA2ENU", "canopy_material"], ["-", "m", "-", "-"],
            [[WING, _vec(struc_nodes[0] + origin), identity, CANOPY_MATERIAL]],
        ),
        "canopy_faces": _table(["name", "wing", "points"], ["-", "-", "-"], face_rows),
        "bodies": _table(
            ["name", "pos_ENU", "Q_KA2ENU", "extra_mass"], ["-", "m", "-", "kg"], body_rows
        ),
        "tubes": _table(
            ["name", "bodies", "diameter", "pressure", "material"], ["-", "-", "m", "Pa", "-"], tube_table
        ),
        "awetrim": {
            "kcu_point": _name(0),
            "pulley_mass": float(geometry["pulley_mass"]),
            "strut_padding": strut_padding,
        },
    }
    document["metadata"]["connectivity_sha"] = connectivity_sha(document)
    return document


def nodal_masses(document: dict) -> np.ndarray:
    """Mass of every point, by awesIO's rule, in ``points`` row order [kg].

    The point's extra mass, half of each segment it ends (density x
    pi d^2/4 x rest_length), an equal share of each canopy face it is a corner
    of (area x the wing's areal density) and, for a point riding on a body, that
    body's extra mass. Every body here carries exactly one point, at its origin.
    """
    points = _rows(document["points"])
    index = {p["name"]: i for i, p in enumerate(points)}
    pos = np.array([p["pos_ENU"] for p in points], dtype=float)
    mass = np.array([float(p.get("extra_mass") or 0.0) for p in points])
    materials = {m["name"]: m for m in _rows(document.get("materials"))}
    for seg in _rows(document["segments"]):
        density = float(materials[seg["material"]]["parameters"]["density"])
        m = density * np.pi * (float(seg["diameter"]) / 2.0) ** 2 * float(seg["rest_length"])
        for p in seg["points"]:
            mass[index[p]] += m / 2.0
    canopy = {w["name"]: w.get("canopy_material") for w in _rows(document.get("wings"))}
    for face in _rows(document.get("canopy_faces")):
        material = canopy.get(face["wing"])
        if material is None:
            continue
        corners = [index[p] for p in face["points"]]
        areal = float(materials[material]["parameters"]["areal_density"])
        for c in corners:
            mass[c] += _quad_area(pos[corners]) * areal / len(corners)
    riders = {}
    for i, p in enumerate(points):
        if p["body"] is not None:
            riders.setdefault(p["body"], []).append(i)
    for body in _rows(document.get("bodies")):
        on_body = riders.get(body["name"], [])
        if len(on_body) != 1:
            raise ValueError(f"body {body['name']!r} carries {len(on_body)} points, not one")
        mass[on_body[0]] += float(body.get("extra_mass") or 0.0)
    return mass


#: Stiffness [N/m] of the spring net the reader lays over the canopy faces.
CANOPY_SPRING_STIFFNESS = 5000.0


def from_awesio(document: dict, kcu_mass: float = 0.0) -> tuple:
    """The resolved model of an awesIO document, as :func:`read_struc_geometry_yaml.main`
    returns it, so every consumer of that tuple reads awesIO unchanged.

    The document must be one :func:`to_awesio` writes: one wing whose stations
    are its chordwise sections from +y to -y, leading edge first; tubes on
    one-point bodies; bridle lines grouped by the ``line`` column. Points are
    indexed in row order. The element order is the reader's: strut tubes,
    leading-edge tubes, the canopy spring net, then the bridle connection by
    connection. ``simplified_bridle_points`` (the strut nodes before the
    reader projected them) is empty: a resolved document holds only the
    projected ones.
    """
    points = _rows(document["points"])
    index = {p["name"]: i for i, p in enumerate(points)}
    struc_nodes = np.array([p["pos_ENU"] for p in points], dtype=float)
    wing = _rows(document["wings"])
    if len(wing) != 1:
        raise ValueError(f"expected one wing, found {len(wing)}")
    struc_nodes = struc_nodes - np.asarray(wing[0]["pos_ENU"], dtype=float)
    if not np.allclose(wing[0]["Q_KA2ENU"], [1.0, 0.0, 0.0, 0.0]):
        raise ValueError("only a wing at the schema's zenith pose (Q_KA2ENU = 1) is read")
    m_arr = nodal_masses(document)
    # system.yaml is the KCU mass's single source; the document carries none.
    m_arr[index[(document.get("awetrim") or {}).get("kcu_point", "0")]] += float(kcu_mass)
    point_on = {p["body"]: index[p["name"]] for p in points if p["body"] is not None}

    materials = {m["name"]: m for m in _rows(document.get("materials"))}
    stations = [[index[p] for p in s["points"]] for s in _rows(document["spanwise_stations"])]
    tubes = _rows(document.get("tubes"))
    tube_pairs = {
        frozenset(point_on[b] for b in t["bodies"]) for t in tubes
    }

    def is_strut(section):
        return all(frozenset(pair) in tube_pairs for pair in zip(section[:-1], section[1:]))

    # Strut order is the order the tubes list them in.
    first_tube = {}
    for k, t in enumerate(tubes):
        for b in t["bodies"]:
            first_tube.setdefault(point_on[b], k)
    strut_sections = sorted(
        (s for s in stations if is_strut(s)), key=lambda s: first_tube[s[0]]
    )
    canopy_sections = [s for s in stations if not is_strut(s)]
    le_indices = np.array([s[0] for s in stations])
    te_indices = np.array([s[-1] for s in stations])

    kite_connectivity, l0_arr, k_arr, c_arr, linktype_arr = [], [], [], [], []

    def add(a, b, l0, k, c, linktype):
        kite_connectivity.append([a, b])
        l0_arr.append(l0)
        k_arr.append(k)
        c_arr.append(c)
        linktype_arr.append(linktype)

    for t in tubes:
        a, b = (point_on[x] for x in t["bodies"])
        add(a, b, float(np.linalg.norm(struc_nodes[a] - struc_nodes[b])),
            float(t["diameter"]), float(t["pressure"]) / BAR_TO_PA, "inflatable_beam")

    def spring(a, b):
        add(a, b, float(np.linalg.norm(struc_nodes[a] - struc_nodes[b])),
            CANOPY_SPRING_STIFFNESS, 0, "noncompressive")

    for section in canopy_sections:
        for a, b in zip(section[:-1], section[1:]):
            spring(a, b)
    for s1, s2 in zip(stations[:-1], stations[1:]):
        for i in range(1, min(len(s1), len(s2))):
            spring(s1[i], s2[i])
            spring(s1[i], s2[i - 1])
            spring(s1[i - 1], s2[i])

    segments = {s["name"]: s for s in _rows(document["segments"])}
    arm_of = {}
    for pulley in _rows(document.get("pulleys")):
        a, b = pulley["segments"]
        arm_of[a], arm_of[b] = (pulley, 0), (pulley, 1)

    bridle_connectivity, bridle_diameter = [], []
    pulley_node_indices, pulley_line_indices = [], []
    pulley_pairs = {}
    power_tape_index, steering_tape_indices = 0, []

    def line_stiffness(seg, l0):
        params = materials[seg["material"]]["parameters"]
        return float(params["youngs_modulus"]) * np.pi * (float(seg["diameter"]) / 2.0) ** 2 / l0

    def add_line(a, b, l0, seg):
        bridle_connectivity.append([a, b])
        bridle_diameter.append(float(seg["diameter"]))
        add(a, b, l0, line_stiffness(seg, l0), 0.0, seg["linktype"])

    for seg in _rows(document["segments"]):
        if seg["name"] in arm_of:
            pulley, arm = arm_of[seg["name"]]
            if arm == 1:
                continue
            other = segments[pulley["segments"][1]]
            ci, cj = (index[p] for p in seg["points"])
            cj2, ck = (index[p] for p in other["points"])
            if cj2 != cj:
                raise ValueError(f"pulley {pulley['name']!r}: its arms do not meet")
            l_a, l_b = float(seg["rest_length"]), float(other["rest_length"])
            rope = l_a + l_b
            first = len(kite_connectivity)
            add_line(ci, cj, rope, seg)
            add_line(cj, ck, rope, seg)
            pulley_node_indices.append(cj)
            pulley_line_indices += [first, first + 1]
            pulley_pairs[str(first)] = np.array([cj, ck, l_b, l_a, ci])
            pulley_pairs[str(first + 1)] = np.array([cj, ci, l_a, l_b, ci])
        else:
            a, b = (index[p] for p in seg["points"])
            add_line(a, b, float(seg["rest_length"]), seg)
        last = len(kite_connectivity) - 1
        if seg["line"] == "depower_tape":
            power_tape_index = last
        elif seg["line"] == "steering_tape":
            steering_tape_indices.append(last)

    return (
        struc_nodes,
        m_arr,
        le_indices,
        te_indices,
        power_tape_index,
        steering_tape_indices,
        pulley_node_indices,
        canopy_sections,
        strut_sections,
        np.empty((0, 3), dtype=object),
        np.array(kite_connectivity),
        np.array(bridle_connectivity),
        np.array(bridle_diameter),
        np.array(l0_arr),
        np.array(k_arr),
        np.array(c_arr),
        np.array(linktype_arr),
        pulley_line_indices,
        pulley_pairs,
    )


def _rows(table: dict | None) -> list[dict]:
    """A headers/units/data table as one dict per row, keyed by header."""
    if not table:
        return []
    return [dict(zip(table["headers"], row)) for row in table["data"]]


#: Material of the wireframe's wing members: diameter 0, so an awesIO reader
#: derives no mass from them; their stiffness and mass are tool columns.
WIREFRAME_MEMBER_MATERIAL = "wireframe_member"


def _is_awesio(document: dict) -> bool:
    return (document.get("metadata") or {}).get("schema") == SCHEMA


def load_structure(source, *, convert_legacy: bool = True) -> "Structure":
    """The :class:`Structure` of an awesIO file (path or loaded dict).

    A file in the retired AWETrim formats -- a full FEM geometry
    (``strut_tubes``) or a PSM-format wireframe (``wing_elements``) -- is
    converted on the fly with a DeprecationWarning; convert it once with
    ``python -m awetrim.aerostructural.coupled.awesio_structure`` instead.
    """
    import warnings

    if isinstance(source, Structure):
        return source
    if isinstance(source, dict):
        document, label = source, "<dict>"
    else:
        label = str(source)
        with open(source, "r", encoding="utf-8") as handle:
            document = yaml.safe_load(handle)
    if _is_awesio(document):
        return Structure(document)
    if not convert_legacy:
        raise ValueError(f"{label} is not an awesIO structure document")
    warnings.warn(
        f"{label} is in a retired AWETrim structure format; converting it to "
        "awesIO in memory. Convert the file once with "
        "`python -m awetrim.aerostructural.coupled.awesio_structure`.",
        DeprecationWarning,
        stacklevel=2,
    )
    if "strut_tubes" in document:
        return Structure(to_awesio(document))
    if "wing_elements" in document:
        return Structure(wireframe_to_awesio(document))
    raise ValueError(f"{label}: neither awesIO nor a known AWETrim structure format")


class Structure:
    """Read-only view of an awesIO structure document, in AWETrim's terms.

    Nodes are indexed in ``points`` row order. Positions are in the wing's KA
    frame (the document's ENU positions less the wing origin), which is the
    frame every AWETrim model works in.
    """

    def __init__(self, document: dict):
        if not _is_awesio(document):
            raise ValueError("not an awesIO structure document")
        self.document = document
        points = _rows(document["points"])
        self.names = [p["name"] for p in points]
        self.index = {name: i for i, name in enumerate(self.names)}
        wings = _rows(document.get("wings"))
        if len(wings) != 1:
            raise ValueError(f"expected one wing, found {len(wings)}")
        self.origin = np.asarray(wings[0]["pos_ENU"], dtype=float)
        self.positions = np.array([p["pos_ENU"] for p in points], dtype=float) - self.origin
        self.fixed = [i for i, p in enumerate(points) if p["fixed"]]
        tool = document.get("awetrim") or {}
        self.kcu = self.index[tool.get("kcu_point", "0")]
        self.pulley_mass = float(tool.get("pulley_mass", 0.0))
        self.materials = {m["name"]: m for m in _rows(document.get("materials"))}
        self.stations = [
            [self.index[p] for p in s["points"]] for s in _rows(document["spanwise_stations"])
        ]
        self.station_names = [s["name"] for s in _rows(document["spanwise_stations"])]
        self._station_le_diameter = [
            s.get("le_diameter") for s in _rows(document["spanwise_stations"])
        ]
        self.segments = _rows(document["segments"])
        self._point_on = {p["body"]: i for i, p in enumerate(points) if p["body"] is not None}

    # -- wing -------------------------------------------------------------
    @property
    def leading_edge(self) -> list[int]:
        """First node of every station, +y to -y."""
        return [s[0] for s in self.stations]

    @property
    def trailing_edge(self) -> list[int]:
        return [s[-1] for s in self.stations]

    @property
    def is_wireframe(self) -> bool:
        """No tubes: a reduced wireframe, whose wing is spring members."""
        return not (self.document.get("tubes") or {}).get("data")

    @property
    def tubes(self) -> list[dict]:
        """``{name, nodes: (a, b), diameter, pressure}`` per tube [m, Pa]."""
        return [
            {
                "name": t["name"],
                "nodes": tuple(self._point_on[b] for b in t["bodies"]),
                "diameter": float(t["diameter"]),
                "pressure": float(t["pressure"]),
            }
            for t in _rows(self.document.get("tubes"))
        ]

    @property
    def pressure_bar(self) -> float:
        pressures = {t["pressure"] for t in self.tubes}
        if len(pressures) != 1:
            raise ValueError(f"tubes at {len(pressures)} different pressures")
        return pressures.pop() / BAR_TO_PA

    def _tube_pairs(self):
        return {frozenset(t["nodes"]): t for t in self.tubes}

    @property
    def struts(self) -> list[dict]:
        """Stations a tube chain runs along: ``{name, nodes, diameters, le_diameter}``, +y first.

        ``diameters`` are per element, leading edge first. ``le_diameter`` is
        the leading edge's diameter at the strut (AWETrim's own station column),
        None where the file does not state it.
        """
        pairs = self._tube_pairs()
        struts = []
        for name, section, le_diameter in zip(
            self.station_names, self.stations, self._station_le_diameter
        ):
            links = [frozenset(p) for p in zip(section[:-1], section[1:])]
            if links and all(link in pairs for link in links):
                struts.append(
                    {"name": name, "nodes": list(section),
                     "diameters": [pairs[link]["diameter"] for link in links],
                     "le_diameter": None if le_diameter is None else float(le_diameter)}
                )
        return struts

    @property
    def leading_edge_chain(self) -> list[int]:
        """Node path along the leading-edge tube, +y to -y."""
        pairs = self._tube_pairs()
        le = self.leading_edge
        for a, b in zip(le[:-1], le[1:]):
            if frozenset((a, b)) not in pairs:
                raise ValueError(f"no leading-edge tube between nodes {a} and {b}")
        return list(le)

    # -- bridle -----------------------------------------------------------
    @property
    def bridle_segments(self) -> list[dict]:
        """Bridle segments in document order -- the order VSM's line system
        and every reader's element list use (a pulley's two arms adjacent)."""
        return [s for s in self.segments if s.get("line") is not None]

    @property
    def bridle(self) -> list[dict]:
        """Bridle lines connection by connection, a pulley as ONE element.

        ``{name (the line), nodes ([a, b] or [a, sheave, b]), rest_length (the
        whole rope), diameter, tape_width, material, linktype, segments}``.
        """
        arms = {}
        for pulley in _rows(self.document.get("pulleys")):
            first, second = pulley["segments"]
            arms[first], arms[second] = second, None
        by_name = {s["name"]: s for s in self.segments}
        elements = []
        for seg in self.bridle_segments:
            if seg["name"] in arms and arms[seg["name"]] is None:
                continue
            nodes = [self.index[p] for p in seg["points"]]
            rest = float(seg["rest_length"])
            names = [seg["name"]]
            other = arms.get(seg["name"])
            if other is not None:
                second = by_name[other]
                a, b = (self.index[p] for p in second["points"])
                if a != nodes[1]:
                    raise ValueError(f"pulley arms {seg['name']} / {other} do not meet")
                nodes.append(b)
                rest += float(second["rest_length"])
                names.append(other)
            elements.append(
                {
                    "name": seg["line"],
                    "nodes": nodes,
                    "rest_length": rest,
                    "diameter": float(seg["diameter"]),
                    "tape_width": seg.get("tape_width"),
                    "material": seg["material"],
                    "linktype": seg["linktype"],
                    "segments": names,
                }
            )
        return elements

    def bridle_line_system(self, positions=None) -> list[list]:
        """``[p1, p2, diameter]`` per bridle segment, VSM's ``bridle_line_system``."""
        pos = self.positions if positions is None else np.asarray(positions, dtype=float)
        system = []
        for seg in self.bridle_segments:
            a, b = seg["points"]
            system.append(
                [pos[self.index[a]].copy(), pos[self.index[b]].copy(), float(seg["diameter"])]
            )
        return system

    # -- mass and the solver model ------------------------------------------
    def masses(self, kcu_mass: float = 0.0) -> np.ndarray:
        """Nodal masses by awesIO's rule, plus the KCU's from ``system.yaml``."""
        masses = nodal_masses(self.document)
        masses[self.kcu] += float(kcu_mass)
        return masses

    def resolved(self, kcu_mass: float = 0.0) -> tuple:
        """The full model's solver arrays (``read_struc_geometry_yaml.main``'s tuple)."""
        if self.is_wireframe:
            raise ValueError("a wireframe has no full model; use wireframe_dict()")
        return from_awesio(self.document, kcu_mass=kcu_mass)

    def wireframe_dict(self) -> dict:
        """The PSM-format dict ``wireframe.structural_geometry_io`` reads."""
        return wireframe_from_awesio(self.document)

    # -- edited copies ----------------------------------------------------
    def with_positions(self, positions) -> "Structure":
        """A copy with every point (and the body it rides on) moved to ``positions``."""
        positions = np.asarray(positions, dtype=float)
        document = copy.deepcopy(self.document)
        column = document["points"]["headers"].index("pos_ENU")
        for row, pos in zip(document["points"]["data"], positions):
            row[column] = [float(v) for v in pos + self.origin]
        bodies = document.get("bodies") or {}
        if bodies.get("data"):
            column = bodies["headers"].index("pos_ENU")
            for row in bodies["data"]:
                row[column] = [float(v) for v in positions[self._point_on[row[0]]] + self.origin]
        return Structure(document)

    def with_rest_lengths(self, rest_lengths: dict) -> "Structure":
        """A copy with the named segments' rest lengths replaced [m]."""
        document = copy.deepcopy(self.document)
        column = document["segments"]["headers"].index("rest_length")
        for row in document["segments"]["data"]:
            if row[0] in rest_lengths:
                row[column] = float(rest_lengths[row[0]])
        return Structure(document)

    def write(self, path) -> None:
        document = copy.deepcopy(self.document)
        document["metadata"]["connectivity_sha"] = connectivity_sha(document)
        write_awesio(_as_rows(document), path)


def _as_rows(document: dict) -> dict:
    """Mark every table's headers, units and rows for one-line output."""
    for value in document.values():
        if isinstance(value, dict) and {"headers", "units", "data"} <= set(value):
            value["headers"] = _Row(value["headers"])
            value["units"] = _Row(value["units"])
            value["data"] = [_Row(row) for row in value["data"]]
    return document


# -- the wireframe ----------------------------------------------------------
def wireframe_to_awesio(psm: dict, *, name: str = "kite_wireframe", height_kcu: float = 0.0) -> dict:
    """The awesIO document of a PSM-format wireframe (``wireframe.reduce_fem``'s
    output, or a deformed snapshot).

    Ribs (``wing_particles`` ``2r+1`` front, ``2r+2`` rear) are the stations.
    Wing members are segments of material ``wireframe_member`` (diameter 0)
    with their ``stiffness``, ``damping`` and ``mass`` as tool columns; their
    mass, half to each end, and the pulley masses are the points' extra mass,
    so an awesIO reader weighs the wireframe's wing as the PSM reader does.
    """
    from awetrim.aerostructural.utils import bridle_connection_rest_lengths

    positions = {0: np.asarray(psm.get("bridle_point_node", [0, 0, 0]), dtype=float)}
    wing_ids = []
    for key in ("wing_particles", "bridle_particles"):
        for row in psm.get(key, {}).get("data", []):
            positions[int(row[0])] = np.asarray(row[1:4], dtype=float)
            if key == "wing_particles":
                wing_ids.append(int(row[0]))
    n_nodes = max(positions) + 1
    if sorted(positions) != list(range(n_nodes)):
        raise ValueError("wireframe node ids are not contiguous from 0")
    origin = np.array([0.0, 0.0, float(height_kcu)])

    extra = np.zeros(n_nodes)
    member_rows = []
    element_headers = psm["wing_elements"]["headers"]
    elements = {row[0]: dict(zip(element_headers, row)) for row in psm["wing_elements"]["data"]}
    for conn in psm["wing_connections"]["data"]:
        props = elements[conn[0]]
        a, b = int(conn[1]), int(conn[2])
        extra[a] += float(props["m"]) / 2.0
        extra[b] += float(props["m"]) / 2.0
        member_rows.append(
            [conn[0], _Row([_name(a), _name(b)]), float(props["l0"]), 0.0,
             WIREFRAME_MEMBER_MATERIAL, None, props["linktype"], None,
             float(props["k"]), float(props["c"]), float(props["m"])]
        )

    lines = {row[0]: dict(zip(psm["bridle_lines"]["headers"][1:], row[1:]))
             for row in psm["bridle_lines"]["data"]}
    overrides = bridle_connection_rest_lengths(psm)
    materials = set()
    bridle_rows, pulley_rows = [], []
    for conn_row, conn in enumerate(psm["bridle_connections"]["data"]):
        line_name = conn[0]
        nodes = [int(n) for n in conn[1:] if n is not None]
        line = lines[line_name]
        materials.add(line["material"])
        rope = float(overrides.get(conn_row, line["l0"]))
        width = line.get("w")
        tail = [line_name, line["linktype"], None if width in (None, "") else float(width),
                None, None, None]
        if len(nodes) == 2:
            bridle_rows.append(
                [f"{line_name}_{conn_row}", _Row(_name(n) for n in nodes), rope,
                 float(line["d"]), line["material"], *tail]
            )
            continue
        ci, cj, ck = nodes
        arm_a = np.linalg.norm(positions[ci] - positions[cj])
        arm_b = np.linalg.norm(positions[cj] - positions[ck])
        names = (f"{line_name}_{conn_row}_a", f"{line_name}_{conn_row}_b")
        lengths = (rope * arm_a / (arm_a + arm_b), rope * arm_b / (arm_a + arm_b))
        for seg_name, (p, q), length in zip(names, ((ci, cj), (cj, ck)), lengths):
            bridle_rows.append(
                [seg_name, _Row([_name(p), _name(q)]), float(length), float(line["d"]),
                 line["material"], *tail]
            )
        pulley_rows.append([f"{line_name}_{conn_row}", _Row(names)])
        extra[cj] += float(psm.get("pulley_mass", 0.0))

    fixed = {int(i) for i in psm.get("fixed_point_indices", [0])}
    point_rows = [
        [_name(i), i in fixed, None, _vec(positions[i] + origin), float(extra[i])]
        for i in range(n_nodes)
    ]
    wing_ids.sort()
    stations = [
        [f"rib_{r + 1}", WING, _Row([_name(wing_ids[2 * r]), _name(wing_ids[2 * r + 1])])]
        for r in range(len(wing_ids) // 2)
    ]
    material_rows = [
        [m, "linear", {"density": float(psm[m]["density"]),
                       "youngs_modulus": float(psm[m]["youngs_modulus"])}]
        for m in sorted(materials)
    ]
    material_rows.append(
        [WIREFRAME_MEMBER_MATERIAL, "linear", {"density": 0.0, "youngs_modulus": 0.0}]
    )
    tool = {"kcu_point": "0", "pulley_mass": float(psm.get("pulley_mass", 0.0))}
    for key in ("reduction", "billowing"):
        if key in psm:
            tool[key] = copy.deepcopy(psm[key])
    document = {
        "metadata": {
            "name": name, "description": "", "note": "", "awesIO_version": AWESIO_VERSION,
            "schema": SCHEMA, "n_points": n_nodes, "connectivity_sha": "",
        },
        "materials": _table(["name", "law", "parameters"], ["-", "-", "-"], material_rows),
        "points": _table(
            ["name", "fixed", "body", "pos_ENU", "extra_mass"], ["-", "-", "-", "m", "kg"], point_rows
        ),
        "segments": _table(
            ["name", "points", "rest_length", "diameter", "material", "line", "linktype",
             "tape_width", "stiffness", "damping", "mass"],
            ["-", "-", "m", "m", "-", "-", "-", "m", "N/m", "N*s/m", "kg"],
            member_rows + bridle_rows,
        ),
        "spanwise_stations": _table(["name", "wing", "points"], ["-", "-", "-"], stations),
        "pulleys": _table(["name", "segments"], ["-", "-"], pulley_rows),
        "wings": _table(
            ["name", "pos_ENU", "Q_KA2ENU"], ["-", "m", "-"],
            [[WING, _vec(positions[0] + origin), _Row([1.0, 0.0, 0.0, 0.0])]],
        ),
        "awetrim": tool,
    }
    document["metadata"]["connectivity_sha"] = connectivity_sha(document)
    return document


def wireframe_from_awesio(document: dict) -> dict:
    """The PSM-format dict of a wireframe document (inverse of :func:`wireframe_to_awesio`)."""
    structure = Structure(document)
    if not structure.is_wireframe:
        raise ValueError("the document has tubes: it is a full model, not a wireframe")
    pos = structure.positions
    wing_ids = sorted({n for s in structure.stations for n in s})
    if structure.kcu != 0 or wing_ids != list(range(1, len(wing_ids) + 1)):
        raise ValueError("a wireframe numbers the KCU 0 and its wing nodes 1..2R")
    bridle = structure.bridle
    bridle_ids = sorted(
        {n for e in bridle for n in e["nodes"]} - set(wing_ids) - {structure.kcu}
    )
    members = [s for s in structure.segments if s.get("line") is None]
    line_rows, connections, overrides, first_rest = {}, [], [], {}
    for row_index, element in enumerate(bridle):
        name = element["name"]
        if name not in line_rows:
            row = [name, element["rest_length"], element["diameter"], element["material"],
                   element["linktype"]]
            if element.get("tape_width") is not None:
                row.append(float(element["tape_width"]))
            line_rows[name] = row
            first_rest[name] = element["rest_length"]
        elif abs(first_rest[name] - element["rest_length"]) > 1e-12:
            overrides.append([row_index, element["rest_length"]])
        connections.append([name, *element["nodes"]])
    headers = ["name", "l0", "d", "material", "linktype"]
    if any(len(r) == 6 for r in line_rows.values()):
        headers.append("w")
        for r in line_rows.values():
            if len(r) == 5:
                r.append(None)
    materials = {e["material"] for e in bridle}
    psm = {
        "bridle_point_node": [float(v) for v in pos[0]],
        "pulley_mass": structure.pulley_mass,
        **{m: {**structure.materials[m]["parameters"], "damping_per_stiffness": 0.0}
           for m in sorted(materials)},
        "fixed_point_indices": list(structure.fixed),
        "wing_particles": {"headers": ["id", "x", "y", "z"],
                           "data": [[n, *map(float, pos[n])] for n in wing_ids]},
        "wing_connections": {"headers": ["name", "ci", "cj"],
                             "data": [[m["name"], *(structure.index[p] for p in m["points"])]
                                      for m in members]},
        "wing_elements": {"headers": ["name", "l0", "k", "c", "m", "linktype"],
                          "data": [[m["name"], float(m["rest_length"]), float(m["stiffness"]),
                                    float(m["damping"]), float(m["mass"]), m["linktype"]]
                                   for m in members]},
        "bridle_particles": {"headers": ["id", "x", "y", "z"],
                             "data": [[n, *map(float, pos[n])] for n in bridle_ids]},
        "bridle_connections": {"headers": ["name", "ci", "cj", "ck"], "data": connections},
        "bridle_lines": {"headers": headers, "data": list(line_rows.values())},
    }
    if overrides:
        from awetrim.aerostructural.utils import BRIDLE_CONNECTION_REST_LENGTHS

        psm[BRIDLE_CONNECTION_REST_LENGTHS] = {"headers": ["row", "l0"], "data": overrides}
    for key in ("reduction", "billowing"):
        if key in (document.get("awetrim") or {}):
            psm[key] = copy.deepcopy(document["awetrim"][key])
    return psm


def write_awesio(document: dict, path: str | Path) -> None:
    """Write ``document`` as YAML, one table row per line."""
    with open(path, "w", encoding="utf-8") as handle:
        yaml.dump(document, handle, Dumper=_Dumper, sort_keys=False, allow_unicode=True, width=10_000)


def convert_file(source, target, *, name=None, description="", note="") -> Structure:
    """Convert a retired-format structure file (FEM geometry or PSM wireframe) to awesIO."""
    with open(source, "r", encoding="utf-8") as handle:
        legacy = yaml.safe_load(handle)
    if _is_awesio(legacy):
        raise ValueError(f"{source} is already an awesIO document")
    name = name or Path(source).parent.name
    if "strut_tubes" in legacy:
        document = to_awesio(legacy, name=name, description=description, note=note)
    elif "wing_elements" in legacy:
        document = wireframe_to_awesio(legacy, name=name)
        document["metadata"].update(description=description, note=note)
    else:
        raise ValueError(f"{source}: not a known AWETrim structure format")
    write_awesio(document, target)
    return Structure(document)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert an AWETrim structure file (FEM geometry or PSM wireframe) to awesIO."
    )
    parser.add_argument("source")
    parser.add_argument("target")
    parser.add_argument("--name")
    args = parser.parse_args()
    convert_file(args.source, args.target, name=args.name)

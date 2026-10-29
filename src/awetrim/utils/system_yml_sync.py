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

"""Update system.yml fields derived from the structural geometry."""

from pathlib import Path

from ruamel.yaml import YAML

import numpy as np

from awetrim.aerostructural.coupled.awesio_structure import load_structure
from awetrim.aerostructural.wireframe.structural_geometry_io import wing_planform_stats
from awetrim.aerostructural.utils import (
    calculate_cg,
    calculate_inertia,
    compute_kite_aggregate,
)
from awetrim.utils.system_config import get_kite

# Only physically meaningful fields belong in system.yml — counts stay out.
_BRIDLE_COMPUTED_FIELDS = ("total_nominal_line_length", "avg_line_diameter", "mass")
_WING_COMPUTED_FIELDS = (
    "mass",
    "center_of_mass",
    "inertia_tensor",
    "span",
    "projected_surface_area",
    "planform_surface_area",
    "side_projected_area",
)


def _line_mass(structure, element: dict) -> float:
    """Mass of one bridle element (a pulley's whole rope) [kg]: density x area x
    rest length, the rule ``awesio_structure.nodal_masses`` lumps."""
    density = float(structure.materials[element["material"]]["parameters"]["density"])
    return density * np.pi * (float(element["diameter"]) / 2.0) ** 2 * float(
        element["rest_length"]
    )


def compute_wing_stats(structure) -> dict:
    """The canopy-only wing fields of system.yaml from a structure.

    ``structure`` is a full model or a wireframe (a ``Structure``, or anything
    ``load_structure`` accepts). Masses are ``Structure.masses`` with the
    bridle and the KCU weightless (each bridle segment's half-masses and the
    pulley masses taken back off its ends), so what is left is the canopy and
    its tubes, or the wireframe's wing members. CG and inertia (about the CG,
    geometry axes) are in the structure's KA frame; span and areas from the
    stations' leading and trailing edges.
    """
    structure = load_structure(structure)
    nodes = structure.positions
    masses = structure.masses(kcu_mass=0.0)
    for segment in structure.bridle_segments:
        half = 0.5 * _line_mass(structure, segment)
        for point in segment["points"]:
            masses[structure.index[point]] -= half
    for element in structure.bridle:
        if len(element["nodes"]) == 3:
            masses[element["nodes"][1]] -= structure.pulley_mass
    masses[np.abs(masses) < 1e-12] = 0.0
    cg = calculate_cg(nodes, masses)
    inertia = calculate_inertia([(nodes[i], masses[i]) for i in range(len(masses))], cg)
    stats = {
        "mass": round(float(masses.sum()), 4),
        "center_of_mass": [round(float(v), 4) for v in cg],
        "inertia_tensor": [[round(float(v), 4) for v in row] for row in inertia],
    }
    stats.update(
        wing_planform_stats(nodes[structure.leading_edge], nodes[structure.trailing_edge])
    )
    return stats


def compute_bridle_stats(structure) -> dict:
    """Bridle summary statistics of a structure, from ``Structure.bridle``.

    Returns a dict with:
      total_nominal_line_length  — sum of the rest lengths of the unique lines,
                                   the first element of each (m)
      avg_line_diameter          — element-length-weighted average diameter (m)
      bridle_line_count          — number of unique lines
      bridle_particle_count      — bridle nodes, neither wing nor KCU
      bridle_connection_count    — number of elements (a pulley is one)
      pulley_count               — number of pulley elements
      mass                       — total bridle mass: lines + pulleys (kg)
    """
    structure = load_structure(structure)
    elements = structure.bridle
    first_rest: dict[str, float] = {}
    for element in elements:
        first_rest.setdefault(element["name"], float(element["rest_length"]))
    rest = np.array([float(e["rest_length"]) for e in elements])
    diameter = np.array([float(e["diameter"]) for e in elements])
    pulley_count = sum(len(e["nodes"]) == 3 for e in elements)
    mass_lines = sum(_line_mass(structure, e) for e in elements)
    wing = {n for station in structure.stations for n in station}
    bridle_nodes = {n for e in elements for n in e["nodes"]} - wing - {structure.kcu}
    return {
        "total_nominal_line_length": round(sum(first_rest.values()), 5),
        "avg_line_diameter": round(float((diameter * rest).sum() / rest.sum()), 5),
        "bridle_line_count": len(first_rest),
        "bridle_particle_count": len(bridle_nodes),
        "bridle_connection_count": len(elements),
        "pulley_count": int(pulley_count),
        "mass": round(mass_lines + pulley_count * structure.pulley_mass, 4),
    }


def update_from_geometry(
    system_yml_path: Path | str,
    struc_geometry_path: Path | str,
    output_path: Path | str | None = None,
) -> dict:
    """Update kite, wing, and bridle fields in system.yml from a structure file
    (awesIO ``structure.yml``, full model or wireframe; a retired-format file
    is converted by ``load_structure``).

    Reads both files once, computes all derived fields, and writes the result,
    preserving comments and formatting via ruamel.yaml.

    By default the input ``system_yml_path`` is updated in place. Pass
    ``output_path`` to read the canonical system.yml but write the updated copy
    elsewhere (e.g. into a deformed-result case folder), leaving the source file
    untouched.

    Returns a dict with keys 'kite', 'wing', and 'bridle'.
    """
    system_yml_path = Path(system_yml_path)

    ruamel_yaml = YAML()
    ruamel_yaml.preserve_quotes = True
    with open(system_yml_path) as f:
        system_config = ruamel_yaml.load(f)

    structure = load_structure(Path(struc_geometry_path))
    bridle_stats = compute_bridle_stats(structure)
    wing_stats = compute_wing_stats(structure)
    # The kite aggregate is derived, not stored in system.yml (the awesIO schema
    # has no slot for it); computed here only for the return value / logging.
    kite_stats = compute_kite_aggregate(structure, system_config)

    kite_node = get_kite(system_config)

    wing_struct = kite_node["wing"]["structure"]
    for key in _WING_COMPUTED_FIELDS:
        wing_struct[key] = wing_stats[key]

    bridle_struct = kite_node["bridle"]["structure"]
    for key in _BRIDLE_COMPUTED_FIELDS:
        bridle_struct[key] = bridle_stats[key]

    out_path = Path(output_path) if output_path is not None else system_yml_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        ruamel_yaml.dump(system_config, f)

    return {"kite": kite_stats, "wing": wing_stats, "bridle": bridle_stats}

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

"""Update system.yml fields derived from PSS structural geometry."""

from pathlib import Path

import yaml
from ruamel.yaml import YAML

from awetrim.aerostructural.wireframe.structural_geometry_io import (
    compute_bridle_stats_from_pss,
    compute_wing_stats_from_pss,
)
from awetrim.aerostructural.utils import compute_kite_aggregate
from awetrim.utils.system_config import get_kite

# Only physically meaningful fields belong in system.yml — PSS counts stay out.
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


def update_from_geometry(
    system_yml_path: Path | str,
    struc_geometry_path: Path | str,
    output_path: Path | str | None = None,
) -> dict:
    """Update kite, wing, and bridle fields in system.yml from a PSS struc_geometry file.

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

    with open(Path(struc_geometry_path)) as f:
        struc_geometry = yaml.safe_load(f)

    bridle_stats = compute_bridle_stats_from_pss(struc_geometry)
    wing_stats = compute_wing_stats_from_pss(struc_geometry)
    # The kite aggregate is derived, not stored in system.yml (the awesIO schema
    # has no slot for it); computed here only for the return value / logging.
    kite_stats = compute_kite_aggregate(struc_geometry, system_config)

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

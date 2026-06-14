"""Update system.yml fields derived from PSS structural geometry."""

from pathlib import Path

import numpy as np
import yaml
from ruamel.yaml import YAML

from awetrim.aerostructural.pss.structural_geometry_io import (
    compute_bridle_stats_from_pss,
    compute_wing_stats_from_pss,
    main as pss_initialize,
)
from awetrim.aerostructural.utils import calculate_cg, calculate_inertia

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
_KITE_AGGREGATE_FIELDS = ("mass", "center_of_mass", "inertia_tensor")


def _compute_kite_aggregate(struc_geometry, system_config) -> dict:
    """Compute aggregate kite mass, CG, and inertia from the full PSS system."""
    result = pss_initialize(struc_geometry, system_config=system_config)
    struc_nodes, m_arr = result[0], result[1]

    total_mass = float(np.sum(m_arr))
    cg = calculate_cg(struc_nodes, m_arr)
    I = calculate_inertia(
        [(struc_nodes[i], m_arr[i]) for i in range(len(m_arr))],
        desired_point=cg,
    )
    return {
        "mass": round(total_mass, 4),
        "center_of_mass": [round(float(v), 4) for v in cg],
        "inertia_tensor": [[round(float(v), 4) for v in row] for row in I],
    }


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
    kite_stats = _compute_kite_aggregate(struc_geometry, system_config)

    kite_node = system_config["components"]["kite"]

    for key in _KITE_AGGREGATE_FIELDS:
        kite_node[key] = kite_stats[key]

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

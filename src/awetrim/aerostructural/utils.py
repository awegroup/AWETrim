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

import yaml
from pathlib import Path
import numpy as np
import json
import logging

import h5py


def load_yaml(path: Path) -> dict:
    """
    Read a YAML file and return the parsed data as a Python dict.

    Args:
        path (Path): The path to the YAML file.

    Returns:
        dict: The parsed data from the YAML file.
    """
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


#: Optional block of per-CONNECTION bridle rest lengths in a PSM-format dict
#: (the wireframe's internal geometry, and the retired file formats)
#: ``{headers: [row, l0], data: [[bridle_connections row, l0 [m]], ...]}``.
#: The bridle_lines table keys rest lengths by line NAME, and the two steering
#: tapes share one name, so an actuated (asymmetric) state is written here; a
#: listed connection's l0 replaces its line's. An awesIO document needs none:
#: every segment carries its own rest length
#: (``awesio_structure.wireframe_to_awesio`` / ``wireframe_from_awesio``
#: translate between the two).
BRIDLE_CONNECTION_REST_LENGTHS = "bridle_connection_rest_lengths"


def bridle_connection_rest_lengths(struc_geometry: dict) -> dict:
    """``{bridle_connections row index: l0}`` from the optional
    :data:`BRIDLE_CONNECTION_REST_LENGTHS` block; empty when absent."""
    block = struc_geometry.get(BRIDLE_CONNECTION_REST_LENGTHS) or {}
    return {int(row): float(l0) for row, l0 in block.get("data", [])}


def resolve_kcu_mass(config=None, system_config=None) -> float:
    """The KCU mass [kg] a structural model carries at its KCU point.

    ``system.yaml`` is the single source (``utils.system_config.get_kcu_mass``:
    the control unit plus its onboard turbine); a run config's ``kcu.mass`` /
    ``kcu_mass`` is the per-run fallback. An awesIO structure file carries no
    KCU mass. 0 with a warning when neither states one.
    """
    from awetrim.utils.system_config import get_kcu_mass

    if isinstance(system_config, dict):
        mass_kcu = get_kcu_mass(system_config)
        if mass_kcu is not None:
            return mass_kcu
    if isinstance(config, dict):
        kcu_cfg = config.get("kcu", {})
        if isinstance(kcu_cfg, dict) and "mass" in kcu_cfg:
            return float(kcu_cfg["mass"])
        if "kcu_mass" in config:
            return float(config["kcu_mass"])
    logging.warning(
        "KCU mass found in neither system_config nor config; defaulting to 0.0 kg."
    )
    return 0.0


def resolve_structure(structure, config=None, system_config=None) -> tuple:
    """The full model's solver arrays of an awesIO structure, KCU mass included.

    ``structure`` is an awesIO file path, a loaded document or a
    ``Structure`` (``coupled.awesio_structure.load_structure``). Returns the
    tuple the retired reader ``read_struc_geometry_yaml.main`` returned
    (``Structure.resolved``), with the KCU mass of :func:`resolve_kcu_mass`
    and the same mirror-symmetry warning. ``strut_padding`` in ``config`` no
    longer applies: an awesIO file holds the resolved mesh.
    """
    from awetrim.aerostructural.coupled.awesio_structure import load_structure
    from awetrim.aerostructural.symmetry import warn_if_asymmetric

    structure = load_structure(structure)
    if (config or {}).get("strut_padding") is not None:
        logging.warning(
            "config strut_padding=%r is ignored: an awesIO structure file holds "
            "the resolved strut mesh",
            config["strut_padding"],
        )
    resolved = structure.resolved(kcu_mass=resolve_kcu_mass(config, system_config))
    # An asymmetric input turns an unsteered kite; warn (see symmetry.py).
    warn_if_asymmetric(
        resolved[0], resolved[10], resolved[13], resolved[14], resolved[16], resolved[1],
        config,
    )
    return resolved


def save_results(tracking, meta, filename):
    """
    Save tracking arrays and metadata to an HDF5 file.

    Args:
        tracking (dict): Dictionary of arrays to save under the "tracking" group.
        meta (dict): Metadata dictionary to save as attributes.
        filename (str or Path): Output HDF5 file path.

    Returns:
        None
    """
    with h5py.File(filename, "w") as f:
        grp = f.create_group("tracking")
        for name, arr in tracking.items():
            grp.create_dataset(name, data=arr[: meta["n_iter"]], compression="gzip")
        for k, v in meta.items():
            # A single unstorable value must not cost the whole run. h5py
            # rejects anything without a native equivalent (lists of dicts,
            # ragged sequences, None), and this write is the LAST thing a
            # solve does -- losing minutes of coupled iteration here because
            # one new diagnostic was the wrong shape is never the right
            # trade. Store what fits, JSON the rest, and say so.
            try:
                grp.attrs[k] = v
            except (TypeError, ValueError):
                try:
                    grp.attrs[k] = json.dumps(v, default=str)
                    logging.debug("meta[%r] stored as a JSON string", k)
                except (TypeError, ValueError):
                    grp.attrs[k] = repr(v)
                    logging.warning(
                        "meta[%r] (%s) is not HDF5-storable; saved its repr",
                        k,
                        type(v).__name__,
                    )


def load_sim_output(h5_path):
    """
    Load simulation results and metadata from an HDF5 file written with h5py.

    Args:
        h5_path (str or Path): Path to the .h5 file (e.g. "sim_output.h5").

    Returns:
        tuple:
            metadata (dict): Run-level metadata (attributes from the file).
            track (dict): Dictionary of numpy arrays for each dataset under "tracking".
    """
    h5_path = Path(h5_path)
    if not h5_path.exists():
        raise FileNotFoundError(f"No such file: {h5_path}")

    with h5py.File(h5_path, "r") as f:
        if "tracking" not in f:
            raise KeyError(f"No 'tracking' group in {h5_path}")
        grp = f["tracking"]

        # load metadata
        metadata = {key: grp.attrs[key] for key in grp.attrs}

        # load all datasets under /tracking into numpy arrays
        track = {}
        for name, item in grp.items():
            if isinstance(item, h5py.Dataset):
                track[name] = item[()]  # read the full array into memory

    return metadata, track


def printing_rest_lengths(tracking_data, structure):
    """
    Print current and initial lengths of the bridle lines, each line's
    connections averaged (a pulley rope as the sum of its two arms), against
    their mean rest length in ``structure`` (an awesIO path, document or
    ``Structure``).
    """
    from awetrim.aerostructural.coupled.awesio_structure import load_structure

    positions = tracking_data["positions"]
    struc_nodes = positions[-1]  # current positions
    initial_struc_nodes = positions[0]  # initial positions

    def rope(nodes, path):
        return sum(np.linalg.norm(nodes[a] - nodes[b]) for a, b in zip(path[:-1], path[1:]))

    by_line = {}
    for element in load_structure(structure).bridle:
        by_line.setdefault(element["name"], []).append(element)

    # Collect data rows: (line_name, curr_length, l0, delta_pct, initial_nodal_dist)
    rows = []
    for line_name, elements in by_line.items():
        curr_l = float(np.mean([rope(struc_nodes, e["nodes"]) for e in elements]))
        init_nodal_dist = float(np.mean([rope(initial_struc_nodes, e["nodes"]) for e in elements]))
        l0 = float(np.mean([e["rest_length"] for e in elements]))
        delta_pct = 100.0 * (curr_l - l0) / l0 if l0 != 0 else 0.0
        rows.append((line_name, curr_l, l0, delta_pct, init_nodal_dist))

    if not rows:
        print("\nNo bridle lines with matching connections found.")
        return

    # Determine column widths for aligned output
    name_w = max(len(r[0]) for r in rows) if rows else 10

    # Print header
    print(
        f"\n{'Line':<{name_w}}   {'current_l':>10}   {'initial_l0_yaml':>16}   {'delta':>8}   {'initial_nodal_distance':>23}"
    )
    print(f"{'-' * name_w}   {'-' * 10}   {'-' * 16}   {'-' * 8}   {'-' * 23}")

    for name, curr_l, yaml_l0, delta_pct, init_dist in rows:
        yaml_str = f"{yaml_l0:.3f} m" if yaml_l0 is not None else "N/A"
        print(
            f"{name:<{name_w}}   {curr_l:>7.3f} m   {yaml_str:>16}   {delta_pct:>+7.2f}%   ({init_dist:>19.3f} m)"
        )


def _axis_to_unit_vector(axis, name="axis"):
    """
    Convert an axis definition to a unit 3D vector.

    Accepted forms:
      - labels: "x", "y", "z"
      - vectors: array-like with 3 numeric components
    """
    if isinstance(axis, str):
        axis_label = axis.strip().lower()
        axis_map = {
            "x": np.array([1.0, 0.0, 0.0]),
            "y": np.array([0.0, 1.0, 0.0]),
            "z": np.array([0.0, 0.0, 1.0]),
        }
        if axis_label not in axis_map:
            raise ValueError(
                f"Invalid {name} label '{axis}'. Allowed labels are 'x', 'y', 'z'."
            )
        vec = axis_map[axis_label]
    else:
        vec = _to_3_vector(axis, name)

    norm = np.linalg.norm(vec)
    if norm <= 1e-15:
        raise ValueError(f"{name} must be non-zero. Got {vec}.")
    return vec / norm


def _normalize_axes(axes):
    """
    Normalize a 3-axis sequence into three unit vectors.

    Supported input layouts:
      - ("x", "y", "z")
      - ([1,0,0], [0,1,0], [0,0,1])
      - np.array([[1,0,0],[0,1,0],[0,0,1]])
    """
    axes_arr = np.asarray(axes, dtype=object)
    if axes_arr.ndim == 1:
        if axes_arr.shape[0] != 3:
            raise ValueError(
                f"`axes` must contain exactly 3 axis definitions. Got {axes_arr.shape[0]}."
            )
        axis_defs = list(axes_arr)
    elif axes_arr.ndim == 2 and axes_arr.shape == (3, 3):
        axis_defs = [axes_arr[i, :] for i in range(3)]
    else:
        raise ValueError(
            "`axes` must be either length-3 labels/vectors or a (3,3) axis matrix."
        )

    return tuple(
        _axis_to_unit_vector(axis_defs[i], name=f"axes[{i}]") for i in range(3)
    )


def _rotation_matrix_from_axis(axis, angle_rad):
    """
    Return a 3x3 right-hand-rule rotation matrix about any 3D axis.

    Args:
        axis (str or array-like): "x"/"y"/"z" or a 3D axis vector.
        angle_rad (float): Rotation angle in radians.
    """
    ux, uy, uz = _axis_to_unit_vector(axis)
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    one_minus_c = 1.0 - c

    return np.array(
        [
            [
                c + ux * ux * one_minus_c,
                ux * uy * one_minus_c - uz * s,
                ux * uz * one_minus_c + uy * s,
            ],
            [
                uy * ux * one_minus_c + uz * s,
                c + uy * uy * one_minus_c,
                uy * uz * one_minus_c - ux * s,
            ],
            [
                uz * ux * one_minus_c - uy * s,
                uz * uy * one_minus_c + ux * s,
                c + uz * uz * one_minus_c,
            ],
        ]
    )


def _to_3_vector(values, name):
    """Convert input to a strict 3-vector of floats."""
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.shape != (3,):
        raise ValueError(
            f"{name} must contain exactly 3 values. Got shape {arr.shape}."
        )
    return arr


def _validate_struct_nodes_and_masses(struc_nodes, m_arr):
    """Validate and return structural nodes and nodal masses as numpy arrays."""
    nodes = np.asarray(struc_nodes, dtype=float)
    if nodes.ndim != 2 or nodes.shape[1] != 3:
        raise ValueError(
            f"`struc_nodes` must have shape (n_nodes, 3). Got shape {nodes.shape}."
        )

    masses = np.asarray(m_arr, dtype=float).reshape(-1)
    if masses.shape[0] != nodes.shape[0]:
        raise ValueError(
            "`m_arr` length must match number of structural nodes. "
            f"Got len(m_arr)={masses.shape[0]} and n_nodes={nodes.shape[0]}."
        )

    return nodes, masses


def calculate_cg(
    struc_nodes,
    m_arr,
    axes=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
):
    """
    Compute center of gravity (CG) coordinates in a chosen axis basis.

    Args:
        struc_nodes (array-like): Structural node positions with shape (n_nodes, 3).
        m_arr (array-like): Nodal masses with shape (n_nodes,).
        axes (array-like, optional): Three axis definitions used for output coordinates.
            Each axis can be "x"/"y"/"z" or a 3D vector.

    Returns:
        np.ndarray: CG coordinates along the provided axes, shape (3,).
    """
    nodes, masses = _validate_struct_nodes_and_masses(struc_nodes, m_arr)
    axes_norm = _normalize_axes(axes)

    total_mass = np.sum(masses)
    if abs(total_mass) <= 1e-15:
        raise ValueError("Total mass must be non-zero to compute center of gravity.")

    cg_xyz = np.sum(nodes * masses[:, None], axis=0) / total_mass
    return np.array([np.dot(cg_xyz, axis_i) for axis_i in axes_norm], dtype=float)


def calculate_inertia(nodes, desired_point=(0.0, 0.0, 0.0)):
    """
    Calculate the full 3x3 inertia tensor of point masses about a desired point.

    Parameters:
        nodes (sequence): Sequence of nodes, where each node is
            ``[position, mass]`` and position is a 3D coordinate.
        desired_point (array-like, optional): Point ``[x, y, z]`` about which
            the inertia tensor is computed. Defaults to the origin.

    Returns:
        np.ndarray: Inertia tensor with shape ``(3, 3)``.
    """
    ref_point = _to_3_vector(desired_point, "desired_point")
    inertia_tensor = np.zeros((3, 3), dtype=float)

    for i, node in enumerate(nodes):
        if len(node) != 2:
            raise ValueError(
                f"Each node must be [position, mass]. Invalid entry at index {i}: {node}"
            )

        position = _to_3_vector(node[0], f"nodes[{i}][0]")
        mass = float(node[1])

        r_x, r_y, r_z = position - ref_point
        inertia_tensor[0, 0] += mass * (r_y**2 + r_z**2)  # Ixx
        inertia_tensor[1, 1] += mass * (r_x**2 + r_z**2)  # Iyy
        inertia_tensor[2, 2] += mass * (r_x**2 + r_y**2)  # Izz
        inertia_tensor[0, 1] -= mass * r_x * r_y  # Ixy
        inertia_tensor[0, 2] -= mass * r_x * r_z  # Ixz
        inertia_tensor[1, 2] -= mass * r_y * r_z  # Iyz

    inertia_tensor[1, 0] = inertia_tensor[0, 1]
    inertia_tensor[2, 0] = inertia_tensor[0, 2]
    inertia_tensor[2, 1] = inertia_tensor[1, 2]

    return inertia_tensor


def compute_kite_aggregate(struc_geometry, system_config) -> dict:
    """Compute the aggregate kite mass, CG, and inertia from the structural nodes.

    The kite aggregate (wing + bridle + KCU) is *derived*, not stored in
    ``system.yaml``: the total mass is the sum of the nodal masses, while the CG
    and inertia tensor require the structural node distribution and are therefore
    computed from the structural nodes (the full model or a wireframe, e.g. a
    deformed snapshot) rather than from the per-component summaries.
    ``system_config`` is the awesIO system config (used for the KCU mass);
    ``struc_geometry`` is an awesIO structure (path, document or
    ``Structure``). The
    nodal masses follow awesIO's rule (``Structure.masses``).

    Returns a dict with ``mass`` (kg), ``center_of_mass`` ([x, y, z] m), and
    ``inertia_tensor`` (3x3, kg·m², about the CG).
    """
    # Imported lazily so this lightweight, widely-imported module does not pull in
    # the structural readers at import time.
    from awetrim.aerostructural.coupled.awesio_structure import load_structure

    structure = load_structure(struc_geometry)
    struc_nodes = structure.positions
    m_arr = structure.masses(kcu_mass=resolve_kcu_mass(system_config=system_config))

    total_mass = float(np.sum(m_arr))
    cg = calculate_cg(struc_nodes, m_arr)
    inertia = calculate_inertia(
        [(struc_nodes[i], m_arr[i]) for i in range(len(m_arr))],
        desired_point=cg,
    )
    return {
        "mass": round(total_mass, 4),
        "center_of_mass": [round(float(v), 4) for v in cg],
        "inertia_tensor": [[round(float(v), 4) for v in row] for row in inertia],
    }


def calculate_moments_of_inertia(
    struc_nodes,
    m_arr,
    point=(0.0, 0.0, 0.0),
    axes=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
):
    """
    Compute scalar moments of inertia around three axes through a reference point.

    Args:
        struc_nodes (array-like): Structural node positions with shape (n_nodes, 3).
        m_arr (array-like): Nodal masses with shape (n_nodes,).
        point (array-like, optional): Reference point on each inertia axis.
        axes (array-like, optional): Three axis definitions.
            Each axis can be "x"/"y"/"z" or a 3D vector.

    Returns:
        np.ndarray: Moments of inertia for the three axes, shape (3,).
    """
    nodes, masses = _validate_struct_nodes_and_masses(struc_nodes, m_arr)
    axes_norm = _normalize_axes(axes)
    point_mass_nodes = [(nodes[i], masses[i]) for i in range(nodes.shape[0])]
    inertia_tensor = calculate_inertia(point_mass_nodes, desired_point=point)

    return np.array(
        [float(axis_i @ inertia_tensor @ axis_i) for axis_i in axes_norm], dtype=float
    )


def _resolve_rotation_angles(angle_deg, angle_rad, axes):
    """Normalize `rotate_geometry`-style angle/axes inputs to radians + unit axes."""
    if (angle_deg is None) == (angle_rad is None):
        raise ValueError("Provide exactly one of `angle_deg` or `angle_rad`.")

    if angle_deg is not None:
        angle_vec_rad = np.radians(_to_3_vector(angle_deg, "angle_deg"))
    else:
        angle_vec_rad = _to_3_vector(angle_rad, "angle_rad")
    return angle_vec_rad, _normalize_axes(axes)


def rotation_matrix_from_angles(
    angle_deg=None,
    angle_rad=None,
    axes=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
):
    """
    Composed 3x3 matrix of the three sequential rotations `rotate_geometry`
    applies, so a caller can transform OTHER quantities (stored solver
    history, force vectors) with exactly the geometry's rotation:
    ``rotate_geometry(x, ...) == (x - point) @ R.T + point``.
    """
    angle_vec_rad, axes_norm = _resolve_rotation_angles(angle_deg, angle_rad, axes)
    rotation = np.eye(3)
    for ax, ang in zip(axes_norm, angle_vec_rad):
        rotation = _rotation_matrix_from_axis(ax, ang) @ rotation
    return rotation


def rotate_geometry(
    struc_nodes,
    angle_deg=None,
    angle_rad=None,
    point=(0.0, 0.0, 0.0),
    axes=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
):
    """
    Rotate structural nodes with three sequential axis-angle rotations.

    Args:
        struc_nodes (np.ndarray): Array of node positions (n_nodes, 3).
        angle_deg (array-like, optional): Three angles in degrees.
        angle_rad (array-like, optional): Three angles in radians.
        point (array-like, optional): Pivot point for rotation. Defaults to origin.
        axes (array-like, optional): Three axis definitions defining order.
            Each entry can be a label ("x"/"y"/"z") or a 3D axis vector.
            Defaults to canonical Cartesian vectors (x, y, z).

    Notes:
        - Exactly one of `angle_deg` or `angle_rad` must be provided.
    """
    rotation = rotation_matrix_from_angles(
        angle_deg=angle_deg, angle_rad=angle_rad, axes=axes
    )

    pivot = _to_3_vector(point, "point")
    nodes = np.asarray(struc_nodes, dtype=float)
    if nodes.ndim != 2 or nodes.shape[1] != 3:
        raise ValueError(
            f"`struc_nodes` must have shape (n_nodes, 3). Got shape {nodes.shape}."
        )

    return (nodes - pivot) @ rotation.T + pivot

# Copyright (c) 2023-2026 Oriol Cayon, Delft University of Technology
#
# Licensed under the EUPL, Version 1.2 or - as soon they will be approved by
# the European Commission - subsequent versions of the EUPL (the "Licence");
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at:
#
#     https://joinup.ec.europa.eu/software/page/eupl
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the Licence is distributed on an "AS IS" basis,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the Licence for the specific language governing permissions and
# limitations under the Licence.
#
# SPDX-License-Identifier: EUPL-1.2

"""Rigid-body principal axis identification from a PSM mass distribution."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from awetrim.aerostructural.utils import (
    _validate_struct_nodes_and_masses,
    calculate_cg,
    calculate_inertia,
)


@dataclass
class RigidBodyAxes:
    """Principal body axes derived from inertia tensor eigendecomposition.

    Attributes:
        cg: Center of gravity in the structural frame, shape (3,).
        cg_body: CG position expressed in body-axis coordinates, shape (3,).
            Components of the structural-frame CG vector along each body axis,
            i.e. ``body_axes @ cg``.
        inertia_cg: Full 3x3 inertia tensor about the CG in the structural frame.
        principal_moments: Principal moments of inertia [I_x, I_y, I_z] for the
            assigned body axes, shape (3,).
        body_axes: Rotation matrix whose *rows* are the unit body-axis vectors
            [x_body, y_body, z_body], shape (3,3).  Transforms a vector from body
            frame to structural frame via ``body_axes.T @ v_body``.
    """

    cg: np.ndarray
    cg_body: np.ndarray
    inertia_cg: np.ndarray
    principal_moments: np.ndarray
    body_axes: np.ndarray


def compute_rigid_body_axes(
    struc_nodes: np.ndarray,
    m_arr: np.ndarray,
) -> RigidBodyAxes:
    """Compute CG, inertia tensor, and principal body axes from PSM node masses.

    Steps:
    1. Compute CG as the mass-weighted average node position.
    2. Assemble the 3×3 inertia tensor about the CG.
    3. Eigendecompose the (symmetric) inertia tensor to obtain the three
       principal-axis directions (eigenvectors) and the corresponding principal
       moments (eigenvalues).
    4. Assign each principal axis to the body axis (x, y, or z) whose canonical
       direction it is most closely aligned with, using a greedy max-alignment
       strategy.  Each axis is flipped when its dot product with the
       corresponding canonical direction is negative.

    Args:
        struc_nodes: Particle positions, shape (n_nodes, 3).
        m_arr: Particle masses, shape (n_nodes,).

    Returns:
        RigidBodyAxes dataclass containing cg, inertia_cg, principal_moments,
        and body_axes.
    """
    nodes, masses = _validate_struct_nodes_and_masses(struc_nodes, m_arr)

    cg = calculate_cg(nodes, masses)

    node_mass_pairs = [(nodes[i], masses[i]) for i in range(len(nodes))]
    inertia_cg = calculate_inertia(node_mass_pairs, desired_point=cg)

    # eigh guarantees real, orthonormal eigenvectors for a symmetric matrix
    eigenvalues, eigenvectors = np.linalg.eigh(inertia_cg)
    # eigenvectors[:,i] is the i-th principal axis (unit vector)

    # abs_dots[i,j] = |dot(eigenvec_i, e_j)|  where e_j is the j-th canonical axis
    abs_dots = np.abs(eigenvectors.T).copy()  # shape (3, 3)

    assigned = [-1, -1, -1]  # assigned[j] = eigenvec index assigned to body axis j
    for _ in range(3):
        i, j = np.unravel_index(np.argmax(abs_dots), abs_dots.shape)
        assigned[j] = int(i)
        abs_dots[i, :] = -1.0  # mark eigenvec i as consumed
        abs_dots[:, j] = -1.0  # mark canonical axis j as consumed

    # Course-frame unit vectors expressed in the structural frame.
    # The structural frame has X and Y negated relative to the course frame:
    #   T_structural_from_C = diag(-1, -1, 1)
    # so the course canonical directions in structural coordinates are:
    #   X_C → [-1, 0, 0],  Y_C → [0, -1, 0],  Z_C → [0, 0, 1]
    _COURSE_IN_STRUC = np.array([[-1., 0., 0.], [0., -1., 0.], [0., 0., 1.]])

    body_axes = np.zeros((3, 3))
    principal_moments = np.zeros(3)
    for body_j, eigen_i in enumerate(assigned):
        axis = eigenvectors[:, eigen_i].copy()
        if np.dot(axis, _COURSE_IN_STRUC[body_j]) < 0:  # flip to match course-frame sense
            axis = -axis
        body_axes[body_j] = axis
        principal_moments[body_j] = eigenvalues[eigen_i]

    return RigidBodyAxes(
        cg=cg,
        cg_body=body_axes @ cg,
        inertia_cg=inertia_cg,
        principal_moments=principal_moments,
        body_axes=body_axes,
    )


def load_psm_nodes_and_masses(struc_geometry: dict) -> tuple[np.ndarray, np.ndarray]:
    """Extract particle positions and lumped masses from a struc_geometry dict.

    Masses are distributed to nodes by splitting each element's mass equally
    between its two endpoints (consistent with initialize_wing_structure).
    Wing elements, bridle lines, and pulley masses are all included when present.

    Args:
        struc_geometry: Parsed struc_geometry YAML as a Python dict.

    Returns:
        Tuple (struc_nodes, m_arr) where struc_nodes has shape (n_nodes, 3) and
        m_arr has shape (n_nodes,).
    """
    node_positions: dict[int, np.ndarray] = {}
    node_mass: dict[int, float] = {}

    # Wing particles
    for row in struc_geometry["wing_particles"]["data"]:
        nid = int(row[0])
        node_positions[nid] = np.array(row[1:4], dtype=float)
        node_mass[nid] = 0.0

    # KCU attachment particle (node 0) — mass lives in struc_geometry["kcu_mass"]
    if "bridle_point_node" in struc_geometry:
        nid = 0
        node_positions[nid] = np.array(struc_geometry["bridle_point_node"], dtype=float)
        node_mass[nid] = float(struc_geometry.get("kcu_mass", 0.0))

    # Bridle particles
    if "bridle_particles" in struc_geometry:
        for row in struc_geometry["bridle_particles"]["data"]:
            nid = int(row[0])
            node_positions[nid] = np.array(row[1:4], dtype=float)
            node_mass.setdefault(nid, 0.0)

    # Wing element masses → split equally to endpoints
    wing_elements = {
        row[0]: dict(zip(struc_geometry["wing_elements"]["headers"][1:], row[1:]))
        for row in struc_geometry["wing_elements"]["data"]
    }
    for conn_name, ci, cj in struc_geometry["wing_connections"]["data"]:
        m = float(wing_elements[conn_name]["m"])
        node_mass[ci] += m / 2.0
        node_mass[cj] += m / 2.0

    # Bridle line masses (computed from geometry + material density)
    if "bridle_elements" in struc_geometry and "bridle_connections" in struc_geometry:
        bridle_defs = {
            row[0]: row[1:] for row in struc_geometry["bridle_elements"]["data"]
        }
        for conn_row in struc_geometry["bridle_connections"]["data"]:
            conn_name = conn_row[0]
            if conn_name not in bridle_defs:
                continue
            bdef = bridle_defs[conn_name]
            # bdef: [l0, diameter, material]
            try:
                l0 = float(bdef[0])
                d = float(bdef[1])
                material = str(bdef[2])
                density = float(struc_geometry[material]["density"])
                area = np.pi * (d / 2.0) ** 2
                m_line = density * area * l0
            except (KeyError, IndexError, ValueError):
                continue
            ci = int(conn_row[1])
            cj = int(conn_row[2])
            node_mass[ci] += m_line / 2.0
            node_mass[cj] += m_line / 2.0
            # Pulley mass
            if len(conn_row) > 3 and conn_row[3] not in (None, "", 0):
                ck = int(conn_row[3])
                node_mass.setdefault(ck, 0.0)
                node_mass[ck] += m_line / 2.0
                node_mass[cj] += float(
                    struc_geometry.get("pulley_mass", 0.0)
                )

    node_ids = sorted(node_positions.keys())
    struc_nodes = np.array([node_positions[nid] for nid in node_ids])
    m_arr = np.array([node_mass[nid] for nid in node_ids])

    return struc_nodes, m_arr

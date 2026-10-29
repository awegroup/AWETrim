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

import numpy as np

from awetrim.aerodynamics.apparent_wind import apparent_wind_at, inflow_state_of


def compute_line_aerodynamic_force(p1, p2, d, va, cd_cable, cf_cable, rho):

    if p1[2] > p2[2]:
        p1, p2 = p2, p1

    length = np.linalg.norm(p2 - p1)
    ej = (p2 - p1) / length
    theta = np.arccos(np.dot(va, ej) / (np.linalg.norm(va) * np.linalg.norm(ej)))

    cd_t = cd_cable * np.sin(theta) ** 3 + np.pi * cf_cable * np.cos(theta) ** 3
    cl_t = (
        cd_cable * np.sin(theta) ** 2 * np.cos(theta)
        - np.pi * cf_cable * np.sin(theta) * np.cos(theta) ** 2
    )
    dir_D = va / np.linalg.norm(va)  # Drag direction
    dir_L = -(ej - np.dot(ej, dir_D) * dir_D)  # Lift direction
    dynamic_pressure_area = 0.5 * rho * np.linalg.norm(va) ** 2 * length * d

    # Calculate lift and drag using the common factor
    lift_j = dynamic_pressure_area * cl_t * dir_L
    drag_j = dynamic_pressure_area * cd_t * dir_D

    return lift_j + drag_j


def _extract_body_aero_bridle_line_system(body_aero):
    """Return VSM bridle line system as a list of [p1, p2, d] or None."""
    if body_aero is None:
        return None
    line_system = getattr(body_aero, "_bridle_line_system", None)
    if not line_system:
        return None
    return line_system


def build_bridle_node_pairs_from_line_system(struc_nodes, line_system):
    """
    Build structural node index pairs that correspond to each VSM bridle line endpoint.

    The mapping is computed from the initial geometry and can be reused at each iteration.
    """
    if line_system is None:
        return None

    node_pairs = []
    for line in line_system:
        p1 = np.asarray(line[0], dtype=float)
        p2 = np.asarray(line[1], dtype=float)
        idx_1 = int(np.argmin(np.linalg.norm(struc_nodes - p1, axis=1)))
        idx_2 = int(np.argmin(np.linalg.norm(struc_nodes - p2, axis=1)))
        node_pairs.append([idx_1, idx_2])

    return np.asarray(node_pairs, dtype=int)


def main(
    struc_nodes,
    bridle_connectivity_arr,
    bridle_diameters_arr,
    vel_app,
    rho,
    cd_cable,
    cf_cable,
    body_aero=None,
    bridle_node_pairs=None,
):
    """
    Compute aerodynamic forces on all bridle lines and distribute them to nodes.

    Args:
        struc_nodes: Array of structural node positions, shape (n_nodes, 3)
        bridle_connectivity_arr: Array of bridle line connections, shape (n_bridle_lines, 2)
                                 Each row contains [node_i, node_j] indices
        bridle_diameters_arr: Array of bridle line diameters, shape (n_bridle_lines,)
        vel_app: Apparent wind velocity vector, shape (3,)
        rho: Air density [kg/m³]
        cd_cable: Drag coefficient for cables
        cf_cable: Friction coefficient for cables

    Returns:
        f_aero_bridle: Array of aerodynamic forces on nodes, shape (n_nodes, 3)
    """
    # Initialize force array with zeros for all structural nodes
    f_aero_bridle = np.zeros_like(struc_nodes)

    line_system = _extract_body_aero_bridle_line_system(body_aero)
    # ``body_aero.va`` is the FREESTREAM at the reference point, not the inflow
    # at a bridle segment: the VSM adds the rotational term
    # ``-omega x (r - r0)`` per station (see awetrim.aerodynamics.apparent_wind).
    # Charging va_free to every segment is the same class of error as charging
    # the wing's mean -- it just errs the other way -- and it vanishes on an
    # unsteered baseline, which is why it went unseen. Each segment below is
    # evaluated at its own midpoint instead, matching what the trim balances.
    if body_aero is None:
        va, omega, reference_point = (
            np.asarray(vel_app, dtype=float),
            np.zeros(3),
            np.zeros(3),
        )
    else:
        va, omega, reference_point = inflow_state_of(body_aero)
        if va is None:
            # Per-panel distribution: no single freestream to rotate about, so
            # charge the caller's own vector and add no rotational term.
            va, omega = np.asarray(vel_app, dtype=float), np.zeros(3)

    def _inflow(p1, p2):
        """Apparent wind at this segment's midpoint, where its force acts."""
        return apparent_wind_at(va, omega, 0.5 * (p1 + p2), reference_point)

    # Preferred path: use VSM bridle mapping but evaluate force on current structural
    # endpoints each iteration.
    if line_system is not None:
        if bridle_node_pairs is None:
            bridle_node_pairs = build_bridle_node_pairs_from_line_system(
                struc_nodes, line_system
            )

        for idx, (ci, cj) in enumerate(bridle_node_pairs):
            p1 = np.asarray(struc_nodes[int(ci)], dtype=float)
            p2 = np.asarray(struc_nodes[int(cj)], dtype=float)
            if idx < len(line_system) and len(line_system[idx]) >= 3:
                d = float(line_system[idx][2])
            else:
                d = float(bridle_diameters_arr[idx])

            line = [p1, p2, d]
            va_segment = _inflow(p1, p2)
            if body_aero is not None and hasattr(
                body_aero, "compute_line_aerodynamic_force"
            ):
                f_line_total = body_aero.compute_line_aerodynamic_force(
                    va_segment,
                    line,
                    cd_cable=cd_cable,
                    cf_cable=cf_cable,
                    rho=rho,
                )
            else:
                f_line_total = compute_line_aerodynamic_force(
                    p1,
                    p2,
                    d,
                    va_segment,
                    cd_cable,
                    cf_cable,
                    rho,
                )

            f_aero_bridle[int(ci)] += f_line_total / 2.0
            f_aero_bridle[int(cj)] += f_line_total / 2.0

        return f_aero_bridle

    # Loop through each bridle line segment
    for idx, (ci, cj) in enumerate(bridle_connectivity_arr):
        # Get node positions
        p1 = struc_nodes[ci]
        p2 = struc_nodes[cj]

        # Get diameter for this line segment
        d = bridle_diameters_arr[idx]

        # Compute total aerodynamic force on this line segment
        f_line_total = compute_line_aerodynamic_force(
            p1, p2, d, _inflow(np.asarray(p1, dtype=float), np.asarray(p2, dtype=float)),
            cd_cable, cf_cable, rho,
        )

        # Distribute force equally to both nodes (50/50 split)
        f_aero_bridle[ci] += f_line_total / 2.0
        f_aero_bridle[cj] += f_line_total / 2.0

    return f_aero_bridle

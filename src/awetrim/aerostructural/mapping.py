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

"""Aerodynamic/structural mapping adapters."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from awetrim.aerostructural.protocols import (
    AeroToStructureMap,
    AerodynamicGeometryUpdate,
)


def interpolate_points(points: np.ndarray, n_panels_per_section: int) -> np.ndarray:
    """Return points with linear subdivisions between structural sections."""
    points = np.asarray(points, dtype=float)
    n_panels_per_section = int(n_panels_per_section)
    if n_panels_per_section <= 0:
        raise ValueError("n_panels_per_section must be greater than 0.")

    interpolated = []
    for idx in range(len(points) - 1):
        p0 = points[idx]
        p1 = points[idx + 1]
        for panel_idx in range(n_panels_per_section):
            t = panel_idx / n_panels_per_section
            interpolated.append((1.0 - t) * p0 + t * p1)
    interpolated.append(points[-1])
    return np.asarray(interpolated, dtype=float)


class LinearStructuralToAeroMapper:
    """Map structural leading/trailing edge nodes to aerodynamic edge points."""

    def map(
        self,
        nodes: np.ndarray,
        le_node_indices: np.ndarray,
        te_node_indices: np.ndarray,
        n_panels_per_section: int,
    ) -> AerodynamicGeometryUpdate:
        nodes = np.asarray(nodes, dtype=float)
        le_points = nodes[np.asarray(le_node_indices, dtype=int)]
        te_points = nodes[np.asarray(te_node_indices, dtype=int)]
        if int(n_panels_per_section) == 1:
            leading_edge_points = le_points
            trailing_edge_points = te_points
        else:
            leading_edge_points = interpolate_points(le_points, n_panels_per_section)
            trailing_edge_points = interpolate_points(te_points, n_panels_per_section)
        return AerodynamicGeometryUpdate(
            leading_edge_points=leading_edge_points,
            trailing_edge_points=trailing_edge_points,
        )


class BilinearAeroToStructuralLoadMapper:
    """Distribute panel loads to structural corner nodes with force preservation."""

    def initialize(
        self,
        panels: Sequence[object],
        nodes: np.ndarray,
        le_node_indices: np.ndarray,
        te_node_indices: np.ndarray,
    ) -> AeroToStructureMap:
        nodes = np.asarray(nodes, dtype=float)
        le_node_indices = np.asarray(le_node_indices, dtype=int)
        te_node_indices = np.asarray(te_node_indices, dtype=int)

        le_y = nodes[le_node_indices, 1]
        te_y = nodes[te_node_indices, 1]
        le_order = np.argsort(le_y)
        te_order = np.argsort(te_y)
        le_sorted_idx = le_node_indices[le_order]
        te_sorted_idx = te_node_indices[te_order]
        le_sorted_y = le_y[le_order]
        te_sorted_y = te_y[te_order]

        panel_corner_map = np.zeros((len(panels), 4), dtype=int)
        for idx, panel in enumerate(panels):
            y_cp = panel.aerodynamic_center[1]
            hi_le = np.searchsorted(le_sorted_y, y_cp)
            lo_le = np.clip(hi_le - 1, 0, len(le_sorted_y) - 1)
            hi_le = np.clip(hi_le, 0, len(le_sorted_y) - 1)

            hi_te = np.searchsorted(te_sorted_y, y_cp)
            lo_te = np.clip(hi_te - 1, 0, len(te_sorted_y) - 1)
            hi_te = np.clip(hi_te, 0, len(te_sorted_y) - 1)

            panel_corner_map[idx, :] = [
                le_sorted_idx[lo_le],
                le_sorted_idx[hi_le],
                te_sorted_idx[lo_te],
                te_sorted_idx[hi_te],
            ]
        return AeroToStructureMap(panel_corner_map=panel_corner_map)

    def map_loads(
        self,
        panel_forces: np.ndarray,
        panel_points: np.ndarray,
        nodes: np.ndarray,
        mapping: AeroToStructureMap,
    ) -> np.ndarray:
        panel_forces = np.asarray(panel_forces, dtype=float)
        nodes = np.asarray(nodes, dtype=float)
        panel_points = np.asarray(panel_points, dtype=float)
        panel_corner_map = np.asarray(mapping.panel_corner_map, dtype=int)
        nodal_forces = np.zeros((len(nodes), 3), dtype=float)

        for panel_idx, (cp, force) in enumerate(zip(panel_points, panel_forces)):
            le_lo, le_hi, te_lo, te_hi = panel_corner_map[panel_idx]
            r_le_lo = nodes[le_lo]
            r_le_hi = nodes[le_hi]
            r_te_lo = nodes[te_lo]
            r_te_hi = nodes[te_hi]

            dy_le = r_le_hi[1] - r_le_lo[1]
            eta = 0.5 if abs(dy_le) < 1e-6 else (cp[1] - r_le_lo[1]) / dy_le
            eta = np.clip(eta, 0.0, 1.0)

            chord_lo = r_te_lo - r_le_lo
            chord_hi = r_te_hi - r_le_hi
            chord_lo_sq = np.dot(chord_lo, chord_lo)
            chord_hi_sq = np.dot(chord_hi, chord_hi)
            xi_lo = (
                0.0
                if chord_lo_sq < 1e-12
                else np.dot(cp - r_le_lo, chord_lo) / chord_lo_sq
            )
            xi_hi = (
                0.0
                if chord_hi_sq < 1e-12
                else np.dot(cp - r_le_hi, chord_hi) / chord_hi_sq
            )
            xi_lo = np.clip(xi_lo, 0.0, 1.0)
            xi_hi = np.clip(xi_hi, 0.0, 1.0)

            weights = [
                (1.0 - eta) * (1.0 - xi_lo),
                eta * (1.0 - xi_hi),
                (1.0 - eta) * xi_lo,
                eta * xi_hi,
            ]
            for node_idx, weight in zip([le_lo, le_hi, te_lo, te_hi], weights):
                nodal_forces[node_idx] += weight * force

        return nodal_forces


def check_moment_preservation(
    panel_forces: np.ndarray,
    panel_points: np.ndarray,
    nodal_forces: np.ndarray,
    nodes: np.ndarray,
    ref_point: np.ndarray | None = None,
) -> dict:
    """Report force and moment errors introduced by aero-to-structure mapping."""
    if ref_point is None:
        ref_point = np.zeros(3)
    else:
        ref_point = np.asarray(ref_point, dtype=float)

    panel_forces = np.asarray(panel_forces, dtype=float)
    panel_points = np.asarray(panel_points, dtype=float)
    nodal_forces = np.asarray(nodal_forces, dtype=float)
    nodes = np.asarray(nodes, dtype=float)

    force_panel_total = np.sum(panel_forces, axis=0)
    force_node_total = np.sum(nodal_forces, axis=0)
    d_force = force_node_total - force_panel_total

    moment_panel = np.zeros(3)
    for point, force in zip(panel_points, panel_forces):
        moment_panel += np.cross(point - ref_point, force)

    moment_node = np.zeros(3)
    for node, force in zip(nodes, nodal_forces):
        moment_node += np.cross(node - ref_point, force)

    d_moment = moment_node - moment_panel
    moment_panel_norm = np.linalg.norm(moment_panel)
    d_moment_rel = (
        np.linalg.norm(d_moment) / moment_panel_norm
        if moment_panel_norm > 1e-12
        else 0.0
    )

    return {
        "F_aero_total": force_panel_total,
        "F_struc_total": force_node_total,
        "dF": d_force,
        "dF_norm": np.linalg.norm(d_force),
        "M_aero": moment_panel,
        "M_struc": moment_node,
        "dM": d_moment,
        "dM_norm": np.linalg.norm(d_moment),
        "dM_rel": d_moment_rel,
    }


__all__ = [
    "BilinearAeroToStructuralLoadMapper",
    "LinearStructuralToAeroMapper",
    "check_moment_preservation",
    "interpolate_points",
]

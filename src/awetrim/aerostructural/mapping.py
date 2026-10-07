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


class SkinStructuralToAeroMapper:
    """Carry aerodynamic edge points on a coarser structure by shape functions.

    For a structure that does not have a node at every aerodynamic section --
    a wireframe of a few ribs flying the aerodynamic mesh of the full geometry
    -- each aerodynamic leading- and trailing-edge point is attached to the
    rib bay it lies in: bilinear coordinates ``(eta, xi)`` on the quad of the
    bay's four rib nodes (front/rear of two ribs) plus an offset ``w`` along the
    quad's normal, all solved once on the reference shape. Afterwards each
    point follows the structure as ``S(eta, xi) + w n(eta, xi)``, which
    reproduces the reference points exactly and moves rigidly with a rigidly
    moving bay. Points beyond a bay (the true leading edge ahead of the front
    attachment, the tips outboard of the last rib) are extrapolated with the
    same functions.

    The loads go back through :class:`BilinearAeroToStructuralLoadMapper` on
    the same rib nodes, so force is conserved either way.
    """

    def initialize(
        self,
        reference_nodes: np.ndarray,
        front_indices: Sequence[int],
        rear_indices: Sequence[int],
        le_points: np.ndarray,
        te_points: np.ndarray,
        n_panels_per_section: int,
    ) -> "SkinStructuralToAeroMapper":
        """``front_indices`` / ``rear_indices``: the rib nodes, in span order."""
        self.front = np.asarray(front_indices, dtype=int)
        self.rear = np.asarray(rear_indices, dtype=int)
        if len(self.front) != len(self.rear) or len(self.front) < 2:
            raise ValueError("need at least two ribs, with a front and a rear node each")
        self.n_panels_per_section = int(n_panels_per_section)
        nodes = np.asarray(reference_nodes, dtype=float)
        rib_y = 0.5 * (nodes[self.front, 1] + nodes[self.rear, 1])
        self._descending = rib_y[0] > rib_y[-1]
        self.anchors = []
        for points in (le_points, te_points):
            anchors = []
            for point in np.asarray(points, dtype=float):
                bay = self._bay(rib_y, point[1])
                anchors.append((bay, *self._coordinates(nodes, bay, point)))
            self.anchors.append(anchors)
        return self

    def _bay(self, rib_y, y):
        order = rib_y if not self._descending else -rib_y
        target = y if not self._descending else -y
        bay = int(np.searchsorted(order, target)) - 1
        return int(np.clip(bay, 0, len(rib_y) - 2))

    def _quad(self, nodes, bay):
        return (
            nodes[self.front[bay]], nodes[self.front[bay + 1]],
            nodes[self.rear[bay]], nodes[self.rear[bay + 1]],
        )

    @staticmethod
    def _surface(quad, eta, xi):
        a, b, c, d = quad
        point = (1 - eta) * (1 - xi) * a + eta * (1 - xi) * b + (1 - eta) * xi * c + eta * xi * d
        d_eta = (1 - xi) * (b - a) + xi * (d - c)
        d_xi = (1 - eta) * (c - a) + eta * (d - b)
        normal = np.cross(d_eta, d_xi)
        return point, d_eta, d_xi, normal / np.linalg.norm(normal)

    def _coordinates(self, nodes, bay, point):
        quad = self._quad(nodes, bay)
        eta, xi, w = 0.5, 0.5, 0.0
        for _ in range(50):
            surface, d_eta, d_xi, normal = self._surface(quad, eta, xi)
            residual = point - (surface + w * normal)
            jacobian = np.column_stack([d_eta, d_xi, normal])
            step = np.linalg.solve(jacobian, residual)
            eta, xi, w = eta + step[0], xi + step[1], w + step[2]
            if np.linalg.norm(step) < 1e-13:
                break
        return eta, xi, w

    def _points(self, nodes, anchors):
        out = []
        for bay, eta, xi, w in anchors:
            surface, _, _, normal = self._surface(self._quad(nodes, bay), eta, xi)
            out.append(surface + w * normal)
        return np.asarray(out)

    def map(self, nodes: np.ndarray, *_ignored) -> AerodynamicGeometryUpdate:
        """Aerodynamic edges on ``nodes``, subdivided like the reference mesh.

        Takes (and ignores) the extra arguments of
        :meth:`LinearStructuralToAeroMapper.map`, so either can be plugged in.
        """
        nodes = np.asarray(nodes, dtype=float)
        le, te = (self._points(nodes, anchors) for anchors in self.anchors)
        if self.n_panels_per_section > 1:
            le = interpolate_points(le, self.n_panels_per_section)
            te = interpolate_points(te, self.n_panels_per_section)
        return AerodynamicGeometryUpdate(leading_edge_points=le, trailing_edge_points=te)


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

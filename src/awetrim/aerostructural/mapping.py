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

            # Spanwise station: projection on the 3-D LE segment. Not the
            # y-ratio it was until 2026-10-26: at the tip the LE curls down to
            # near-vertical (LEI-V3 nodes 17-19: 805 mm long, a few mm in y),
            # so a mm of drift flipped the sign of dy and threw the whole tip
            # panel load to the other node -- a 2-iteration ~37 N residual
            # spike every 5-6 iterations at 45 deg elevation.
            le_seg = r_le_hi - r_le_lo
            le_seg_sq = np.dot(le_seg, le_seg)
            eta = 0.5 if le_seg_sq < 1e-12 else np.dot(cp - r_le_lo, le_seg) / le_seg_sq
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


def _normal_basis(directions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two unit vectors normal to each row of ``directions`` (n, 3)."""
    u = directions / np.linalg.norm(directions, axis=1, keepdims=True)
    helper = np.where(np.abs(u[:, 2:3]) < 0.9, [[0.0, 0.0, 1.0]], [[1.0, 0.0, 0.0]])
    e1 = np.cross(u, helper)
    e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
    return e1, np.cross(u, e1)


class StrutLeverAeroToStructuralLoadMapper:
    """Rigid-body panel loads onto the struts of their bay, force AND moment exact.

    Each panel belongs to one bay, between two struts; a strut is the straight
    line from its front rib node to its rear one. A panel load -- force ``F``
    at the point ``x`` and moment ``m`` about ``x`` -- goes to the bay's four
    nodes as a rigid body would carry it (Cayon 2022, MSc thesis, Sec. 6.3:
    lever rule onto the strut lines, then onto their nodes), with the moment
    of every move carried along so nothing is lost:

    1. span: ``F`` is split between the two struts by the lever rule at the
       closest points ``p_A``, ``p_B`` of the strut lines to ``x``;
    2. carried moment: the struts still owe
       ``R = m - (p_A - x) x F_A - (p_B - x) x F_B``, zero only when ``x``
       lies on ``p_A p_B`` (a flat bay) -- the term the thesis code
       (``functions_PSM.force2nodes``, closest points on skew lines) dropped;
    3. ``R`` goes to the struts as couples about axes normal to each strut and,
       for its part about a strut's own axis, as a couple across the span:
       ``R = M_A + M_B + (p_B - p_A) x G``, the minimum-norm split of these
       three equations over their six unknowns;
    4. each strut: its force at ``p_i`` by the lever rule along the strut, its
       couple ``M_i`` as ``+-(M_i x d)/|d|^2`` on its rear/front node.

    Every step conserves force and moment, so the nodal forces reproduce the
    panel's resultant and its moment about any point to rounding, on warped
    bays and for points off the bay (billowed panels). On a flat rectangle with
    the point inside it is the bilinear split. The panel's bay is assigned by
    construction (:func:`awetrim.aerostructural.billowing.panel_bays`), not
    by sorting on y, which fails at a curled tip.
    """

    def initialize(
        self,
        panel_bays: Sequence[int],
        front_indices: Sequence[int],
        rear_indices: Sequence[int],
    ) -> AeroToStructureMap:
        """``front_indices`` / ``rear_indices``: the rib nodes in span order;
        panel ``j`` lies between ribs ``panel_bays[j]`` and ``panel_bays[j] + 1``."""
        bays = np.asarray(panel_bays, dtype=int)
        front = np.asarray(front_indices, dtype=int)
        rear = np.asarray(rear_indices, dtype=int)
        if len(front) != len(rear) or bays.min() < 0 or bays.max() > len(front) - 2:
            raise ValueError("panel bays must index pairs of consecutive ribs")
        # [front_a, front_b, rear_a, rear_b], the bilinear map's layout
        corner = np.column_stack([front[bays], front[bays + 1], rear[bays], rear[bays + 1]])
        return AeroToStructureMap(panel_corner_map=corner)

    def map_loads(
        self,
        panel_forces: np.ndarray,
        panel_points: np.ndarray,
        nodes: np.ndarray,
        mapping: AeroToStructureMap,
        panel_moments: np.ndarray | None = None,
    ) -> np.ndarray:
        """Nodal forces (len(nodes), 3). ``panel_moments``: each panel's moment
        about its own ``panel_points`` entry (None = zero)."""
        F = np.asarray(panel_forces, dtype=float)
        x = np.asarray(panel_points, dtype=float)
        m = np.zeros_like(F) if panel_moments is None else np.asarray(panel_moments, dtype=float)
        nodes = np.asarray(nodes, dtype=float)
        corner = np.asarray(mapping.panel_corner_map, dtype=int)
        front_a, front_b = nodes[corner[:, 0]], nodes[corner[:, 1]]
        rear_a, rear_b = nodes[corner[:, 2]], nodes[corner[:, 3]]

        def dot(u, v):
            return np.einsum("ij,ij->i", u, v)

        # 1. lever rule along the span, at the closest points of the strut lines
        d_a, d_b = rear_a - front_a, rear_b - front_b
        t_a = dot(x - front_a, d_a) / dot(d_a, d_a)
        t_b = dot(x - front_b, d_b) / dot(d_b, d_b)
        p_a = front_a + t_a[:, None] * d_a
        p_b = front_b + t_b[:, None] * d_b
        span = p_b - p_a
        s = dot(x - p_a, span) / dot(span, span)
        f_a, f_b = (1.0 - s)[:, None] * F, s[:, None] * F

        # 2. the moment the struts still owe
        R = m - np.cross(p_a - x, f_a) - np.cross(p_b - x, f_b)

        # 3. R = M_a + M_b + span x G, minimum norm (G scaled by the span length
        #    so every column is a moment of comparable size)
        a1, a2 = _normal_basis(d_a)
        b1, b2 = _normal_basis(d_b)
        g1, g2 = _normal_basis(span)
        length = np.linalg.norm(span, axis=1)[:, None]
        A = np.stack(
            [a1, a2, b1, b2, np.cross(span, g1) / length, np.cross(span, g2) / length],
            axis=2,
        )  # (n, 3, 6)
        c = (np.transpose(A, (0, 2, 1)) @ np.linalg.solve(A @ np.transpose(A, (0, 2, 1)), R[..., None]))[..., 0]
        M_a = c[:, 0:1] * a1 + c[:, 1:2] * a2
        M_b = c[:, 2:3] * b1 + c[:, 3:4] * b2
        G = (c[:, 4:5] * g1 + c[:, 5:6] * g2) / length

        # 4. each strut onto its front and rear node
        out = np.zeros_like(nodes)
        for f_i, t_i, M_i, d_i, front_i, rear_i in (
            (f_a - G, t_a, M_a, d_a, corner[:, 0], corner[:, 2]),
            (f_b + G, t_b, M_b, d_b, corner[:, 1], corner[:, 3]),
        ):
            couple = np.cross(M_i, d_i) / dot(d_i, d_i)[:, None]
            np.add.at(out, front_i, (1.0 - t_i)[:, None] * f_i - couple)
            np.add.at(out, rear_i, t_i[:, None] * f_i + couple)
        return out


def check_moment_preservation(
    panel_forces: np.ndarray,
    panel_points: np.ndarray,
    nodal_forces: np.ndarray,
    nodes: np.ndarray,
    ref_point: np.ndarray | None = None,
    panel_moments: np.ndarray | None = None,
) -> dict:
    """Report force and moment errors introduced by aero-to-structure mapping.

    ``panel_moments``: each panel's own moment about its ``panel_points``
    entry, part of the aerodynamic moment the nodes must reproduce.
    """
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
    if panel_moments is not None:
        moment_panel += np.asarray(panel_moments, dtype=float).sum(axis=0)

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
    "StrutLeverAeroToStructuralLoadMapper",
    "check_moment_preservation",
    "interpolate_points",
]

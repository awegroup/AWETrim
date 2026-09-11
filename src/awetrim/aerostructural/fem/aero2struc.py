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

import logging
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from awetrim.plotting import (
    plot_aerodynamic_forces_chordwise_distributed,
)


def check_moment_preservation(
    f_aero_panel: np.ndarray,
    panel_cps: np.ndarray,
    f_aero_mapped: np.ndarray,
    struc_nodes: np.ndarray,
    ref_point: np.ndarray = None,
) -> dict:
    """
    Check whether the aero→struc force mapping preserves total force and moment.

    Args:
        f_aero_panel:  (n_panels, 3) panel forces at their CPs.
        panel_cps:     (n_panels, 3) panel control-point locations.
        f_aero_mapped: (n_struc, 3)  mapped forces on structural nodes.
        struc_nodes:   (n_struc, 3)  structural node positions.
        ref_point:     (3,)  reference point for moment calc. Default: origin.

    Returns:
        dict with force/moment totals, errors, and relative moment error.
    """
    if ref_point is None:
        ref_point = np.zeros(3)

    # --- total force ---
    F_aero = np.sum(f_aero_panel, axis=0)
    F_struc = np.sum(f_aero_mapped, axis=0)
    dF = F_struc - F_aero

    # --- total moment about ref_point ---
    M_aero = np.zeros(3)
    for cp, frc in zip(panel_cps, f_aero_panel):
        M_aero += np.cross(cp - ref_point, frc)

    M_struc = np.zeros(3)
    for node, frc in zip(struc_nodes, f_aero_mapped):
        M_struc += np.cross(node - ref_point, frc)

    dM = M_struc - M_aero
    M_aero_norm = np.linalg.norm(M_aero)
    dM_rel = np.linalg.norm(dM) / M_aero_norm if M_aero_norm > 1e-12 else 0.0

    result = {
        "F_aero_total": F_aero,
        "F_struc_total": F_struc,
        "dF": dF,
        "dF_norm": np.linalg.norm(dF),
        "M_aero": M_aero,
        "M_struc": M_struc,
        "dM": dM,
        "dM_norm": np.linalg.norm(dM),
        "dM_rel": dM_rel,
    }

    logging.info(
        f"Moment preservation check (ref={ref_point}):\n"
        f"  Force error  ||dF|| = {result['dF_norm']:.6e} N\n"
        f"  Moment aero  ||M||  = {M_aero_norm:.3f} Nm\n"
        f"  Moment error ||dM|| = {result['dM_norm']:.6e} Nm  "
        f"(relative: {result['dM_rel']:.4%})\n"
        f"  dM components = [{dM[0]:.4f}, {dM[1]:.4f}, {dM[2]:.4f}] Nm"
    )

    return result


def build_ordered_sections(struc_nodes, canopy_sections, strut_sections):
    """
    Build spanwise-ordered section lists and their coordinates.

    Args:
        struc_nodes (np.ndarray): Structural node positions (n_nodes, 3).
        canopy_sections (list[list[int]]): Chordwise node indices per canopy section.
        strut_sections (list[list[int]]): Chordwise node indices per strut section.

    Returns:
        tuple: (sections, section_coords)
            - sections: list of index lists ordered by LE y-coordinate.
            - section_coords: list of arrays with coordinates for each section.
    """
    sections = canopy_sections + strut_sections
    # sort by LE y (or by section[0] if that encodes span order)
    sections = sorted(sections, key=lambda sec: struc_nodes[sec[0]][1])
    # return indices + coords
    section_coords = [struc_nodes[np.array(sec)] for sec in sections]
    return sections, section_coords


def map_aero_forces_to_struct_nodes(aero_points, aero_forces, struc_nodes, sections):
    """
    Map distributed aerodynamic forces onto structural beam nodes using
    bilinear interpolation across two spanwise-bracketing sections.

    For each aerodynamic force application point:
      1. Find the two sections that bracket it in spanwise (y) direction.
      2. Compute spanwise weight eta in [0, 1] between those sections.
      3. Within each section, find the closest beam segment and project
         onto it → chordwise weight xi in [0, 1].
      4. Distribute force to the resulting 4 nodes (2 per section) via
         combined weights:  w_section * (1-xi) and w_section * xi.

    Symmetry preservation: because sections are sorted by y and bracketing
    is deterministic, mirror-symmetric aero points receive mirror-symmetric
    weights, so symmetric loads produce exactly symmetric nodal forces.

    Force is exactly preserved.  Moment error is proportional to the
    out-of-plane offset between the aero application point and the beam
    segments (same as the single-segment version).

    Args:
        aero_points (np.ndarray): (N_aero, 3) coordinates where forces act.
        aero_forces (np.ndarray): (N_aero, 3) force vectors.
        struc_nodes (np.ndarray): (N_struc, 3) structural node coordinates.
        sections (list[list[int]]): Each element is a list of node indices
            forming a chordwise beam (consecutive pairs are segments).
            Must be sorted by ascending LE y-coordinate (as returned by
            build_ordered_sections).

    Returns:
        np.ndarray: (N_struc, 3) lumped force vectors at structural nodes.
    """
    aero_points = np.asarray(aero_points, dtype=float)
    aero_forces = np.asarray(aero_forces, dtype=float)
    n_struc = len(struc_nodes)
    nodal_forces = np.zeros((n_struc, 3), dtype=float)

    n_sections = len(sections)
    if n_sections == 0:
        return nodal_forces

    # LE y-coordinate of each section (sections are assumed sorted by y)
    section_y = np.array([struc_nodes[sec[0]][1] for sec in sections])

    # Pre-build segments per section for fast lookup
    section_segments = []
    for sec in sections:
        segs = []
        for j in range(len(sec) - 1):
            idx_a, idx_b = sec[j], sec[j + 1]
            p_a = struc_nodes[idx_a]
            seg_vec = struc_nodes[idx_b] - p_a
            seg_len_sq = np.dot(seg_vec, seg_vec)
            segs.append((idx_a, idx_b, p_a, seg_vec, seg_len_sq))
        section_segments.append(segs)

    for k in range(len(aero_points)):
        p = aero_points[k]
        f = aero_forces[k]
        y_p = p[1]

        # --- spanwise bracketing ---
        right = int(np.searchsorted(section_y, y_p))
        left = right - 1

        if right >= n_sections:
            right = n_sections - 1
            left = right - 1 if right > 0 else right
        if left < 0:
            left = 0
            right = 1 if n_sections > 1 else 0

        if left == right:
            # single section available
            pairs = [(left, 1.0)]
        else:
            dy = section_y[right] - section_y[left]
            if dy < 1e-30:
                eta = 0.5
            else:
                eta = (y_p - section_y[left]) / dy
            eta = max(0.0, min(1.0, eta))
            pairs = [(left, 1.0 - eta), (right, eta)]

        # --- distribute to both sections ---
        for s_idx, s_w in pairs:
            if s_w < 1e-15:
                continue
            segs = section_segments[s_idx]
            if len(segs) == 0:
                # section has only 1 node → lump everything there
                nodal_forces[sections[s_idx][0]] += s_w * f
                continue

            # find closest segment in this section
            best_dist_sq = np.inf
            best_idx_a = 0
            best_idx_b = 0
            best_xi = 0.0

            for idx_a, idx_b, p_a, seg_vec, seg_len_sq in segs:
                if seg_len_sq < 1e-30:
                    xi = 0.0
                else:
                    xi = np.dot(p - p_a, seg_vec) / seg_len_sq
                xi_c = max(0.0, min(1.0, xi))
                proj = p_a + xi_c * seg_vec
                d_sq = np.sum((p - proj) ** 2)

                if d_sq < best_dist_sq:
                    best_dist_sq = d_sq
                    best_idx_a = idx_a
                    best_idx_b = idx_b
                    best_xi = xi_c

            nodal_forces[best_idx_a] += s_w * (1.0 - best_xi) * f
            nodal_forces[best_idx_b] += s_w * best_xi * f

    return nodal_forces


def consistent_chordwise_density(weights, stations):
    """Piecewise-linear chordwise density whose consistent nodal loads are ``weights``.

    The aerodynamic side hands over a DISCRETE chordwise distribution: weights
    ``w_j`` at stations ``xi_j``. To integrate a load over structural elements it
    has to be a FUNCTION of ``xi``. This returns the nodal values ``q_j`` of the
    piecewise-linear ``q(xi) = sum_j q_j N_j(xi)`` whose consistent loads
    ``int q N_i dxi`` reproduce ``w`` exactly, i.e. ``q = M^-1 w`` with ``M`` the
    one-dimensional consistent mass matrix of the station mesh.

    That choice is what makes the transfer exact at the panel level. It keeps
    the zeroth moment, ``int q = sum_j w_j``, because the hats are a partition of
    unity; and it keeps the first, ``int xi q = sum_j xi_j w_j``, because ``xi``
    is itself linear on the mesh, ``xi = sum_j xi_j N_j``. So the panel's force
    AND its centre of pressure survive the reconstruction. A lumped (row-sum)
    reconstruction keeps only the first: it moves each end station's load a
    third of a bin inward, which on a suction peak at the leading edge is a
    moment the aerodynamics never produced.

    Args:
        weights: ``(n_stations,)`` or ``(n_panels, n_stations)``.
        stations: ``(n_stations,)`` increasing chordwise fractions.

    Returns:
        Array shaped like ``weights``: the density's nodal values.
    """
    stations = np.asarray(stations, dtype=float)
    weights = np.asarray(weights, dtype=float)
    n_stations = len(stations)
    mass = np.zeros((n_stations, n_stations))
    for element, length in enumerate(np.diff(stations)):
        mass[element:element + 2, element:element + 2] += (
            length / 6.0 * np.array([[2.0, 1.0], [1.0, 2.0]])
        )
    return np.linalg.solve(mass, weights.T).T


def canopy_surface_coordinates(struc_nodes, grid, triangles):
    """Surface coordinates ``(s, xi)`` of every canopy node.

    ``s`` is the canopy grid's row index, the spanwise coordinate in which the
    aerodynamic sections are laid out: they are linear subdivisions between
    consecutive structural sections, so aero section ``m`` sits at exactly
    ``s = m (rows - 1) / n_panels`` at any refinement.

    ``xi`` is the chordwise fraction: the projection onto that row's own
    leading-edge-to-trailing-edge chord line. That is how the aerodynamic side
    places its chordwise stations, so a load meant for ``xi`` lands on the
    surface point directly above the chord-line point it was computed at --
    offset along the section normal, roughly along the load itself, which is
    why this costs almost no moment. It is recomputed from the CURRENT shape,
    so a node that slides chordwise as the canopy billows carries the load for
    where it now is.

    Nodes off the grid -- the ``cross`` pattern's quad centres -- take the mean
    of the grid nodes they share an element with: the material point at the
    centre of the quad.
    """
    struc_nodes = np.asarray(struc_nodes, dtype=float)
    grid = np.asarray(grid, dtype=int)
    triangles = np.asarray(triangles, dtype=int)
    coordinates = np.full((len(struc_nodes), 2), np.nan)
    for row_index, row in enumerate(grid):
        leading, trailing = struc_nodes[row[0]], struc_nodes[row[-1]]
        chord = trailing - leading
        coordinates[row, 0] = row_index
        coordinates[row, 1] = (struc_nodes[row] - leading) @ chord / float(chord @ chord)

    missing = np.isnan(coordinates[:, 0])
    if missing[triangles].any():
        sums = np.zeros((len(struc_nodes), 2))
        counts = np.zeros(len(struc_nodes))
        corner_missing = missing[triangles]
        for a in range(3):
            for b in range(3):
                if a == b:
                    continue
                take = corner_missing[:, a] & ~corner_missing[:, b]
                np.add.at(sums, triangles[take, a], coordinates[triangles[take, b]])
                np.add.at(counts, triangles[take, a], 1.0)
        fill = missing & (counts > 0)
        coordinates[fill] = sums[fill] / counts[fill, None]
    return coordinates


def _clip_to_box(polygon, s_low, s_high, xi_low, xi_high):
    """Sutherland-Hodgman clip of a convex polygon to an axis-aligned box."""
    for axis, bound, keep_above in (
        (0, s_low, True), (0, s_high, False), (1, xi_low, True), (1, xi_high, False)
    ):
        if not polygon:
            return polygon
        clipped = []
        previous = polygon[-1]
        previous_in = (previous[axis] >= bound) if keep_above else (previous[axis] <= bound)
        for current in polygon:
            current_in = (current[axis] >= bound) if keep_above else (current[axis] <= bound)
            if current_in != previous_in:
                t = (bound - previous[axis]) / (current[axis] - previous[axis])
                clipped.append(previous + t * (current - previous))
            if current_in:
                clipped.append(current)
            previous, previous_in = current, current_in
        polygon = clipped
    return polygon


def map_aero_traction_to_membrane(panel_forces, panel_weights, stations,
                                  struc_nodes, grid, triangles):
    """Consistent nodal loads from the aerodynamic TRACTION over every canopy element.

    The aerodynamic result is a load distributed over the canopy surface, not a
    set of point loads, and a canopy node should carry what the elements around
    it carry. Point-load transfers -- to node chains, or to the nearest element
    -- cannot do that: each point reaches at most three nodes, so once the
    structural mesh is finer than the aerodynamic sampling the total is right
    but it lands on a sparse subset of nodes (measured: 61% of a refined
    canopy unloaded).

    This builds the traction as a field and integrates it. On the surface
    coordinates of ``canopy_surface_coordinates``, panel ``k`` covers the strip
    ``s_k <= s <= s_k+1`` and carries

        tau(s, xi) = F_k q_k(xi) / (s_k+1 - s_k),

    uniform along the span of its strip -- which is what the lifting line
    computes -- and chordwise the piecewise-linear density of
    ``consistent_chordwise_density``. Each element then receives the standard
    consistent load vector

        f_a = int_element tau N_a dA,

    with ``N_a`` its linear shape functions, so every canopy node is loaded by
    every element adjacent to it, at any refinement.

    The integral is evaluated EXACTLY. Each element is clipped against the
    strip and chord-interval boxes it overlaps; on each piece the integrand is
    a product of two linear functions, which the three-edge-midpoint rule
    integrates exactly. Because the elements tile the surface coordinates and
    the strips tile them too, the nodal loads sum to ``sum_k F_k`` to roundoff:
    the transfer is conservative by construction, not by calibration.

    Force is therefore exact. The moment differs from the one the aerodynamic
    side would compute with its loads on the chord LINE by exactly the load's
    offset from that line to the canopy SURFACE -- the pressure acts on the
    surface, so that difference is the point, not an error.

    Args:
        panel_forces: ``(n_panels, 3)``, ordered along the grid rows.
        panel_weights: ``(n_panels, n_stations)`` chordwise weights per panel.
        stations: ``(n_stations,)`` chordwise fractions of those weights.
        struc_nodes: ``(n_nodes, 3)`` current node positions.
        grid: ``(rows, columns)`` canopy grid, rows spanning leading to
            trailing edge in the same order as the aerodynamic sections.
        triangles: ``(n_elements, 3)`` canopy elements.

    Returns:
        ``(n_nodes, 3)`` nodal forces.
    """
    panel_forces = np.asarray(panel_forces, dtype=float)
    stations = np.asarray(stations, dtype=float)
    grid = np.asarray(grid, dtype=int)
    triangles = np.asarray(triangles, dtype=int)
    n_panels = len(panel_forces)
    density = consistent_chordwise_density(panel_weights, stations)
    coordinates = canopy_surface_coordinates(struc_nodes, grid, triangles)
    span_edges = np.linspace(0.0, grid.shape[0] - 1.0, n_panels + 1)
    widths = np.diff(span_edges)
    midpoint_pairs = ((0, 1), (1, 2), (2, 0))

    nodal = np.zeros((len(struc_nodes), 3))
    covered = 0.0
    for element in triangles:
        corners = coordinates[element]
        edge_1, edge_2 = corners[1] - corners[0], corners[2] - corners[0]
        twice_area = edge_1[0] * edge_2[1] - edge_1[1] * edge_2[0]
        if twice_area == 0.0:
            continue
        covered += 0.5 * abs(twice_area)
        inverse = np.linalg.inv(np.array([edge_1, edge_2]).T)

        low, high = corners.min(axis=0), corners.max(axis=0)
        first_panel = max(int(np.searchsorted(span_edges, low[0], "right")) - 1, 0)
        last_panel = min(int(np.searchsorted(span_edges, high[0], "left")), n_panels)
        first_bin = max(int(np.searchsorted(stations, low[1], "right")) - 1, 0)
        last_bin = min(int(np.searchsorted(stations, high[1], "left")), len(stations) - 1)

        loads = np.zeros((3, 3))
        for k in range(first_panel, last_panel):
            share = np.zeros(3)
            for j in range(first_bin, last_bin):
                piece = _clip_to_box(
                    [corners[0], corners[1], corners[2]],
                    span_edges[k], span_edges[k + 1], stations[j], stations[j + 1],
                )
                if len(piece) < 3:
                    continue
                bin_width = stations[j + 1] - stations[j]
                for fan in range(1, len(piece) - 1):
                    vertices = (piece[0], piece[fan], piece[fan + 1])
                    a, b = vertices[1] - vertices[0], vertices[2] - vertices[0]
                    area = 0.5 * abs(a[0] * b[1] - a[1] * b[0])
                    if area == 0.0:
                        continue
                    for p, q in midpoint_pairs:
                        point = 0.5 * (vertices[p] + vertices[q])
                        local = inverse @ (point - corners[0])
                        shape = np.array([1.0 - local[0] - local[1], local[0], local[1]])
                        t = (point[1] - stations[j]) / bin_width
                        value = (1.0 - t) * density[k, j] + t * density[k, j + 1]
                        share += (area / 3.0) * value * shape
            loads += share[:, None] * (panel_forces[k] / widths[k])[None, :]
        np.add.at(nodal, element, loads)

    # The elements must tile the surface once. A pattern that covers it twice
    # (``union`` superposes two sheets) would double the load; a gap or a fold
    # in the surface coordinates would lose or double part of it. Say which.
    domain = float(grid.shape[0] - 1)
    multiplicity = covered / domain
    sheets = max(int(round(multiplicity)), 1)
    if abs(multiplicity - sheets) > 1e-9:
        logging.warning(
            "canopy elements cover %.9f of the wing surface (expected a whole "
            "number of sheets): a gap or a fold in the surface coordinates; "
            "the traction transfer is not conservative by %.3e of the load",
            multiplicity, abs(multiplicity - sheets) / sheets,
        )
    return nodal / sheets


def _panels_follow_grid(panels, struc_nodes, grid):
    """True if panel 0 sits at grid row 0, False if the span runs backwards."""
    first = 0.5 * (panels[0].LE_point_1 + panels[0].LE_point_2)
    struc_nodes = np.asarray(struc_nodes, dtype=float)
    return (np.linalg.norm(first - struc_nodes[grid[0, 0]])
            <= np.linalg.norm(first - struc_nodes[grid[-1, 0]]))

def map_aero_forces_to_membrane(aero_points, aero_forces, struc_nodes, triangles):
    """Distribute aerodynamic point loads through the canopy ELEMENTS.

    The lattice mapping above routes every load to the chordwise node chains
    read from the YAML, so a node that is not on one of those chains receives
    nothing however much canopy surrounds it. That is wrong in principle -- a
    canopy node should be loaded by the elements adjacent to it, and the
    pressure it carries is a property of that surface, not of the bookkeeping
    used to list the nodes -- and it is wrong in practice for any mesh richer
    than the YAML's: the ``cross`` pattern's quad-centre nodes and every
    interior node added by refinement are held by the membrane alone.

    This routes the load the way the surface does. Each application point is
    assigned to the canopy triangle whose closest point to it is nearest, and
    its force is split over that triangle's three vertices by the barycentric
    coordinates of that closest point.

    Barycentric rather than a third each: the weights then reproduce the point's
    LINE OF ACTION within the element, so the in-plane moment about the element
    is preserved exactly rather than smeared to the centroid. What remains is
    the out-of-plane offset between the centre of pressure and the membrane
    surface, which no nodal lumping can represent -- the same residual the
    lattice mapping has, minus the in-plane part.

    Symmetry: the closest-point test and the barycentric weights are both
    isometry-invariant, so a mirrored point on a mirrored triangle gets the same
    weights. Ties are broken on the lowest triangle index, which is deterministic
    but NOT mirror-invariant, so exact ties would break symmetry; they require a
    point equidistant from two triangles to the last bit and have not been
    observed. Force is exactly preserved: the three weights sum to one.

    Args:
        aero_points (np.ndarray): (N_aero, 3) coordinates where forces act.
        aero_forces (np.ndarray): (N_aero, 3) force vectors.
        struc_nodes (np.ndarray): (N_struc, 3) structural node positions.
        triangles (np.ndarray): (N_tri, 3) canopy element connectivity.

    Returns:
        np.ndarray: (N_struc, 3) lumped force vectors at structural nodes.
    """
    aero_points = np.asarray(aero_points, dtype=float)
    aero_forces = np.asarray(aero_forces, dtype=float)
    struc_nodes = np.asarray(struc_nodes, dtype=float)
    triangles = np.asarray(triangles, dtype=int)

    nodal_forces = np.zeros_like(struc_nodes)
    if len(triangles) == 0 or len(aero_points) == 0:
        return nodal_forces

    a = struc_nodes[triangles[:, 0]]
    b = struc_nodes[triangles[:, 1]]
    c = struc_nodes[triangles[:, 2]]

    for point, force in zip(aero_points, aero_forces):
        weights, index = _closest_barycentric(point, a, b, c)
        for corner in range(3):
            nodal_forces[triangles[index, corner]] += weights[corner] * force
    return nodal_forces


def _closest_barycentric(point, a, b, c):
    """Barycentric weights of ``point`` on the nearest of the triangles ``abc``.

    Ericson's closest-point-on-triangle, vectorised over the whole element set:
    the seven Voronoi regions of a triangle (three vertices, three edges, the
    interior) are evaluated for every triangle at once and the nearest is taken.
    Returns ``(weights, triangle index)``.
    """
    ab, ac, ap = b - a, c - a, point - a
    d1 = np.einsum("ij,ij->i", ab, ap)
    d2 = np.einsum("ij,ij->i", ac, ap)
    bp = point - b
    d3 = np.einsum("ij,ij->i", ab, bp)
    d4 = np.einsum("ij,ij->i", ac, bp)
    cp = point - c
    d5 = np.einsum("ij,ij->i", ab, cp)
    d6 = np.einsum("ij,ij->i", ac, cp)

    # Interior of the triangle: the plane projection, in barycentric form.
    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2
    denom = va + vb + vc
    safe = np.where(np.abs(denom) < 1e-300, 1.0, denom)
    v, w = vb / safe, vc / safe
    u = 1.0 - v - w
    bary = np.stack([u, v, w], axis=1)

    # Degenerate or outside: clamp onto the nearest vertex or edge.
    outside = (np.abs(denom) < 1e-300) | (u < 0) | (v < 0) | (w < 0)
    if outside.any():
        i = np.flatnonzero(outside)
        clamped = np.zeros((len(i), 3))
        # vertex regions
        at_a = (d1[i] <= 0) & (d2[i] <= 0)
        at_b = (d3[i] >= 0) & (d4[i] <= d3[i])
        at_c = (d6[i] >= 0) & (d5[i] <= d6[i])
        clamped[at_a] = (1.0, 0.0, 0.0)
        clamped[at_b & ~at_a] = (0.0, 1.0, 0.0)
        clamped[at_c & ~at_a & ~at_b] = (0.0, 0.0, 1.0)
        done = at_a | at_b | at_c
        # edge regions
        for mask, num, den, slot in (
            (vc[i] <= 0, d1[i], d1[i] - d3[i], (0, 1)),
            (vb[i] <= 0, d2[i], d2[i] - d6[i], (0, 2)),
            (va[i] <= 0, d4[i] - d3[i], (d4[i] - d3[i]) + (d5[i] - d6[i]), (1, 2)),
        ):
            take = mask & ~done
            if take.any():
                t = np.clip(np.divide(num[take], den[take],
                                      out=np.zeros(int(take.sum())),
                                      where=np.abs(den[take]) > 1e-300), 0.0, 1.0)
                clamped[take] = 0.0
                clamped[np.flatnonzero(take), slot[0]] = 1.0 - t
                clamped[np.flatnonzero(take), slot[1]] = t
                done |= take
        # anything left (a fully degenerate element) goes to its first vertex
        clamped[~done] = (1.0, 0.0, 0.0)
        bary[i] = clamped

    closest = (bary[:, :1] * a + bary[:, 1:2] * b + bary[:, 2:3] * c)
    index = int(np.argmin(np.einsum("ij,ij->i", closest - point, closest - point)))
    return bary[index], index

def verify_force_moment_conservation(
    aero_points, aero_forces, struc_nodes, nodal_forces, ref_point=None
):
    """
    Print a check of total force and moment conservation.

    Args:
        aero_points (np.ndarray): (N_aero, 3)
        aero_forces (np.ndarray): (N_aero, 3)
        struc_nodes (np.ndarray): (N_struc, 3)
        nodal_forces (np.ndarray): (N_struc, 3)
        ref_point (np.ndarray, optional): (3,) reference for moment. Default: origin.
    """
    if ref_point is None:
        ref_point = np.zeros(3)

    F_aero = np.sum(aero_forces, axis=0)
    F_struc = np.sum(nodal_forces, axis=0)
    M_aero = np.sum(np.cross(aero_points - ref_point, aero_forces), axis=0)
    M_struc = np.sum(np.cross(struc_nodes - ref_point, nodal_forces), axis=0)

    print("=== Force & Moment Conservation Check ===")
    print(f"  Total aero force:    {F_aero}")
    print(f"  Total struct force:  {F_struc}")
    print(f"  Force error:         {np.linalg.norm(F_struc - F_aero):.2e}")
    print(f"  Total aero moment:   {M_aero}")
    print(f"  Total struct moment: {M_struc}")
    print(f"  Moment error:        {np.linalg.norm(M_struc - M_aero):.2e}")
    print("==========================================")


def _load_cp_distribution(cp_path):
    """
    Load Cp distribution data from a file.

    Args:
        cp_path (str or Path): Path to Cp file with columns: x y Cp.

    Returns:
        tuple: (x, y, cp) arrays.
    """
    rows = []
    with open(cp_path, "r") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) < 3:
                continue
            rows.append([float(parts[0]), float(parts[1]), float(parts[2])])
    data = np.array(rows, dtype=float)
    if data.size == 0:
        raise ValueError(f"No Cp data found in {cp_path}")
    return data[:, 0], data[:, 1], data[:, 2]


def _average_duplicate_x(x, cp):
    """
    Average Cp values for duplicate x positions.

    Args:
        x (np.ndarray): Chordwise positions.
        cp (np.ndarray): Cp values at those positions.

    Returns:
        tuple: (uniq_x, cp_mean) with averaged Cp per unique x.
    """
    order = np.argsort(x)
    x_sorted = x[order]
    cp_sorted = cp[order]
    uniq_x, inv = np.unique(x_sorted, return_inverse=True)
    cp_sum = np.zeros_like(uniq_x, dtype=float)
    counts = np.zeros_like(uniq_x, dtype=float)
    np.add.at(cp_sum, inv, cp_sorted)
    np.add.at(counts, inv, 1.0)
    cp_mean = cp_sum / np.maximum(counts, 1.0)
    return uniq_x, cp_mean


def _split_surfaces_by_order(x, y, cp):
    """
    Split Cp data into upper and lower surfaces using file order.

    Assumes the data is ordered around the airfoil (LE -> TE -> LE). The split
    is performed at the first crossing where x >= 1.0 (TE). The first segment
    is treated as upper, the second as lower. Only x-ordering is used.

    Args:
        x (np.ndarray): Chordwise positions.
        y (np.ndarray): Surface-normal positions.
        cp (np.ndarray): Cp values.

    Returns:
        tuple: (x_u, y_u, cp_u), (x_l, y_l, cp_l)
    """
    x = np.asarray(x)
    y = np.asarray(y)
    cp = np.asarray(cp)

    te_candidates = np.where(x >= 1.0)[0]
    if te_candidates.size > 0:
        te_idx = int(te_candidates[0])
    else:
        te_idx = int(np.argmax(x))

    if te_idx == 0 or te_idx >= len(x) - 1:
        mid = len(x) // 2
        return (x[:mid], y[:mid], cp[:mid]), (x[mid:], y[mid:], cp[mid:])

    x_u, y_u, cp_u = x[: te_idx + 1], y[: te_idx + 1], cp[: te_idx + 1]
    x_l, y_l, cp_l = x[te_idx:], y[te_idx:], cp[te_idx:]
    return (x_u, y_u, cp_u), (x_l, y_l, cp_l)


def weights_at_centre_of_pressure(
    base_weights: np.ndarray,
    stations: np.ndarray,
    target: float,
    *,
    margin: float = 1e-3,
    tolerance: float = 1e-12,
    max_iterations: int = 60,
) -> np.ndarray:
    """Re-weight a chordwise distribution so its centroid lands on ``target``.

    The chordwise weights decide where along the chord a panel's force ends up
    acting, and that station *is* the panel's local pitching moment. A fixed
    weight set therefore imposes one fixed local ``C_m`` on every panel at every
    angle of attack, which is wrong in general and wrong in a way the coupled
    loop cannot settle: the wing's trim is a moment balance, so each outer
    iteration re-trims against a moment the structure was never given.

    ``base_weights`` is the prior -- the measured ``Delta C_p`` shape -- and
    ``target`` the chordwise fraction the panel's force must act at to
    reproduce its own moment. The correction is the exponential tilt

        w_i  =  w0_i exp(lambda t_i) / sum_j w0_j exp(lambda t_j)

    with ``lambda`` solved from ``sum_i w_i t_i = target``. Three properties
    make this the right choice over, say, a least-squares correction:

    * every weight stays **strictly positive**, so the distribution never
      develops an unphysical negative patch of load;
    * the weights still sum to one, so the panel force is preserved exactly;
    * ``lambda`` is the natural parameter of an exponential family, whose mean
      is strictly increasing in it, so the root exists, is unique for any
      target strictly inside the station range, and Newton converges from
      ``lambda = 0`` (the derivative is a variance, hence positive).

    It is also the minimum-relative-entropy correction: of all distributions
    with the required mean, this is the one closest to the measured shape, so
    the ``Delta C_p`` prior is kept wherever the moment does not contradict it.
    """
    base = np.asarray(base_weights, dtype=float)
    stations = np.asarray(stations, dtype=float)
    total = base.sum()
    if total <= 0.0:
        raise ValueError("base_weights must have a positive sum")
    base = base / total

    support = base > 0.0
    lowest, highest = stations[support].min(), stations[support].max()
    span = highest - lowest
    # lambda diverges as the target approaches the ends of the support, so hold
    # it just inside. VSM already clamps the centre of pressure to [LE, TE], so
    # this only bites on a panel whose moment puts the load right at an edge.
    target = float(np.clip(target, lowest + margin * span, highest - margin * span))

    tilt = 0.0
    for _ in range(max_iterations):
        weights = base * np.exp(tilt * (stations - target))
        weights /= weights.sum()
        mean = float(weights @ stations)
        residual = mean - target
        if abs(residual) <= tolerance * max(span, 1.0):
            return weights
        variance = float(weights @ (stations - mean) ** 2)
        if variance <= 1e-30:
            break
        tilt -= residual / variance
    logging.warning(
        "chordwise centre-of-pressure tilt did not converge (target %.4f, "
        "reached %.4f); using the untilted weights",
        target,
        float(base @ stations),
    )
    return base


def chordwise_weights_from_cp_file(cp_path, x_targets):
    """
    Compute normalized chordwise weights from a Cp distribution file.

    The weights are based on Delta Cp = Cp_lower - Cp_upper interpolated
    onto x_targets. If Delta Cp is invalid, returns uniform weights.

    Args:
        cp_path (str or Path): Path to Cp file with columns: x y Cp.
        x_targets (np.ndarray): Chordwise positions in [0, 1] to weight.

    Returns:
        np.ndarray: Normalized weights for each x_target.
    """
    x, y, cp = _load_cp_distribution(cp_path)
    (x_u, y_u, cp_u), (x_l, y_l, cp_l) = _split_surfaces_by_order(x, y, cp)

    if len(x_u) == 0 or len(x_l) == 0:
        return np.full_like(x_targets, 1.0 / len(x_targets), dtype=float)

    x_u, cp_u = _average_duplicate_x(x_u, cp_u)
    x_l, cp_l = _average_duplicate_x(x_l, cp_l)

    if len(x_u) < 2 or len(x_l) < 2:
        return np.full_like(x_targets, 1.0 / len(x_targets), dtype=float)

    cp_upper = np.interp(x_targets, x_u, cp_u, left=cp_u[0], right=cp_u[-1])
    cp_lower = np.interp(x_targets, x_l, cp_l, left=cp_l[0], right=cp_l[-1])

    delta_cp = cp_lower - cp_upper
    delta_cp = np.maximum(delta_cp, 0.0)
    total = np.sum(delta_cp)
    if total <= 0.0:
        return np.full_like(x_targets, 1.0 / len(x_targets), dtype=float)
    return delta_cp / total


def plot_delta_cp_and_weights(cp_path, n_bins=10):
    """
    Plot Cp distributions, Delta Cp, and chordwise weights.

    Args:
        cp_path (str or Path): Path to Cp file with columns: x y Cp.
        n_bins (int): Number of chordwise bins/targets for weights.

    Returns:
        None. Displays a 3x1 plot.
    """
    x, y, cp = _load_cp_distribution(cp_path)
    (x_u, y_u, cp_u), (x_l, y_l, cp_l) = _split_surfaces_by_order(x, y, cp)
    if len(x_u) == 0 or len(x_l) == 0:
        raise ValueError("Cp file must contain both upper and lower surface points.")

    x_u, cp_u = _average_duplicate_x(x_u, cp_u)
    x_l, cp_l = _average_duplicate_x(x_l, cp_l)

    x_full = np.union1d(x_u, x_l)
    cp_upper_full = np.interp(x_full, x_u, cp_u, left=cp_u[0], right=cp_u[-1])
    cp_lower_full = np.interp(x_full, x_l, cp_l, left=cp_l[0], right=cp_l[-1])
    delta_cp = cp_lower_full - cp_upper_full

    x_targets = np.linspace(0.0, 1.0, n_bins)
    weights = chordwise_weights_from_cp_file(cp_path, x_targets)

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8, 10))
    n_steps = 6

    axes[0].plot(x_u, cp_u, "o-", color="tab:blue", label=r"$C_p$ upper", markersize=2)
    axes[0].plot(
        x_l, cp_l, "o-", color="tab:orange", label=r"$C_p$ lower", markersize=2
    )
    axes[0].plot([0, 1], [0, 0], "--", color="k", lw=1)
    axes[0].set_ylabel(r"$C_p$ (-)")
    axes[0].set_title("Cp Distribution")
    axes[0].invert_yaxis()
    axes[0].grid(True, linestyle="--", linewidth=0.5)
    axes[0].legend()

    axes[1].plot(
        x_full, delta_cp, "o-", color="tab:green", label=r"$\Delta C_p$", markersize=2
    )
    bar_width = 0.8 / max(n_bins - 1, 1)
    axes[1].bar(
        x_targets,
        weights,
        width=bar_width,
        color="tab:gray",
        alpha=0.4,
        label="Chordwise weights",
    )
    axes[1].plot([0, 1], [0, 0], "--", color="k", lw=1)
    axes[1].set_ylabel(r"$\Delta C_p$ / weights (-)")
    axes[1].set_title("Delta Cp and Weights")
    axes[1].grid(True, linestyle="--", linewidth=0.5)
    axes[1].legend()

    axes[2].plot(
        x_targets,
        weights,
        "o-",
        color="tab:gray",
        label="Chordwise weights",
        markersize=2,
    )
    axes[2].set_xlabel(r"$x/c$ (-)")
    axes[2].set_ylabel("Weight (-)")
    axes[2].set_title("Weights")
    axes[2].grid(True, linestyle="--", linewidth=0.5)
    axes[2].legend()

    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_xticks(np.linspace(0, 1, n_steps))

    plt.tight_layout()
    plt.show()


def main(
    coupling_method: str,
    f_aero_wing_vsm_format: np.ndarray,
    struc_nodes: np.ndarray,
    panel_cp_locations: np.ndarray,
    aero2struc_mapping: np.ndarray,
    is_with_coupling_plot: bool,
    config_aer2struc: dict,
    canopy_sections,
    strut_sections,
    panels,
    section_ids=None,
    canopy_triangles=None,
    canopy_grid=None,
    cp_distribution_path=None,
    is_with_delta_cp_and_weights_plot=False,
    is_with_conservation_check=False,
    return_distributed_aero=False,
):
    """
    Main interface for mapping aerodynamic panel forces to structural nodes.

    Args:
        coupling_method (str): Coupling method name (e.g., "NN").
        f_aero_wing_vsm_format (np.ndarray): Aerodynamic forces per panel (n_panels,3).
        struc_nodes (np.ndarray): Structural node positions (n_struc,3).
        panel_cp_locations (np.ndarray): Panel control points (n_panels,3).
        aero2struc_mapping (np.ndarray): Mapping from panels to 4 node indices (n_panels,4).
        is_with_coupling_plot (bool): If True, plot the mapping.
        p (float): Power for inverse-distance weighting.
        eps (float): Small value to avoid division by zero.
        cp_distribution_path (str or Path, optional): Path to Cp distribution file
            used for chordwise weighting. Defaults to cp_AOA_8.dat in the data folder.
        is_with_conservation_check (bool): If True, print force/moment conservation
            check for this mapping call.
        return_distributed_aero (bool): If True, also return the chordwise-
            distributed aero points/forces used by the mapping.

    Returns:
        np.ndarray or tuple[np.ndarray, dict]: Mapped forces on structural nodes
            (n_struc,3), and optionally a dict with keys "points" and "forces"
            containing the distributed aero loads used for mapping.
    """
    # Displacing the single spanwise aero force over 10 chordwise nodes
    n_chordwise_nodes = 10
    vsm_wing_nodes_distributed_chordwise = []
    vsm_wing_forces_distributed_chordwise = []

    if cp_distribution_path is None:
        cp_distribution_path = config_aer2struc.get("cp_distribution_path")

    t_vals = np.linspace(0.0, 1.0, n_chordwise_nodes)
    if cp_distribution_path is not None and Path(cp_distribution_path).exists():
        chordwise_weights = chordwise_weights_from_cp_file(cp_distribution_path, t_vals)
    else:
        chordwise_weights = np.full(n_chordwise_nodes, 1.0 / n_chordwise_nodes)

    # Where the chordwise weights put a panel's resultant IS that panel's local
    # pitching moment. "cp_file" uses one measured Delta C_p shape for every
    # panel, which pins the resultant at a single chordwise station (0.291 c on
    # the LEI-V3 file) whatever the panel's own C_m says -- so the structure is
    # loaded with a moment the aerodynamics never produced, and the coupled loop
    # re-trims each iteration against a moment balance it was never given.
    # "moment_matched" keeps that shape as a prior and tilts it onto the panel's
    # own centre of pressure, which VSM already computes from F and M.
    distribution = config_aer2struc.get("chordwise_distribution", "cp_file")
    if distribution not in ("cp_file", "moment_matched"):
        raise ValueError(
            "aero2struc.chordwise_distribution must be 'cp_file' or "
            f"'moment_matched', got {distribution!r}"
        )
    centres_of_pressure = np.asarray(panel_cp_locations, dtype=float)
    if distribution == "moment_matched" and len(centres_of_pressure) != len(panels):
        raise ValueError(
            f"{len(centres_of_pressure)} centres of pressure for {len(panels)} "
            "panels; moment matching needs one per panel"
        )

    panel_weights = []
    for index, (panel, f_panel) in enumerate(zip(panels, f_aero_wing_vsm_format)):
        le_mid = 0.5 * (panel.LE_point_1 + panel.LE_point_2)
        te_mid = 0.5 * (panel.TE_point_1 + panel.TE_point_2)
        vec_chord = te_mid - le_mid

        chord_nodes = le_mid[None, :] + t_vals[:, None] * vec_chord[None, :]
        vsm_wing_nodes_distributed_chordwise.append(chord_nodes)

        weights = chordwise_weights
        if distribution == "moment_matched":
            # Chordwise fraction of the panel's centre of pressure. VSM builds
            # it as ac + lever * y_airf with the lever clamped to [LE, TE], so
            # this lands in [0, 1] on a well-posed panel.
            chord_squared = float(vec_chord @ vec_chord)
            if chord_squared > 1e-24:
                t_cp = float((centres_of_pressure[index] - le_mid) @ vec_chord)
                weights = weights_at_centre_of_pressure(
                    chordwise_weights, t_vals, t_cp / chord_squared
                )

        panel_weights.append(weights)
        f_nodes = weights[:, None] * f_panel[None, :]
        vsm_wing_forces_distributed_chordwise.append(f_nodes)

    vsm_wing_nodes_distributed_chordwise = np.vstack(
        vsm_wing_nodes_distributed_chordwise
    )
    vsm_wing_forces_distributed_chordwise = np.vstack(
        vsm_wing_forces_distributed_chordwise
    )

    if is_with_delta_cp_and_weights_plot:
        plot_delta_cp_and_weights(
            Path("data/TUDELFT_V3_KITE/cp_distributions/cp_AOA_8.dat"),
            n_bins=10,
        )

    # mapping the distributed aerodynamic forces to the structural nodes
    # Element-consistent transfer when the caller knows the canopy elements.
    # The lattice path below can only reach nodes that sit on a chordwise chain
    # from the YAML, so on any mesh richer than that one -- a quad-centre node,
    # a refined interior node -- it silently leaves nodes unloaded. Both routes
    # conserve the resultant force exactly; this one also keeps the in-plane
    # moment, because barycentric weights reproduce the load's line of action
    # inside the element instead of smearing it to the chain.
    #
    #   traction          the aerodynamic load as a field over the canopy,
    #                     integrated over every element (the default whenever
    #                     the elements are known): every canopy node is loaded
    #                     by the elements around it, and the force is exact.
    #   nearest_element   each point load through its nearest element: exact
    #                     force and in-plane moment, but a point reaches only
    #                     three nodes, so a fine mesh is mostly left unloaded.
    #   sections          the lattice below.
    route = str(config_aer2struc.get("load_transfer", "traction"))
    if route not in ("traction", "nearest_element", "sections"):
        raise ValueError(
            "aero2struc.load_transfer must be 'traction', 'nearest_element' or "
            f"'sections', got {route!r}"
        )
    if canopy_triangles is not None and route != "sections":
        if route == "traction":
            if canopy_grid is None:
                raise ValueError("the traction transfer needs the canopy grid")
            forces_k = np.asarray(f_aero_wing_vsm_format, dtype=float)
            weights_k = np.asarray(panel_weights, dtype=float)
            if not _panels_follow_grid(panels, struc_nodes, canopy_grid):
                forces_k, weights_k = forces_k[::-1], weights_k[::-1]
            f_aero_wing = map_aero_traction_to_membrane(
                forces_k, weights_k, t_vals, struc_nodes, canopy_grid,
                canopy_triangles,
            )
        else:
            f_aero_wing = map_aero_forces_to_membrane(
                aero_points=vsm_wing_nodes_distributed_chordwise,
                aero_forces=vsm_wing_forces_distributed_chordwise,
                struc_nodes=struc_nodes,
                triangles=canopy_triangles,
            )
        if is_with_conservation_check:
            verify_force_moment_conservation(
                aero_points=vsm_wing_nodes_distributed_chordwise,
                aero_forces=vsm_wing_forces_distributed_chordwise,
                struc_nodes=struc_nodes,
                nodal_forces=f_aero_wing,
            )
        if return_distributed_aero:
            return f_aero_wing, {
                "points": vsm_wing_nodes_distributed_chordwise,
                "forces": vsm_wing_forces_distributed_chordwise,
            }
        return f_aero_wing

    sections = build_ordered_sections(struc_nodes, canopy_sections, strut_sections)[0]

    if section_ids is None:
        active_sections = sections
    elif isinstance(section_ids, int):
        active_sections = [sections[section_ids]]
    else:
        active_sections = [sections[sid] for sid in section_ids]

    f_aero_wing = map_aero_forces_to_struct_nodes(
        aero_points=vsm_wing_nodes_distributed_chordwise,
        aero_forces=vsm_wing_forces_distributed_chordwise,
        struc_nodes=struc_nodes,
        sections=active_sections,
    )

    if is_with_conservation_check:
        verify_force_moment_conservation(
            aero_points=vsm_wing_nodes_distributed_chordwise,
            aero_forces=vsm_wing_forces_distributed_chordwise,
            struc_nodes=struc_nodes,
            nodal_forces=f_aero_wing,
        )
    if is_with_coupling_plot:
        plot_aerodynamic_forces_chordwise_distributed(
            panel_cps=panel_cp_locations,
            f_aero_chordwise=f_aero_wing_vsm_format,
            vsm_wing_nodes_distributed_chordwise=vsm_wing_nodes_distributed_chordwise,
            vsm_wing_forces_distributed_chordwise=vsm_wing_forces_distributed_chordwise,
            nodes_struc=struc_nodes,
            force_struc=f_aero_wing,
        )
    if return_distributed_aero:
        return f_aero_wing, {
            "points": vsm_wing_nodes_distributed_chordwise,
            "forces": vsm_wing_forces_distributed_chordwise,
        }
    return f_aero_wing


def initialize_mapping(
    panels: np.ndarray,
    struc_nodes: np.ndarray,
    struc_node_le_indices: np.ndarray,
    struc_node_te_indices: np.ndarray,
) -> np.ndarray:
    """
    For each panel CP, find the two LE and two TE structural‐node indices
    whose y-coordinates bracket the CP’s y. Returns an (n_panels, 4) array
    of [le_lo, le_hi, te_lo, te_hi].

    Args:
        panels (np.ndarray): Array of panel objects with .aerodynamic_center attribute.
        struc_nodes (np.ndarray): Structural node positions (n_nodes,3).
        struc_node_le_indices (np.ndarray): Indices of leading edge nodes.
        struc_node_te_indices (np.ndarray): Indices of trailing edge nodes.

    Returns:
        np.ndarray: Mapping array (n_panels, 4).
    """

    # extract and sort LE candidates by their y
    le_coords = []
    for struc_node_le_idx in struc_node_le_indices:
        le_coords.append(struc_nodes[struc_node_le_idx])

    le_coords = np.array(le_coords)
    le_y = le_coords[:, 1]
    le_order = np.argsort(le_y)
    le_sorted_idx = np.array(struc_node_le_indices)[le_order]
    le_sorted_y = le_y[le_order]

    # same for TE
    te_coords = []
    for struc_node_te_idx in struc_node_te_indices:
        te_coords.append(struc_nodes[struc_node_te_idx])

    te_coords = np.array(te_coords)
    te_y = te_coords[:, 1]
    te_order = np.argsort(te_y)
    te_sorted_idx = np.array(struc_node_te_indices)[te_order]
    te_sorted_y = te_y[te_order]

    n = len(panels)
    mapping = np.zeros((n, 4), dtype=int)

    for i, panel in enumerate(panels):
        y = panel.aerodynamic_center[1]
        # LE insertion point
        hi_le = np.searchsorted(le_sorted_y, y)
        lo_le = np.clip(hi_le - 1, 0, len(le_sorted_y) - 1)
        hi_le = np.clip(hi_le, 0, len(le_sorted_y) - 1)

        # TE insertion
        hi_te = np.searchsorted(te_sorted_y, y)
        lo_te = np.clip(hi_te - 1, 0, len(te_sorted_y) - 1)
        hi_te = np.clip(hi_te, 0, len(te_sorted_y) - 1)

        mapping[i, :] = [
            le_sorted_idx[lo_le],
            le_sorted_idx[hi_le],
            te_sorted_idx[lo_te],
            te_sorted_idx[hi_te],
        ]

    return mapping


def aero2struc_NN_vsm(
    f_aero_wing_vsm_format: np.ndarray,
    struc_nodes: np.ndarray,
    panel_cps: np.ndarray,
    panel_corner_map: np.ndarray,
    power_for_inverse_weighting: float = 2,
    eps: float = 1e-6,
    is_with_coupling_plot: bool = False,
):
    """
    Distribute each panel's resultant force (at its CoP) onto the four
    structural corner nodes given in panel_corner_map using inverse-distance weighting.

    Args:
        f_aero_wing_vsm_format (np.ndarray): Aerodynamic forces per panel (n_panels,3).
        struc_nodes (np.ndarray): Structural node positions (n_struc,3).
        panel_cps (np.ndarray): Panel control points (n_panels,3).
        panel_corner_map (np.ndarray): Mapping from panels to 4 node indices (n_panels,4).
        p (float): Power for inverse-distance weighting.
        eps (float): Small value to avoid division by zero.
        is_with_coupling_plot (bool): If True, plot the mapping.

    Returns:
        np.ndarray: Forces on structural nodes (n_struc,3).
    """

    n_struc = len(struc_nodes)
    f_aero_wing = np.zeros((n_struc, 3), dtype=float)

    for i, (cp, frc) in enumerate(zip(panel_cps, f_aero_wing_vsm_format)):
        sel_idx = panel_corner_map[i]  # [le_lo, le_hi, te_lo, te_hi]
        sel_coords = struc_nodes[sel_idx]  # (4,3)

        # true inverse-distance weighting across the 4 nodes
        d = np.linalg.norm(sel_coords - cp[None, :], axis=1)
        w = 1.0 / (d**power_for_inverse_weighting + eps)
        w /= np.sum(w)

        f_vals = w[:, None] * frc[None, :]  # (4,3)

        # accumulate
        for local_j, glob_j in enumerate(sel_idx):
            f_aero_wing[glob_j] += f_vals[local_j]

    if is_with_coupling_plot:
        plot_aerodynamic_forces_chordwise_distributed(
            panel_cps=panel_cps,
            f_aero_chordwise=f_aero_wing_vsm_format,
            nodes_struc=struc_nodes,
            force_struc=f_aero_wing,
        )

    return f_aero_wing


# def main(
#     coupling_method: str,
#     f_aero_wing_vsm_format: np.ndarray,
#     struc_nodes: np.ndarray,
#     panel_cp_locations: np.ndarray,
#     aero2struc_mapping: np.ndarray,
#     is_with_coupling_plot: bool,
#     config_aer2struc: dict,
# ):
#     """
#     Main interface for mapping aerodynamic panel forces to structural nodes.

#     Args:
#         coupling_method (str): Coupling method name (e.g., "NN").
#         f_aero_wing_vsm_format (np.ndarray): Aerodynamic forces per panel (n_panels,3).
#         struc_nodes (np.ndarray): Structural node positions (n_struc,3).
#         panel_cp_locations (np.ndarray): Panel control points (n_panels,3).
#         aero2struc_mapping (np.ndarray): Mapping from panels to 4 node indices (n_panels,4).
#         is_with_coupling_plot (bool): If True, plot the mapping.
#         p (float): Power for inverse-distance weighting.
#         eps (float): Small value to avoid division by zero.

#     Returns:
#         np.ndarray: Forces on structural nodes (n_struc,3).
#     """

#     if coupling_method == config_aer2struc["coupling_method"]:
#         return aero2struc_NN_vsm(
#             f_aero_wing_vsm_format,  # (n_panels,3)
#             struc_nodes,  # (n_struc,3)
#             panel_cp_locations,  # (n_panels,3)
#             aero2struc_mapping,  # (n_panels,4)
#             power_for_inverse_weighting=config_aer2struc["power_for_inverse_weighting"],
#             eps=config_aer2struc["eps"],
#             is_with_coupling_plot=is_with_coupling_plot,
#         )
#     else:
#         raise ValueError("Coupling method not recognized; wrong name or typo")

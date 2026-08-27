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

"""Kite body axes anchored to the CENTRE PANEL of the wing.

The body frame K is the reference for the kite attitude angles (phi_b,
theta_b, psi_b) and for the 6-DOF force and moment coefficients. For a soft or
morphing kite the usual references are ambiguous across flight conditions:
both the aerodynamic axes and the principal axes shift with the deformed
shape. The orientation is therefore anchored to the CENTRE PANEL of the wing
-- the panel that spans the centreline, bounded by the two innermost struts --
so the frame is a property of the material structure and means the same thing
at every point in the wind window.

Construction (all in the structural / VSM frame)::

    x_K = unit(P_LE,c - P_TE,c)        centre-panel chord, TE -> LE (forward)
    z_K = unit(x_K x s),  s = panel span direction     panel normal, downward
    y_K = z_K x x_K                    completes the right-handed set

``P_LE,c`` and ``P_TE,c`` are the mid-span leading- and trailing-edge points of
the centre panel (the mean of its two bounding struts), and ``s`` is the
panel's span direction: the vector between those struts' mid-chord points,
equally the mean of the panel's leading- and trailing-edge spanwise edges.
``z_K`` is therefore the normal of the centre panel -- normal to ``x_K`` and to
the panel span -- and ``y_K`` lies in the panel, pointing to starboard.

The chord is taken EXACTLY; the span direction only fixes the roll of the frame
about it, so its chordwise component is irrelevant. A wing that deforms
asymmetrically under steering rolls the frame with it, which is a real material
rotation and belongs there.

Wings with an odd number of struts have a strut ON the centreline and hence no
single panel spanning it; the innermost strut on each side is used, i.e. the
pair of half-panels that straddle the centreline together.

The frame is deliberately NOT aligned with the inertia tensor. Principal axes
move with the mass distribution (KCU mass, depower state, bridle stretch) and,
on the LEI V3, are badly conditioned: I_xx and I_zz are nearly equal, so the
x-z eigenvector is nearly free and swings by ~10 deg between deformed shapes.
A near-diagonal inertia tensor buys nothing here, because the rotational
dynamics are never integrated -- the inertia tensor enters only through the
gyroscopic term, which is small and is already dropped in the closed-form
derivation. The price is that the tensor in body axes carries products of
inertia; they are reported in :attr:`RigidBodyAxes.inertia_body`.

Body-axis convention: aircraft **FRD** -- x forward (along the flight
direction), y to the right wing (starboard), z down (from canopy toward the
bridle/ground station). The construction above yields that sense directly, so
the textbook static-stability criteria apply as written: C_m_alpha < 0,
C_n_beta > 0, C_l_beta < 0.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from awetrim.aerostructural.utils import (
    _validate_struct_nodes_and_masses,
    calculate_cg,
    calculate_inertia,
)


# Canonical aircraft body directions (rows: forward, right, down) in the
# structural / VSM frame. The structural frame has X and Y negated relative to
# the course frame (T_structural_from_C = diag(-1, -1, 1)); the course frame is
# X_C = flight direction (forward), Z_C = radial outward (up), and hence
# Y_C = Z_C x X_C = left. So, in structural coordinates:
#   forward = +X_C -> [-1, 0, 0]
#   right   = -Y_C -> [ 0, 1, 0]
#   down    = -Z_C -> [ 0, 0,-1]
# These only CHECK the sense of the triad the centre panel produces; the
# directions themselves come from the panel's corners. See
# ``compute_centre_panel_axes``.
FRD_IN_STRUC = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])

#: Column of ``struc_nodes`` that runs spanwise in the structural frame.
_SPAN_COL = 1


@dataclass
class CentrePanelAxes:
    """Body triad and the centre-panel corners it is anchored to.

    Attributes:
        axes: Rotation matrix whose *rows* are the unit body-axis vectors
            [x_K, y_K, z_K] in structural-frame components, shape (3,3).
            Transforms a vector from body frame to structural frame via
            ``axes.T @ v_body``.
        le_starboard: Leading edge of the panel's starboard strut, shape (3,).
        te_starboard: Trailing edge of the panel's starboard strut, shape (3,).
        le_port: Leading edge of the panel's port strut, shape (3,).
        te_port: Trailing edge of the panel's port strut, shape (3,).
        le_centre: Mid-span leading edge, ``(le_starboard + le_port) / 2``.
        te_centre: Mid-span trailing edge, ``(te_starboard + te_port) / 2``.
        chord_centre: Centre-panel chord length [m], ``|le_centre - te_centre|``.
        span_centre: Distance between the two bounding struts' mid-chord points
            [m] -- the panel's own span, not the wing's.
    """

    axes: np.ndarray
    le_starboard: np.ndarray
    te_starboard: np.ndarray
    le_port: np.ndarray
    te_port: np.ndarray
    le_centre: np.ndarray
    te_centre: np.ndarray
    chord_centre: float
    span_centre: float

    @property
    def reference_points(self) -> dict[str, np.ndarray]:
        """The four panel corners, keyed by name (for plots and reports)."""
        return {
            "le_starboard": self.le_starboard,
            "te_starboard": self.te_starboard,
            "le_port": self.le_port,
            "te_port": self.te_port,
        }


@dataclass
class RigidBodyAxes:
    """Centre-panel body axes plus the mass properties of the node cloud.

    Body axes are FRD (x forward, y right, z down); see the module docstring.

    Attributes:
        cg: Center of gravity in the structural frame, shape (3,).
        cg_body: CG position expressed in body-axis coordinates, shape (3,).
            Components of the structural-frame CG vector along each body axis,
            i.e. ``body_axes @ cg``.
        inertia_cg: Full 3x3 inertia tensor about the CG in the structural frame.
        inertia_body: The same tensor resolved in the body axes,
            ``body_axes @ inertia_cg @ body_axes.T``. Its off-diagonal terms are
            the products of inertia this frame does not remove.
        inertia_moments: Diagonal of ``inertia_body``, [I_xx, I_yy, I_zz] about
            the body axes, shape (3,). Moments of inertia about the CENTRE-PANEL
            axes -- not principal moments.
        body_axes: Rotation matrix whose *rows* are the unit body-axis vectors
            [x_K, y_K, z_K], shape (3,3). Transforms a vector from body frame to
            structural frame via ``body_axes.T @ v_body``.
        panel: The :class:`CentrePanelAxes` the triad was built from.
    """

    cg: np.ndarray
    cg_body: np.ndarray
    inertia_cg: np.ndarray
    inertia_body: np.ndarray
    inertia_moments: np.ndarray
    body_axes: np.ndarray
    panel: CentrePanelAxes


def wing_le_te_indices(struc_geometry: dict) -> tuple[np.ndarray, np.ndarray]:
    """Leading- and trailing-edge node indices of the wing, paired per strut.

    Mirrors the PSM node ordering of
    ``aerostructural.pss.structural_geometry_io``: node 0 is the KCU/bridle
    point and the wing particles follow in id order, odd ids on the leading
    edge and even ids on the trailing edge, so a node's array index equals its
    id. ``le[k]`` and ``te[k]`` are the two ends of strut ``k``.

    Args:
        struc_geometry: Parsed struc_geometry YAML as a Python dict.

    Returns:
        Tuple ``(le_indices, te_indices)`` of integer arrays, spanwise-ordered
        as the YAML lists them, one entry per strut.
    """
    le_indices: list[int] = []
    te_indices: list[int] = []
    for row in struc_geometry["wing_particles"]["data"]:
        node_idx = int(row[0])
        if node_idx % 2 != 0:
            le_indices.append(node_idx)
        else:
            te_indices.append(node_idx)

    if len(le_indices) != len(te_indices):
        raise ValueError(
            "wing_particles must hold one leading-edge (odd id) and one "
            f"trailing-edge (even id) node per strut; got {len(le_indices)} LE "
            f"and {len(te_indices)} TE nodes."
        )
    return np.asarray(le_indices, dtype=int), np.asarray(te_indices, dtype=int)


def _unit(vec: np.ndarray) -> np.ndarray:
    """Unit vector, raising on a degenerate (near-zero) direction."""
    norm = float(np.linalg.norm(vec))
    if norm < 1e-12:
        raise ValueError("Degenerate direction: the reference points coincide.")
    return np.asarray(vec, dtype=float) / norm


def _centre_panel_struts(p_le: np.ndarray, p_te: np.ndarray) -> tuple[int, int]:
    """Indices of the two struts bounding the centre panel (starboard, port).

    The innermost strut on each side of the structural symmetry plane. A wing
    with an odd number of struts has one ON the centreline and no single panel
    spanning it; the pair returned then bounds the two half-panels that
    straddle the centreline together.
    """
    span_strut = 0.5 * (p_le[:, _SPAN_COL] + p_te[:, _SPAN_COL])
    starboard = np.flatnonzero(span_strut > 0.0)
    port = np.flatnonzero(span_strut < 0.0)
    if not starboard.size or not port.size:
        raise ValueError(
            "Cannot locate the centre panel: the wing struts do not straddle "
            "the structural symmetry plane."
        )
    return (
        int(starboard[np.argmin(span_strut[starboard])]),
        int(port[np.argmax(span_strut[port])]),
    )


def compute_centre_panel_axes(
    struc_nodes: np.ndarray,
    le_indices: np.ndarray,
    te_indices: np.ndarray,
) -> CentrePanelAxes:
    """Body triad anchored to the centre panel of the wing.

    See the module docstring for the construction and for why the frame is
    anchored to the structure rather than to the inertia or the aerodynamics.

    Args:
        struc_nodes: Particle positions in the structural frame, shape (n, 3).
            May be the deformed cloud -- the triad follows the panel corners.
        le_indices: Leading-edge node indices, one per strut.
        te_indices: Trailing-edge node indices, paired with ``le_indices``.

    Returns:
        :class:`CentrePanelAxes` with the FRD triad and the panel corners.
    """
    nodes = np.asarray(struc_nodes, dtype=float)
    if nodes.ndim != 2 or nodes.shape[1] != 3:
        raise ValueError(f"struc_nodes must have shape (n, 3); got {nodes.shape}.")

    le = np.asarray(le_indices, dtype=int)
    te = np.asarray(te_indices, dtype=int)
    if le.shape != te.shape:
        raise ValueError(
            f"le_indices {le.shape} and te_indices {te.shape} must pair per strut."
        )
    if le.size < 2:
        raise ValueError("At least two struts are needed to define the body axes.")

    p_le = nodes[le]
    p_te = nodes[te]
    k_s, k_p = _centre_panel_struts(p_le, p_te)
    le_starboard, te_starboard = p_le[k_s], p_te[k_s]
    le_port, te_port = p_le[k_p], p_te[k_p]

    # Chord axis: the centre-panel chord, trailing edge -> leading edge.
    le_centre = 0.5 * (le_starboard + le_port)
    te_centre = 0.5 * (te_starboard + te_port)
    chord = le_centre - te_centre
    x_axis = _unit(chord)

    # Panel span, port -> starboard: the mid-chord points of the two bounding
    # struts, equally the mean of the panel's LE and TE spanwise edges.
    span = 0.5 * (le_starboard + te_starboard) - 0.5 * (le_port + te_port)

    # Panel normal, downward. The span fixes only the roll of the frame about
    # the chord, so its chordwise component drops out of the cross product.
    z_axis = _unit(np.cross(x_axis, span))
    y_axis = np.cross(z_axis, x_axis)

    axes = np.vstack([x_axis, y_axis, z_axis])
    alignment = np.einsum("ij,ij->i", axes, FRD_IN_STRUC)
    if np.any(alignment <= 0.0):
        raise ValueError(
            "Centre-panel body axes are not in the aircraft FRD sense "
            f"(alignment with forward/right/down = {alignment}). The node "
            "cloud is probably not in the structural/VSM frame."
        )

    return CentrePanelAxes(
        axes=axes,
        le_starboard=le_starboard,
        te_starboard=te_starboard,
        le_port=le_port,
        te_port=te_port,
        le_centre=le_centre,
        te_centre=te_centre,
        chord_centre=float(np.linalg.norm(chord)),
        span_centre=float(np.linalg.norm(span)),
    )


def compute_rigid_body_axes(
    struc_nodes: np.ndarray,
    m_arr: np.ndarray,
    le_indices: np.ndarray,
    te_indices: np.ndarray,
) -> RigidBodyAxes:
    """CG, inertia tensor and centre-panel body axes of a PSM node cloud.

    Steps:
    1. Compute CG as the mass-weighted average node position.
    2. Assemble the 3x3 inertia tensor about the CG.
    3. Build the body triad from the wing's centre panel
       (:func:`compute_centre_panel_axes`) and resolve the inertia tensor in
       it. The tensor is NOT diagonalised -- see the module docstring.

    Args:
        struc_nodes: Particle positions, shape (n_nodes, 3).
        m_arr: Particle masses, shape (n_nodes,).
        le_indices: Leading-edge node indices, one per strut
            (:func:`wing_le_te_indices`).
        te_indices: Trailing-edge node indices, paired with ``le_indices``.

    Returns:
        :class:`RigidBodyAxes` with cg, cg_body, inertia_cg, inertia_body,
        inertia_moments, body_axes and the centre-panel corners.
    """
    nodes, masses = _validate_struct_nodes_and_masses(struc_nodes, m_arr)

    cg = calculate_cg(nodes, masses)

    node_mass_pairs = [(nodes[i], masses[i]) for i in range(len(nodes))]
    inertia_cg = calculate_inertia(node_mass_pairs, desired_point=cg)

    panel = compute_centre_panel_axes(nodes, le_indices, te_indices)
    body_axes = panel.axes
    inertia_body = body_axes @ inertia_cg @ body_axes.T

    return RigidBodyAxes(
        cg=cg,
        cg_body=body_axes @ cg,
        inertia_cg=inertia_cg,
        inertia_body=inertia_body,
        inertia_moments=np.diag(inertia_body).copy(),
        body_axes=body_axes,
        panel=panel,
    )


def load_psm_nodes_and_masses(
    struc_geometry: dict,
    system_config: dict | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Particle positions and lumped masses of the PSM node cloud.

    Delegates to ``aerostructural.pss.structural_geometry_io.main`` -- the
    single source for the node ordering, the element-to-node mass lumping
    (wing tubes and canopy, bridle lines, pulleys) and the KCU-mass
    resolution.

    ``system_config`` is the parsed ``system.yaml``, where the KCU mass lives
    (``components ... control_system.structure.mass``). Without it the
    resolver falls back to the deprecated ``struc_geometry.kcu_mass``, or to
    0 kg with a warning, and the CG and inertia are then wrong by the whole
    KCU. :func:`load_psm_geometry` finds the file for you.

    Args:
        struc_geometry: Parsed struc_geometry YAML as a Python dict.
        system_config: Parsed system.yaml as a Python dict, if available.

    Returns:
        Tuple ``(struc_nodes, m_arr)`` with shapes (n_nodes, 3) and (n_nodes,).
    """
    from awetrim.aerostructural.pss.structural_geometry_io import (
        main as pss_initialize,
    )

    result = pss_initialize(struc_geometry, system_config=system_config)
    return np.asarray(result[0], dtype=float), np.asarray(result[1], dtype=float)


def load_psm_geometry(
    struc_geometry_path: Path | str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Everything the body axes need, from a struc_geometry.yaml path.

    Reads ``system.yaml`` from the same directory when it is there -- the
    single source of the KCU mass, saved alongside every geometry snapshot --
    and builds the cloud through the PSS initializer, so the node ordering,
    the masses and the LE/TE indices all come from one place.

    Args:
        struc_geometry_path: Path to a struc_geometry.yaml (deformed or not).

    Returns:
        Tuple ``(struc_nodes, m_arr, le_indices, te_indices)``.
    """
    import yaml
    from awetrim.aerostructural.pss.structural_geometry_io import (
        main as pss_initialize,
    )

    path = Path(struc_geometry_path)
    struc_geometry = yaml.safe_load(path.read_text(encoding="utf-8"))
    system_yaml = path.parent / "system.yaml"
    system_config = (
        yaml.safe_load(system_yaml.read_text(encoding="utf-8"))
        if system_yaml.exists()
        else None
    )
    if system_config is None:
        logging.warning(
            "No system.yaml next to %s; the KCU mass falls back to the "
            "deprecated struc_geometry key (or 0 kg) and the CG and inertia "
            "are off by the KCU.",
            path,
        )
    result = pss_initialize(struc_geometry, system_config=system_config)
    return (
        np.asarray(result[0], dtype=float),
        np.asarray(result[1], dtype=float),
        np.asarray(result[2], dtype=int),
        np.asarray(result[3], dtype=int),
    )

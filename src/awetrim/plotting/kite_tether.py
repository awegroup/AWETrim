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

"""Shared 3D kite + Williams-tether drawing.

A reusable helper so several scripts can render the same kite/tether picture.
``draw_kite_tether`` draws, onto a provided 3D axis, the Williams tether shape
from ground to kite, the kite tubes/bridles (from ``struc_geometry.yaml`` when
available, else the VSM panel outline) at the trimmed attitude, the KCU, and an
optional faint wind-window surface.

The attitude convention matches the VSM quasi-steady solver exactly (see the
inline notes): the trim Euler angles are applied as R = Yaw*Pitch*Roll about the
VSM axes, then mapped to the course frame via T_C_from_VSM and to the wind frame.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def euler_C_from_K(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Numpy version of reference_frames.transformation_C_from_K (Yaw*Pitch*Roll)."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    Roll = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Pitch = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Yaw = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Yaw @ Pitch @ Roll


def T_Wind_from_C(system_model: Any) -> np.ndarray:
    """Numerical wind<-course matrix evaluated from the current system_model angles."""
    import casadi as ca
    from awetrim.utils.reference_frames import transformation_C_from_W

    az = float(system_model.angle_azimuth)
    el = float(system_model.angle_elevation)
    chi = float(system_model.angle_course)
    dw = float(getattr(system_model.wind, "direction_wind", 0.0))
    T_C_from_W = np.array(
        ca.DM(transformation_C_from_W(az, el, chi)).full(), dtype=float
    )
    T_Wind_from_W = np.array(
        [
            [np.cos(-dw), -np.sin(-dw), 0.0],
            [np.sin(-dw), np.cos(-dw), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    return T_Wind_from_W @ T_C_from_W.T


def load_struc_nodes_and_edges(struc_path: str | None):
    """Return (nodes_C, wing_edges, bridle_edges, wing_ids) from struc_geometry.yaml.

    Nodes are in the course frame with KCU at origin (node 0). Wing nodes come
    from ``wing_particles``; bridle nodes from ``bridle_particles``. A pulley
    row (3 ids) is split into its two segments ci-cj and cj-ck -- the pulley
    node is cj, as the structural reader itself has it (see
    ``aerostructural/fem/read_struc_geometry_yaml``). Returns ``None`` if
    unavailable.
    """
    if not struc_path:
        return None
    import yaml

    path = Path(struc_path)
    if not path.exists():
        return None
    data = yaml.safe_load(path.read_text(encoding="utf-8"))

    nodes: dict[int, np.ndarray] = {0: np.zeros(3, dtype=float)}
    wing_ids: list[int] = []
    for row in data.get("wing_particles", {}).get("data", []):
        pid = int(row[0])
        nodes[pid] = np.array([float(row[1]), float(row[2]), float(row[3])])
        wing_ids.append(pid)
    for row in data.get("bridle_particles", {}).get("data", []):
        pid = int(row[0])
        nodes[pid] = np.array([float(row[1]), float(row[2]), float(row[3])])

    def _edges(key: str) -> list[tuple[int, int]]:
        out: list[tuple[int, int]] = []
        for row in data.get(key, {}).get("data", []):
            ids = [int(v) for v in row[1:] if v is not None]
            if len(ids) == 2:
                out.append((ids[0], ids[1]))
            elif len(ids) == 3:  # pulley: ci-cj and cj-ck, pulley node cj
                out.append((ids[0], ids[1]))
                out.append((ids[1], ids[2]))
        return out

    return nodes, _edges("wing_connections"), _edges("bridle_connections"), wing_ids


def draw_wind_axes(
    ax: Any,
    radius: float,
    *,
    scale: float = 1.12,
    color: str = "0.3",
) -> None:
    """Draw the wind-frame axis triad at the ground origin.

    x_w points downwind (into the window), z_w up, y_w lateral (crosswind). The
    arrows are ``scale`` * tether radius long so they extend just past the
    wind-window shell. With ``direction_wind = 0`` the wind frame coincides with
    the window (azimuth/elevation) frame the kite/tether are drawn in.
    """
    length = scale * float(radius)
    axes_def = [
        ((length, 0.0, 0.0), r"$x_w$ (downwind)"),
        ((0.0, length, 0.0), r"$y_w$"),
        ((0.0, 0.0, length), r"$z_w$ (up)"),
    ]
    for (vx, vy, vz), label in axes_def:
        ax.quiver(
            0.0, 0.0, 0.0, vx, vy, vz,
            color=color, linewidth=1.3, arrow_length_ratio=0.04, zorder=5,
        )
        ax.text(vx, vy, vz, label, color=color, fontsize=9, zorder=6)


def draw_kite_tether(
    ax: Any,
    result: dict[str, Any],
    system_model: Any,
    body: Any,
    struc_geometry_path: str | None,
    *,
    kite_scale: float = 5.0,
    draw_window: bool = True,
    draw_ground: bool = True,
    set_aspect: bool = True,
    color: str = "black",
    state_label: str | None = None,
) -> None:
    """Draw the kite + Williams tether for one trimmed state onto ``ax`` (3D).

    To overlay several states in one wind window, call once per state with a
    distinct ``color`` and ``state_label`` and ``draw_window=draw_ground=False``
    for all but the first. When ``state_label`` is given, the tether line carries
    that label and the per-part proxies (tubes/bridles/KCU) are suppressed so the
    legend stays one-entry-per-state.
    """
    positions = result["williams_positions"]
    rk = np.asarray(result["r_kite"], dtype=float)
    radius = float(result["williams_tether_length"])

    if draw_window:
        # Wind window: az in [-90, 90] deg, el in [0, 90] deg at radial = tether length.
        az_grid = np.deg2rad(np.linspace(-90.0, 90.0, 40))
        el_grid = np.deg2rad(np.linspace(0.0, 90.0, 20))
        az_mesh, el_mesh = np.meshgrid(az_grid, el_grid)
        ww_x = radius * np.cos(el_mesh) * np.cos(az_mesh)
        ww_y = radius * np.cos(el_mesh) * np.sin(az_mesh)
        ww_z = radius * np.sin(el_mesh)
        ax.plot_surface(
            ww_x, ww_y, ww_z, color="#4C78A8", alpha=0.08, linewidth=0, shade=False
        )

    # Color/linewidth convention:
    #   kite + tether: same color, inflatable tubes thickest, tether medium
    #   bridles: grey, thinnest
    kite_color = color
    lw_tube = 2.5
    lw_tether = 1.4
    lw_bridle = 0.7

    ax.plot(
        positions[:, 0],
        positions[:, 1],
        positions[:, 2],
        color=kite_color,
        linewidth=lw_tether,
        marker="o",
        markersize=2,
        label=state_label if state_label is not None else "tether",
    )
    if draw_ground:
        ax.scatter([0.0], [0.0], [0.0], color="black", s=40, label="ground")

    # Kite + bridles, scaled up so they are visible against the tether length.
    # The KCU is placed at r_kite in the course frame; the trim attitude
    # (roll, pitch, yaw) is applied in the VSM frame exactly like the solver
    # (R = Yaw*Pitch*Roll), then mapped to course via T_C_from_VSM and to wind.
    from awetrim.aerodynamics.vsm_quasi_steady import (
        DEFAULT_TRANSFORMATION_C_FROM_VSM,
    )

    opt_x = np.asarray(result["opt_x"], dtype=float)
    R_VSM = euler_C_from_K(
        np.deg2rad(opt_x[1]), np.deg2rad(opt_x[2]), np.deg2rad(opt_x[3])
    )
    T_C_from_VSM = np.asarray(DEFAULT_TRANSFORMATION_C_FROM_VSM, dtype=float)
    T_W_from_C = T_Wind_from_C(system_model)

    def _to_wind_C(p_C_kcu: np.ndarray) -> np.ndarray:
        """Scale around KCU and place at rk in wind frame (attitude pre-applied)."""
        p = (T_W_from_C @ p_C_kcu.T).T * kite_scale
        return p + rk

    struc_data = load_struc_nodes_and_edges(struc_geometry_path)
    wing_edges: list[tuple[int, int]] = []
    bridle_edges: list[tuple[int, int]] = []
    nodes_C: dict[int, np.ndarray] | None = None
    if struc_data is not None and struc_data[3]:
        nodes_C, wing_edges, bridle_edges, _wing_ids = struc_data

    def _draw_struc_edges(edges, *, color, linewidth, linestyle="-"):
        for ci, cj in edges:
            if ci not in nodes_C or cj not in nodes_C:
                continue
            # struc_geometry nodes are in the body/VSM frame (same as the aero
            # geometry and the VSM panels), so apply the trim attitude in VSM
            # then convert to the course frame -- exactly like the panel path.
            # (Previously these were treated as course-frame, which applied an
            # extra T_C_from_VSM = diag(-1,-1,1) and flipped the kite 180 deg.)
            seg_VSM = np.vstack([nodes_C[ci], nodes_C[cj]]).T
            seg_C = (T_C_from_VSM @ (R_VSM @ seg_VSM)).T
            seg_W = _to_wind_C(seg_C)
            ax.plot(
                seg_W[:, 0], seg_W[:, 1], seg_W[:, 2],
                color=color, linewidth=linewidth, linestyle=linestyle,
            )

    if nodes_C is not None and wing_edges:
        _draw_struc_edges(wing_edges, color=kite_color, linewidth=lw_tube)
        ax.plot(
            [], [], color=kite_color, linewidth=lw_tube,
            label=f"kite tubes (x{kite_scale:g})",
        )
    else:
        # Fallback: VSM panel outline if no struc_geometry is available.
        panels = list(getattr(body, "panels", []) or [])
        for panel in panels:
            pts_VSM = np.asarray(panel.corner_points, dtype=float)
            pts_C = (T_C_from_VSM @ (R_VSM @ pts_VSM.T)).T
            loop_C = np.vstack([pts_C, pts_C[:1]])
            loop_W = _to_wind_C(loop_C)
            ax.plot(
                loop_W[:, 0], loop_W[:, 1], loop_W[:, 2],
                color=kite_color, linewidth=lw_tube,
            )
        if panels:
            ax.plot(
                [], [], color=kite_color, linewidth=lw_tube,
                label=f"kite panels (x{kite_scale:g})",
            )

    if nodes_C is not None and bridle_edges:
        _draw_struc_edges(
            bridle_edges, color="tab:grey", linewidth=lw_bridle, linestyle="--"
        )
        ax.plot(
            [], [], color="tab:grey", linewidth=lw_bridle, linestyle="--",
            label="bridles",
        )

    ax.scatter(
        rk[0], rk[1], rk[2], color=kite_color, marker="D", s=10,
        label="KCU" if state_label is None else None,
    )

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    if set_aspect:
        # Equal aspect so the quarter-sphere actually looks like a sphere.
        try:
            ax.set_aspect("equal")
        except NotImplementedError:
            ax.set_xlim(0.0, radius)
            ax.set_ylim(-radius, radius)
            ax.set_zlim(0.0, radius)
            ax.set_box_aspect((1.0, 2.0, 1.0))


def tether_info_str(system_model: Any) -> str:
    """Short ``Name, N=.., cf=..`` describing the tether model, for titles."""
    tether = system_model.tether
    info = type(tether).__name__
    if hasattr(tether, "n_elements"):
        info += f", N={int(tether.n_elements)}"
    if hasattr(tether, "cf"):
        info += f", cf={float(tether.cf):.3g}"
    return info

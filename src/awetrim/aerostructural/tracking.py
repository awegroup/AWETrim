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


def setup_tracking_arrays(n_pts, t_vector, n_panels=0, n_frames=0):
    """
    Initialize tracking arrays for simulation results.

    Args:
        n_pts (int): Number of nodes/particles.
        t_vector (np.ndarray): Array of time steps.
        n_panels (int): Number of aerodynamic panels (0 skips aero tracking arrays).
        n_frames (int): Number of material frames to store per iteration, or 0
            for none. Needed by any backend whose elements carry ROTATIONAL
            state -- a geometrically exact beam's curvature and strain energy
            live in the frames, not in the node positions, so without them a
            saved run cannot be asked afterwards how bent its tubes were or
            re-evaluated at all. NOT one per node: frames are stored per
            rotational slot, so pass ``len(state.frames)``, which on the LEI-V3
            is 98 against 268 nodes.

    Returns:
        dict: Dictionary with preallocated arrays for positions, forces, and tracking metrics.
    """
    nt = len(t_vector)
    arrays = {
        "positions": np.zeros((nt, n_pts, 3)),
        "f_ext": np.zeros((nt, n_pts, 3)),
        "f_int": np.zeros((nt, n_pts, 3)),
        "residual_norm": np.zeros(nt),
        "max_residual": np.zeros(nt),
    }
    if n_panels > 0:
        arrays["alpha_at_ac"] = np.full((nt, n_panels), np.nan)
        arrays["stall_mask"] = np.zeros((nt, n_panels), dtype=bool)
    if n_frames:
        # Stored as a PAIR. The `positions` array above is the Aitken-relaxed
        # geometry the next aero solve sees; these two are the configuration the
        # structural solver actually returned. They are not interchangeable: a
        # geometrically exact beam's energy depends on positions and frames
        # together, so evaluating the relaxed positions against these frames
        # gives a state the solver never produced (measured: a 2288 N residual
        # where the solve reported 0.46 N).
        arrays["solved_positions"] = np.zeros((nt, n_pts, 3))
        arrays["frames"] = np.tile(np.eye(3), (nt, int(n_frames), 1, 1))
    # Per-iteration flight/actuation state. An actuated run walks the tape
    # through several converged states inside ONE call, so without these a saved
    # sweep cannot say which iteration belongs to which tape length or what
    # apparent speed it flew at -- the whole chain collapses to its last point.
    arrays["tape_length"] = np.full(nt, np.nan)
    arrays["speed_apparent"] = np.full(nt, np.nan)
    # Steering tape half-difference [m] (left shortened, right lengthened by
    # it), and the trim unknowns [kite_speed, roll, pitch, yaw, course_rate]
    # each aero solve returned. The attitude entries are INCREMENTS: the
    # driver rotates the geometry by them and restarts the next trim at zero,
    # so the kite's attitude is their composition over the run, not the last
    # row.
    arrays["steering_half_difference"] = np.full(nt, np.nan)
    arrays["trim_state"] = np.full((nt, 5), np.nan)
    return arrays


def update_aero_tracking(tracking_data, idx, alpha_at_ac, stall_mask):
    """Store per-panel aerodynamic state for one iteration.

    Args:
        tracking_data: Tracking dict (must contain 'alpha_at_ac' and 'stall_mask').
        idx: Current iteration index.
        alpha_at_ac: Per-panel local AoA [rad], shape (n_panels,) or None.
        stall_mask: Boolean stall flag per panel, shape (n_panels,) or None.
    """
    if "alpha_at_ac" not in tracking_data or alpha_at_ac is None:
        return
    alpha = np.ravel(alpha_at_ac)
    n = tracking_data["alpha_at_ac"].shape[1]
    tracking_data["alpha_at_ac"][idx, : min(len(alpha), n)] = alpha[: n]
    if stall_mask is not None:
        mask = np.ravel(stall_mask).astype(bool)
        tracking_data["stall_mask"][idx, : min(len(mask), n)] = mask[: n]


def update_tracking_arrays(
    tracking_data,
    idx,
    struc_nodes,
    f_ext_flat,
    f_int_flat,
    frames=None,
    solved_positions=None,
    tape_length=None,
    speed_apparent=None,
    steering_half_difference=None,
    trim_state=None,
):
    """
    Update tracking arrays with simulation results for a single time step.

    Args:
        tracking_data (dict): Tracking arrays to update.
        idx (int): Current time step index.
        pos3d (np.ndarray): Current 3D positions (n_nodes, 3).
        f_ext_flat (np.ndarray): Flattened external force vector (n_nodes*3,).
        f_int_flat (np.ndarray): Flattened internal force vector (n_nodes*3,).
        frames (np.ndarray or None): Material frames, ordered by rotational
            slot, shape ``(n_frames, 3, 3)``. Stored only when the arrays were
            built with a non-zero ``n_frames``, and only meaningful together
            with ``solved_positions``.
        solved_positions (np.ndarray or None): The positions those frames belong
            to -- the structural solver's own output, before Aitken relaxation.

    Returns:
        None. Updates tracking_data in place.
    """
    # Unpack 3D storage
    pos3d = tracking_data["positions"]
    ext3d = tracking_data["f_ext"]
    int3d = tracking_data["f_int"]

    n_pts = pos3d.shape[1]

    # 1) Positions
    pos3d[idx] = struc_nodes

    # 2) External & internal forces: reshape before storing
    ext3d[idx] = f_ext_flat.reshape(n_pts, 3)
    int3d[idx] = f_int_flat.reshape(n_pts, 3)

    # 3) Norms
    tracking_data["residual_norm"][idx] = np.linalg.norm(f_int_flat)
    tracking_data["max_residual"][idx] = np.max(np.abs(f_int_flat))

    # 4) Flight and actuation state, so a chain can be sliced apart later
    for name, value in (("tape_length", tape_length),
                        ("speed_apparent", speed_apparent),
                        ("steering_half_difference", steering_half_difference)):
        if value is not None and name in tracking_data:
            tracking_data[name][idx] = float(value)
    if trim_state is not None and "trim_state" in tracking_data:
        state = np.ravel(np.asarray(trim_state, dtype=float))
        tracking_data["trim_state"][idx, : min(state.size, 5)] = state[:5]

    # 5) Rotational state, where the backend has any
    if frames is not None and "frames" in tracking_data:
        tracking_data["frames"][idx] = np.asarray(frames, dtype=float).reshape(
            -1, 3, 3
        )
    if solved_positions is not None and "solved_positions" in tracking_data:
        tracking_data["solved_positions"][idx] = np.asarray(
            solved_positions, dtype=float
        ).reshape(n_pts, 3)

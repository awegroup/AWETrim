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
#
# Portions of this file are adapted from ASKITE
# (https://github.com/awegroup/ASKITE), licensed under the MIT License,
# Copyright (c) 2024 jellepoland (Jelle Poland, Patrick Roeleveld, TU Delft).
# See the NOTICE file at the repository root for the full MIT licence text.

import time
from tqdm import tqdm
import numpy as np
import logging
from pathlib import Path
import copy
import dataclasses
from . import aero2struc, structural_kite_fem
from ..billow import structural_billow
from ..pss import structural_pss
from .. import aerodynamic_vsm, aerodynamic_bridle_line_drag, tracking
from awetrim import plotting
from awetrim.aerodynamics.apparent_wind import apparent_wind_at, inflow_state_of

from ..mapping import LinearStructuralToAeroMapper
from ..utils import calculate_cg, rotate_geometry

# The ONE mass-proportional split, shared with the PSS driver. This module used
# to keep its own copy that clipped negative masses away and normalised by the
# positive sum -- while `calculate_cg`, which places the trim's inertial and
# gravity resultants, weighs every node as it is. On the LEI-V3 FEM_full
# geometry 96 tube nodes carry a small NEGATIVE mass (the reader hangs
# `mass_without_bridles - mass_canopy` on the tubes, and that is negative when
# the YAML's canopy density over-fills the wing's mass), so the two disagreed
# about where the kite's mass is by 0.64 m in z -- a 20% error on every
# inertial and gravity moment the structure was given, 39 N m of roll on a
# steered state. Same weights on both sides or the moments cannot match.
from ..forces import distribute_total_force_by_particle_mass

_STRUC_TO_AERO_MAPPER = LinearStructuralToAeroMapper()


def build_symmetry_mapping(struc_nodes, tol=1e-5):
    """Build a mapping of symmetrical node pairs from the *initial* geometry.

    For every node with y > 0 (positive-span side), find its mirror partner
    at (x, −y, z) on the negative-span side.  Also identify nodes that sit
    on the symmetry plane (|y| ≤ tol).

    Call this **once** during initialisation and pass the result to
    :func:`forcing_symmetry` at every iteration.

    Parameters
    ----------
    struc_nodes : np.ndarray, shape (n, 3)
        Initial (undeformed) node coordinates.
    tol : float
        Absolute tolerance for matching mirror coordinates.

    Returns
    -------
    symmetry_mapping : dict
        ``{"pairs": np.ndarray shape (m, 2),
           "center_indices": list[int]}``
        *pairs[:, 0]* = positive-y node index (source),
        *pairs[:, 1]* = negative-y node index (mirror).
    """
    pos_indices = [i for i, pt in enumerate(struc_nodes) if pt[1] > tol]
    neg_indices = [i for i, pt in enumerate(struc_nodes) if pt[1] < -tol]
    center_indices = [i for i, pt in enumerate(struc_nodes) if abs(pt[1]) <= tol]

    pairs = []
    for pi in pos_indices:
        mirrored = np.array(
            [struc_nodes[pi][0], -struc_nodes[pi][1], struc_nodes[pi][2]]
        )
        for ni in neg_indices:
            if np.allclose(struc_nodes[ni], mirrored, atol=tol):
                pairs.append((pi, ni))
                break

    pairs = np.array(pairs) if pairs else np.empty((0, 2), dtype=int)

    logging.info(
        f"Symmetry mapping: {len(pairs)} pairs, {len(center_indices)} center nodes"
    )
    return {"pairs": pairs, "center_indices": center_indices}


def forcing_symmetry(struc_nodes, symmetry_mapping):
    """Force y-symmetry on the structural nodes using a pre-built mapping.

    For each (source, mirror) pair the mirror node is set to
    ``[x_source, -y_source, z_source]``.  Centre-plane nodes are forced to
    ``y = 0``.

    Parameters
    ----------
    struc_nodes : np.ndarray, shape (n, 3)
    symmetry_mapping : dict
        Output of :func:`build_symmetry_mapping`.

    Returns
    -------
    struc_nodes : np.ndarray
        The (modified in-place) node array.
    """
    for src, mir in symmetry_mapping["pairs"]:
        struc_nodes[mir] = np.array(
            [struc_nodes[src][0], -struc_nodes[src][1], struc_nodes[src][2]]
        )
    for ci in symmetry_mapping["center_indices"]:
        struc_nodes[ci][1] = 0.0
    return struc_nodes


def _compute_power_tape_increment(
    delta_power_tape,
    power_tape_final_extension,
    power_tape_extension_step,
    tol=1e-9,
):
    """
    Compute the signed rest-length increment needed to move toward the target extension.

    Returns:
        tuple: (increment, should_update)
    """
    remaining = power_tape_final_extension - delta_power_tape
    if np.abs(remaining) <= tol:
        return 0.0, False
    if np.abs(power_tape_extension_step) <= tol:
        return 0.0, False

    # Always move toward target and clamp to avoid overshoot.
    increment = np.sign(remaining) * min(
        np.abs(power_tape_extension_step), np.abs(remaining)
    )
    return increment, True


def _find_kite_fem_spring_id_from_connectivity(
    kite_fem_structure,
    kite_connectivity_arr,
    connectivity_idx,
):
    """
    Map ASKITE connectivity index to the matching kite_fem spring element index.
    """
    ci, cj = [int(v) for v in kite_connectivity_arr[connectivity_idx]]
    target_key = (min(ci, cj), max(ci, cj))

    for spring_id, spring_element in enumerate(kite_fem_structure.spring_elements):
        n1 = int(spring_element.spring.n1)
        n2 = int(spring_element.spring.n2)
        if (min(n1, n2), max(n1, n2)) == target_key:
            return spring_id

    raise ValueError(
        f"Could not map power_tape connectivity index {connectivity_idx} "
        f"with nodes ({ci}, {cj}) to a kite_fem spring element."
    )


def _canopy_triangles(config, billow_structure):
    """Canopy element connectivity, when the backend has one to offer."""
    if config.get("structural_solver") != "billow" or billow_structure is None:
        return None
    from awetrim.aerostructural.billow import structural_billow as _sb

    return billow_structure.model.element_set(_sb.CANOPY).connectivity


def _map_aero_to_structure(
    config, f_aero_wing_vsm_format, struc_nodes, results_aero, aero2struc_mapping,
    canopy_sections, strut_sections, panels, billow_structure, *,
    is_with_conservation_check,
):
    """The coupled loop's ONE aero -> structure load mapping.

    Both call sites -- the pre-loop solve and every coupled iteration -- go
    through here. They used to be two copies of the same call, and when the
    billow canopy mesh was added to one and not the other, only the first
    structural solve got the element-consistent traction transfer; every
    later iteration silently fell back to point loads on the YAML's node
    chains, which never reach a quad-centre node. One call cannot drift.
    """
    canopy_triangles = _canopy_triangles(config, billow_structure)
    return aero2struc.main(
        config["aero2struc"]["coupling_method"],
        f_aero_wing_vsm_format,
        struc_nodes,
        np.array(results_aero["panel_cp_locations"]),
        aero2struc_mapping,
        config["is_with_coupling_plot_per_iteration"],
        config["aero2struc"],
        canopy_sections,
        strut_sections,
        panels,
        # Billow knows its canopy elements, so the load can be transferred
        # through them rather than through the YAML's node chains -- which is
        # the only route that reaches a quad-centre or refined interior node.
        canopy_triangles=canopy_triangles,
        canopy_grid=billow_structure.fine_grid if canopy_triangles is not None else None,
        is_with_conservation_check=is_with_conservation_check,
        return_distributed_aero=True,
    )


def _shared_bridle_line_drag(
    config, struc_nodes, bridle_connectivity_arr, body_aero, attitude_deg,
):
    """The bridle drag the TRIM charged, handed to the structure unchanged.

    Not a second computation of the same load -- the same one. The VSM already
    integrates a force on every bridle segment inside its own force balance
    (``BodyAerodynamics.compute_line_aerodynamic_force``, summed into the
    global totals), so the attitude the trim solves is the attitude that load
    produces. Recomputing it on the structure's side, however carefully, means
    two numbers for one physical load, and any gap between them is a moment
    about the bridle point that NEITHER side balances -- the structure is
    pinned there, so it can only shed it by swinging. Three ways the two drifted
    apart, all removed by taking the VSM's own:

    * **the inflow.** The VSM charges the drag at ``va_ref_vector``, the
      area-weighted mean PANEL inflow, which carries the body rates and the
      attitude. ``va_vel_world`` is the translational apparent wind and
      ``vel_app`` a freestream frozen at ``initialize`` -- on the LEI-V3
      steered case 20.3 N against 76.3 N.
    * **the geometry.** The VSM's segments are the ones handed to it by
      ``_bridle_line_specs_for_vsm``, at the shape and orientation the trim
      saw; the structure's nodes have since been rotated by the attitude.
    * **the law.** Same closure either way here (config ``cd_cable`` 1.1 and
      ``cf_cable`` 0.01 equal the VSM's defaults, and ``initialize`` already
      substitutes the flat tape's drag-equivalent diameter), but nothing was
      keeping them equal.

    Each segment's force is rotated by the trim's roll/pitch/yaw into the frame
    the structural nodes are now in, and split equally between its end nodes --
    which puts the resultant at the segment midpoint, exactly where the VSM
    applies it, so the moment about the reference point is preserved too.

    Falls back to ``None`` when the VSM has no bridle (the caller then keeps the
    old independent path), so a geometry whose segments do not line up with the
    structure's is not silently mischarged.
    """
    if not config["is_with_aero_bridle"]:
        return np.zeros((len(struc_nodes), 3))
    lines = getattr(body_aero, "_bridle_line_system", None)
    if not lines or bridle_connectivity_arr is None:
        return None
    pairs = np.asarray(bridle_connectivity_arr, dtype=int).reshape(-1, 2)
    if len(lines) != len(pairs):
        return None

    # Per-segment inflow, NOT the wing's -- the single-sourced relation in
    # awetrim.aerodynamics.apparent_wind. Every station on a rotating body has
    # its own ``-omega x (r - r0)``, so charging one vector to all of them is
    # wrong wherever that vector was evaluated.
    #
    # Measured on the LEI-V3, 45 bridle segments, va_free 15.68 m/s,
    # omega = radial 0.714 + great-circle 0.125 rad/s:
    #
    #   inflow charged                     |F|        |M| about r0
    #   va_free (what everything else did) 78.9 N     419 N m     +8.3% / +10.7%
    #   per-segment (this)                 72.8 N     379 N m       --
    #
    # ``va_ref_vector``, which the VSM charged its own bridle at, is NOT the
    # wing's mean inflow: it is built from ``self._va``, the inflow as handed
    # to the va setter, before v_rot is added -- for a uniform freestream it IS
    # va_free, so it sits in the first row, not a third one.
    #
    # An earlier note here quoted 8.28 m/s of omega x r at the wing and
    # "roughly doubles"; that is |omega|*|r|, the bound the cross product
    # reaches only with r PERPENDICULAR to omega. The wing sits 2.6 deg off the
    # rotation axis (omega is dominantly the radial course rate and the wing is
    # almost straight out along that same radial), so omega x r is near its
    # MINIMUM there: area-weighted 1.99 m/s at the wing, 1.26 m/s at the bridle
    # midpoints. Both rows coincide when omega is zero, which is why unsteered
    # baselines are blind to the whole question.
    va_free, omega, r0 = inflow_state_of(body_aero)
    if va_free is None or np.linalg.norm(va_free) <= 0.0:
        return None

    rotation = _rotation_from_euler_deg(attitude_deg)
    forces = np.zeros((len(struc_nodes), 3), dtype=float)
    for (i, j), line in zip(pairs, lines):
        midpoint = 0.5 * (np.asarray(line[0], dtype=float) + np.asarray(line[1], dtype=float))
        va_seg = apparent_wind_at(va_free, omega, midpoint, r0)
        if np.linalg.norm(va_seg) <= 0.0:
            return None
        f = np.asarray(
            body_aero.compute_line_aerodynamic_force(
                va_seg,
                line,
                cd_cable=config["aerodynamic_bridle"]["cd_cable"],
                cf_cable=config["aerodynamic_bridle"]["cf_cable"],
                rho=config["rho"],
            ),
            dtype=float,
        )
        if not np.all(np.isfinite(f)):
            return None
        f = rotation @ f
        forces[i] += 0.5 * f
        forces[j] += 0.5 * f
    return forces


def _rotation_from_euler_deg(angle_deg):
    """The rotation ``rotate_geometry`` applies, as a matrix.

    Built by rotating the identity's columns through the same helper, so the
    two can never disagree about order or sign convention.
    """
    basis = rotate_geometry(np.eye(3), angle_deg=list(angle_deg))
    return np.asarray(basis, dtype=float).T


def _bridle_line_drag(
    config, struc_nodes, bridle_connectivity_arr, bridle_diameter_arr,
    results_aero, vel_app,
):
    """Bridle-line drag on the structural nodes, at the TRIM's apparent wind.

    The trim already carries this drag: the VSM body is built with
    ``bridle_path``, so ``calculate_results`` adds every bridle segment's force
    AND its moment about the reference point into the balance the attitude is
    solved from. What the structure is handed therefore has to be that same
    load -- anything else is a moment about the bridle point that neither side
    balances, and the structure is PINNED there, so it can only shed it by
    swinging rigidly.

    That is exactly what it did. These calls used to pass ``vel_app``, the
    freestream vector frozen in ``aerodynamic_vsm.initialize`` from
    ``wind_speed_wind_ref`` -- 8 m/s (the as_config default, whatever wind the
    run asked for) against an apparent 15.9. Measured on the LEI-V3 Billow
    case, 2026-09-12: 20.3 N instead of 76.3 N, leaving **-326 N m of pitch**
    about the bridle point and a **0.76 deg** rigid swing the trim rotated back
    every iteration -- the non-decaying rigid part that pinned Aitken at its
    0.05 floor and stopped steered states from settling. At the trim's own
    apparent wind the disagreement is 0.2 N m on a converged state. What is
    left of it is the attitude: these forces land on nodes the driver has
    rotated by the trim's roll/pitch/yaw, and the trim rotates its WING by
    that, not its bridle -- so the residue is proportional to the attitude
    INCREMENT and dies with the coupled iteration (16 N m from a cold 3.1 deg
    step, 0.2 N m once converged).

    ``va_vel_world`` is the apparent wind in the VSM frame despite its name
    (``solve_vsm_quasi_steady_trim`` stores the course-to-VSM transformed
    vector); ``vel_app`` remains the fallback for a result that lacks it.
    """
    if not config["is_with_aero_bridle"]:
        return np.zeros((len(struc_nodes), 3))
    va = np.asarray(results_aero.get("va_vel_world", vel_app), dtype=float)
    if va.shape != (3,) or not np.all(np.isfinite(va)) or np.linalg.norm(va) <= 0.0:
        va = np.asarray(vel_app, dtype=float)
    return aerodynamic_bridle_line_drag.main(
        struc_nodes,
        bridle_connectivity_arr,
        bridle_diameter_arr,
        va,
        config["rho"],
        config["aerodynamic_bridle"]["cd_cable"],
        config["aerodynamic_bridle"]["cf_cable"],
    )


def _bridle_line_specs_for_vsm(body_aero, bridle_connectivity_arr):
    """``(node_i, node_j, drag diameter)`` rows, so the VSM's bridle can deform.

    ``BodyAerodynamics.instantiate(bridle_path=...)`` bakes the bridle segments
    in as COORDINATES, and ``update_from_points`` refreshes the wings only --
    so without this the trim keeps charging bridle drag to the built shape
    while the structure (and the drag ``_bridle_line_drag`` gives it) follows
    the deformed one. On a steered kite that is not a small difference in the
    same place: the deformed bridle is asymmetric and carries a roll moment the
    built one cannot.

    The indices come from the reader, which owns the structural array; the
    diameters from the body's own segments, which ``aerodynamic_vsm.initialize``
    has already replaced with the DRAG-equivalent ones (a flat tape's projected
    width, not its area-equivalent circle). Both lists are the same parse of
    ``bridle_connections`` in the same order -- verified on the LEI-V3
    FEM_full geometry, where the two agree on all 87 segments to 7.6e-9 m --
    and the count check below catches a geometry where that stops holding.
    """
    lines = getattr(body_aero, "_bridle_line_system", None)
    if not lines or bridle_connectivity_arr is None:
        return None
    pairs = np.asarray(bridle_connectivity_arr, dtype=int).reshape(-1, 2)
    if len(lines) != len(pairs):
        logging.warning(
            "the VSM body has %d bridle segments and the structure %d: leaving "
            "the VSM's bridle drag on the BUILT shape, which costs a moment "
            "about the bridle point once the structure deforms",
            len(lines), len(pairs),
        )
        return None
    return [
        (int(i), int(j), float(line[2])) for (i, j), line in zip(pairs, lines)
    ]


def _billow_solved(config, billow_structure):
    """True when a Billow solve has produced a state worth recording."""
    return (
        config.get("structural_solver") == "billow"
        and billow_structure is not None
        and getattr(billow_structure, "last_solution", None) is not None
    )


def _rest_length(
    config, element_index, psystem, kite_fem_structure, billow_structure,
    kite_connectivity_arr,
):
    """Live rest length [m] of one element, in the reader's ordering."""
    solver = config.get("structural_solver")
    if solver == "pss":
        return float(psystem.extract_rest_length[element_index])
    if solver == "billow":
        return structural_billow.get_rest_length(billow_structure, element_index)
    if solver == "kite_fem":
        spring_id = _find_kite_fem_spring_id_from_connectivity(
            kite_fem_structure=kite_fem_structure,
            kite_connectivity_arr=kite_connectivity_arr,
            connectivity_idx=element_index,
        )
        return float(kite_fem_structure.spring_elements[spring_id].l0)
    raise ValueError(f"unknown structural_solver {solver!r}")


def _set_rest_length(
    config, element_index, rest_length, psystem, kite_fem_structure,
    billow_structure, kite_connectivity_arr,
):
    """Set one element's rest length [m], in the reader's ordering."""
    solver = config.get("structural_solver")
    if solver == "pss":
        psystem.update_rest_length(
            element_index,
            float(rest_length) - float(psystem.extract_rest_length[element_index]),
        )
    elif solver == "billow":
        structural_billow.set_rest_length(billow_structure, element_index, rest_length)
    elif solver == "kite_fem":
        spring_id = _find_kite_fem_spring_id_from_connectivity(
            kite_fem_structure=kite_fem_structure,
            kite_connectivity_arr=kite_connectivity_arr,
            connectivity_idx=element_index,
        )
        kite_fem_structure.modify_get_spring_rest_length(
            spring_ids=[spring_id], new_l0s=[float(rest_length)]
        )
    else:
        raise ValueError(f"unknown structural_solver {solver!r}")


def _current_power_tape_length(
    config, power_tape_index, psystem, kite_fem_structure, billow_structure,
    kite_connectivity_arr,
):
    """Live rest length of the depower tape [m], whichever backend owns it.

    NaN when the backend cannot be asked, which is better than a stale number:
    an actuated run walks the tape through several converged states inside one
    call, and mislabelling those states is worse than not labelling them.
    """
    if power_tape_index is None:
        return float("nan")
    try:
        return _rest_length(
            config, power_tape_index, psystem, kite_fem_structure,
            billow_structure, kite_connectivity_arr,
        )
    except Exception:  # a diagnostic must never take the run down
        return float("nan")


def remaining_drift(values, n_ratios=3):
    """Estimated change still to come in a converging sequence [its unit].

    Models the tail as geometric: with successive changes ``d_j`` and ratio
    ``r = |d_j / d_(j-1)|``, what remains after the last value is
    ``|d_last| r / (1 - r)``. The ratio is the LARGEST of the last
    ``n_ratios``, because two modes decaying at different rates fool a single
    ratio: a fast transient dying away onto a slow tail reads as r ~ 0 for one
    iteration (measured on the steered Billow kite: a 0.0007 rad/s course-rate
    change with 0.02 rad/s still to come). A growing or non-decaying tail
    (r >= 1), or too short a history, is infinitely far from settled.
    """
    values = np.asarray(values, dtype=float)
    if values.size < n_ratios + 2 or not np.all(np.isfinite(values)):
        return float("inf")
    changes = np.abs(np.diff(values[-(n_ratios + 2):]))
    if changes[-1] == 0.0:
        return 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = np.where(changes[:-1] > 0.0, changes[1:] / changes[:-1], np.inf)
    ratio = float(ratios.max())
    if ratio >= 1.0:
        return float("inf")
    return float(changes[-1] * ratio / (1.0 - ratio))


def update_steering_tape_actuation(
    config,
    psystem,
    kite_fem_structure,
    billow_structure,
    kite_connectivity_arr,
    steering_tape_indices,
    initial_lengths_steering,
    steering_tape_final_extension,
    steering_tape_extension_step,
):
    """Step the steering tape half-difference one increment toward its target.

    The first tape of ``steering_tape_indices`` is shortened and the second
    lengthened by the same amount, the convention of
    ``pss.actuation.update_steering_tape_actuation``, so a half-difference
    means the same thing on every backend. The half-difference is read back
    from the live rest lengths rather than counted, so the walk resumes
    correctly whatever set them. A zero step applies the whole target at once.

    Returns:
        tuple: (half_difference, is_finalized, did_update)
    """
    left, right = (int(index) for index in steering_tape_indices[:2])
    left_0, right_0 = (float(length) for length in initial_lengths_steering)

    def current():
        lengths = [
            _rest_length(config, index, psystem, kite_fem_structure,
                         billow_structure, kite_connectivity_arr)
            for index in (left, right)
        ]
        return 0.5 * ((left_0 - lengths[0]) + (lengths[1] - right_0))

    target = float(steering_tape_final_extension)
    step = float(steering_tape_extension_step) or target
    half_difference = current()
    increment, should_update = _compute_power_tape_increment(
        half_difference, target, step
    )
    if not should_update:
        return half_difference, True, False

    half_difference += increment
    for index, length in ((left, left_0 - half_difference),
                          (right, right_0 + half_difference)):
        _set_rest_length(config, index, length, psystem, kite_fem_structure,
                         billow_structure, kite_connectivity_arr)
    logging.info(
        "||--- steering half-difference %.4f m (target %.4f m) | left %.4f m, "
        "right %.4f m",
        half_difference, target, left_0 - half_difference, right_0 + half_difference,
    )
    _, remaining = _compute_power_tape_increment(half_difference, target, step)
    return half_difference, not remaining, True


def update_power_tape_actuation(
    config,
    psystem,
    kite_fem_structure,
    kite_connectivity_arr,
    billow_structure,
    power_tape_index,
    power_tape_extension_step,
    initial_length_power_tape,
    power_tape_final_extension,
    is_residual_below_tol,
    n_power_tape_steps,
    rest_lengths=None,
):
    """
    Calculate current power tape extension and update if needed for actuation.

    Args:
        config: Configuration dictionary
        psystem: Particle system (for PSS solver)
        kite_fem_structure: FEM structure (for kite_fem solver)
        billow_structure: BillowStructure (for the billow solver)
        kite_connectivity_arr: ASKITE connectivity array
        power_tape_index: Index of power tape in connectivity array
        power_tape_extension_step: Increment for power tape extension
        initial_length_power_tape: Initial length of power tape
        power_tape_final_extension: Final desired power tape extension
        is_residual_below_tol: Flag indicating if residual is below tolerance
        n_power_tape_steps: Number of power tape extension steps
        rest_lengths: Current rest lengths array (for kite_fem solver)

    Returns:
        tuple: (delta_power_tape, is_actuation_finalized)
            - delta_power_tape: Current change in power tape length
            - is_actuation_finalized: True if actuation is complete, False otherwise
    """
    is_actuation_finalized = True

    ## Calculate delta tape lengths based on structural solver
    if config["structural_solver"] == "pss":
        current_length = float(psystem.extract_rest_length[power_tape_index])
        delta_power_tape = current_length - initial_length_power_tape

        if is_residual_below_tol:
            increment, should_update = _compute_power_tape_increment(
                delta_power_tape=delta_power_tape,
                power_tape_final_extension=power_tape_final_extension,
                power_tape_extension_step=power_tape_extension_step,
            )
            if should_update:
                psystem.update_rest_length(power_tape_index, increment)
                current_length = float(psystem.extract_rest_length[power_tape_index])
                delta_power_tape = current_length - initial_length_power_tape
                logging.info(
                    f"||--- delta l_d: {delta_power_tape:.3f}m | new l_d: {current_length:.3f}m | Steps required: {n_power_tape_steps}"
                )
                is_actuation_finalized = False

    elif config["structural_solver"] == "kite_fem":
        if kite_connectivity_arr is None:
            raise ValueError(
                "kite_connectivity_arr is required for kite_fem power tape actuation."
            )

        spring_id = _find_kite_fem_spring_id_from_connectivity(
            kite_fem_structure=kite_fem_structure,
            kite_connectivity_arr=kite_connectivity_arr,
            connectivity_idx=power_tape_index,
        )
        current_length = float(kite_fem_structure.spring_elements[spring_id].l0)
        delta_power_tape = current_length - initial_length_power_tape

        if is_residual_below_tol:
            increment, should_update = _compute_power_tape_increment(
                delta_power_tape=delta_power_tape,
                power_tape_final_extension=power_tape_final_extension,
                power_tape_extension_step=power_tape_extension_step,
            )
            if should_update:
                new_length = current_length + increment
                kite_fem_structure.modify_get_spring_rest_length(
                    spring_ids=[spring_id],
                    new_l0s=[new_length],
                )
                delta_power_tape = new_length - initial_length_power_tape
                logging.info(
                    f"||--- delta l_d: {delta_power_tape:.3f}m | new l_d: {new_length:.3f}m | Steps required: {n_power_tape_steps}"
                )
                is_actuation_finalized = False

    elif config["structural_solver"] == "billow":
        # Rest lengths are NLP parameters, so this rebuilds the parameter table
        # and never the compiled graph; the element index is the reader's own,
        # so power_tape_index needs no translation.
        current_length = structural_billow.get_rest_length(
            billow_structure, power_tape_index
        )
        delta_power_tape = current_length - initial_length_power_tape

        if is_residual_below_tol:
            increment, should_update = _compute_power_tape_increment(
                delta_power_tape=delta_power_tape,
                power_tape_final_extension=power_tape_final_extension,
                power_tape_extension_step=power_tape_extension_step,
            )
            if should_update:
                structural_billow.update_rest_length(
                    billow_structure, power_tape_index, increment
                )
                current_length = structural_billow.get_rest_length(
                    billow_structure, power_tape_index
                )
                delta_power_tape = current_length - initial_length_power_tape
                logging.info(
                    f"||--- delta l_d: {delta_power_tape:.3f}m | new l_d: {current_length:.3f}m | Steps required: {n_power_tape_steps}"
                )
                is_actuation_finalized = False

    return delta_power_tape, is_actuation_finalized


def log_top_external_force_nodes(
    struc_nodes,
    m_arr,
    f_ext,
    f_aero,
    f_inertial,
    f_ext_gravity,
    top_n=5,
    tag="",
):
    """Log nodes with largest total external force and component breakdown."""
    f_ext = np.asarray(f_ext, dtype=float).reshape(-1, 3)
    f_aero = np.asarray(f_aero, dtype=float).reshape(-1, 3)
    f_inertial = np.asarray(f_inertial, dtype=float).reshape(-1, 3)
    f_ext_gravity = np.asarray(f_ext_gravity, dtype=float).reshape(-1, 3)
    masses = np.asarray(m_arr, dtype=float).reshape(-1)

    norms = np.linalg.norm(f_ext, axis=1)
    if len(norms) == 0:
        return

    top_n = int(max(1, min(top_n, len(norms))))
    top_idx = np.argsort(norms)[-top_n:][::-1]

    header = f"Top external-force nodes {tag}".strip()
    logging.info(header)
    for idx in top_idx:
        logging.info(
            "  node=%d pos=[%.3f, %.3f, %.3f] m=%.4fkg |f_ext|=%.3fN |f_aero|=%.3fN |f_inertial|=%.3fN |f_gravity|=%.3fN",
            int(idx),
            float(struc_nodes[idx][0]),
            float(struc_nodes[idx][1]),
            float(struc_nodes[idx][2]),
            float(masses[idx]) if idx < len(masses) else float("nan"),
            float(np.linalg.norm(f_ext[idx])),
            float(np.linalg.norm(f_aero[idx])),
            float(np.linalg.norm(f_inertial[idx])),
            float(np.linalg.norm(f_ext_gravity[idx])),
        )


# TODO: this should also use structural is not converging
def check_convergence(
    i,
    f_residual,
    f_residual_list,
    f_aero_wing_vsm_format,
    config,
    stagnation_check_start=0,
):
    """
    Check convergence conditions for the aero-structural solver.

    Args:
        i: Current iteration number
        f_residual: Current residual force vector
        f_residual_list: List of residual force norms from all iterations
        f_aero_wing_vsm_format: Aerodynamic forces in VSM format
        config: Configuration dictionary
        stagnation_check_start: Iteration index from which to check stagnation
            (reset when switching regularization phase)

    Returns:
        tuple: (is_convergence, should_break, is_stagnated)
            - is_convergence: True if converged, False otherwise
            - should_break: True if loop should break, False to continue
            - is_stagnated: True if residual has stagnated (no longer changing)
    """
    is_convergence = False
    should_break = False
    is_stagnated = False

    n_stag = config["aero_structural_solver"].get("n_max_constant_residual_force", 15)
    # Number of iterations since the stagnation check window started
    iters_since_start = i - stagnation_check_start

    ### All the convergence checks, are be done in if-elif because only 1 should hold at once
    # if convergence (residual below set tolerance)
    if np.linalg.norm(f_residual) <= config["aero_structural_solver"]["tol"]:
        is_convergence = True

    # if residual forces are NaN
    elif np.isnan(np.linalg.norm(f_residual)):
        is_convergence = False
        logging.info("Classic PS diverged - residual force is NaN")
        should_break = True

    # if residual forces are not changing anymore (compare start of window vs current)
    elif iters_since_start > n_stag and np.abs(
        f_residual_list[i - n_stag] - f_residual_list[i]
    ) < config["aero_structural_solver"].get("stagnation_tol", 1.0):
        is_convergence = False
        is_stagnated = True

    # if too many iterations are needed
    elif i > config["aero_structural_solver"]["max_iter"]:
        is_convergence = False
        logging.info(
            f"Classic PS non-converging - more than max ({config['aero_structural_solver']['max_iter']}) iterations needed"
        )
        should_break = True

    # special case for running the simulation for only one timestep
    elif config["is_run_only_1_time_step"]:
        should_break = True

    # when aero does not converge
    elif np.sum([force[1] for force in f_aero_wing_vsm_format]) == np.nan:
        is_convergence = False
        logging.info("Classic PS non-converging - aero forces are NaN")
        should_break = True

    return is_convergence, should_break, is_stagnated


def main(
    m_arr=None,
    struc_nodes=None,
    struc_nodes_initial=None,
    system_model=None,
    config=None,
    ### ACTUATION
    initial_length_power_tape=None,
    n_power_tape_steps=None,
    power_tape_final_extension=None,
    power_tape_extension_step=None,
    ### CONNECTIVITY
    kite_connectivity_arr=None,
    bridle_connectivity_arr=None,
    pulley_line_indices=None,
    pulley_line_to_other_node_pair_dict=None,
    ### STRUC --> AERO
    struc_node_le_indices=None,
    struc_node_te_indices=None,
    ### AERO
    body_aero=None,
    vsm_solver=None,
    vel_app=None,
    initial_polar_data=None,
    bridle_diameter_arr=None,
    ### AERO --> STRUC
    aero2struc_mapping=None,
    power_tape_index=None,
    steering_tape_indices=None,
    ### STRUC
    psystem=None,
    kite_fem_structure=None,
    billow_structure=None,
    canopy_sections=None,
    strut_sections=None,
):
    """
    Runs the aero-structural solver for the given input parameters.

    Args:
        config (dict): Main configuration dictionary.
        PROJECT_DIR (Path): Path to the project directory.
        results_dir (Path): Path to the results directory.
        steering_tape_indices (list[int] or None): The two steering tapes, in
            the reader's ordering. Steering is actuated ONLY when these are
            passed: ``steering_tape_final_extension`` in the config is ignored
            otherwise, which keeps every caller written before this driver
            could steer on its old behaviour (as_config ships 0.3 m of it).
            The half-difference walks toward the target in
            ``steering_tape_extension_step`` increments, each taken from a
            converged state once the depower walk has arrived, and each
            followed by ``steering_settle_iterations_after_update`` iterations
            in which no convergence exit is allowed.

    Returns:
        tracking_data (dict): Dictionary containing time histories of positions, forces, etc.
        meta (dict): Dictionary with meta information about the simulation (timing, convergence, etc).
    """

    print(f'--> Running structural_solver: {config["structural_solver"]}')

    ## PRELOOP
    if config["is_with_gravity"]:
        f_ext_gravity_default = np.array(
            [np.array(config["grav_constant"]) * m_pt for m_pt in m_arr]
        )
    else:
        f_ext_gravity_default = np.zeros(struc_nodes.shape)

    if config["structural_solver"] == "kite_fem":
        rest_lengths = kite_fem_structure.modify_get_spring_rest_length()
    elif config["structural_solver"] == "billow":
        rest_lengths = structural_billow.get_rest_lengths(
            billow_structure, kite_connectivity_arr
        )

    max_iter = config["aero_structural_solver"]["max_iter"]
    # Keep index 0 for the pre-loop initial state and reserve max_iter loop slots.
    t_vector = np.linspace(0, max_iter, max_iter + 1)
    # Store the nodal frames for a backend whose elements carry rotational
    # state. Without them a saved run cannot be re-evaluated at all: a
    # geometrically exact beam's strain energy depends on the frames, not just
    # the node positions, so any post-hoc energy or curvature computed from
    # positions alone is wrong -- by orders of magnitude, not a little.
    tracking_data = tracking.setup_tracking_arrays(
        len(struc_nodes), t_vector,
        n_frames=(
            len(billow_structure.state.frames)
            if config.get("structural_solver") == "billow"
            and billow_structure is not None
            else 0
        ),
    )
    is_convergence = False
    f_residual_list = []
    f_tether_drag = np.zeros(3)
    is_residual_below_tol = False
    struc_nodes_prev = None  # Initialize previous points for tracking
    start_time = time.time()
    plotting.set_plot_style()

    # Two-phase regularization: phase 1 = with pseudo_dt, phase 2 = without
    reg_phase = 1  # 1 = regularized, 2 = unregularized (polish)
    stagnation_check_start = 0  # iteration at which current phase started

    # Aitken relaxation state
    omega_relaxation = config["aero_structural_solver"].get("relaxation_factor", 0.3)
    r_prev_flat = None

    # Build symmetry mapping once from initial (undeformed) geometry
    if config["is_with_forcing_symmetry"]:
        symmetry_mapping = build_symmetry_mapping(struc_nodes_initial)

    # Steering walk. Opt-in by passing the tape indices, see the docstring.
    steering_tape_final_extension = float(
        config.get("steering_tape_final_extension", 0.0) or 0.0
    )
    is_steering_active = (
        steering_tape_indices is not None
        and len(steering_tape_indices) >= 2
        and abs(steering_tape_final_extension) > 1e-9
    )
    is_steering_finalized = not is_steering_active
    steering_half_difference = 0.0 if is_steering_active else float("nan")
    initial_lengths_steering = None
    if is_steering_active:
        initial_lengths_steering = [
            _rest_length(config, int(index), psystem, kite_fem_structure,
                         billow_structure, kite_connectivity_arr)
            for index in steering_tape_indices[:2]
        ]
    # A tape half-difference first moves the geometry by millimetres; the
    # rolled, turning equilibrium grows out of that asymmetry over several
    # coupled iterations, and meanwhile the residual can sit BELOW the gate.
    # Without a hold the loop exits on the still-symmetric state (measured on
    # the PSS path: u_s 0.025/0.05 rows "converged" in 3 iterations at roll 0).
    # So after a steering step an exit needs, besides the residual gate:
    #   - steering_settle_iterations_after_update iterations, a MINIMUM that
    #     covers the dead band before the course rate starts to move, and
    #   - the trim's course rate settled: its estimated remaining drift
    #     (remaining_drift) within steering_settle_course_rate_tol [rad/s].
    # The residual cannot stand in for the second: it is the change in load
    # between iterations, so a slowly relaxing loop meets the gate while the
    # steered state is still growing (Billow, Aitken at its floor: residual
    # 0.09 N with the course rate 6% short).
    steering_settle_iterations = max(
        0,
        int(config["aero_structural_solver"].get(
            "steering_settle_iterations_after_update", 6
        )),
    )
    steering_settle_course_rate_tol = float(
        config["aero_structural_solver"].get("steering_settle_course_rate_tol", 1e-3)
    )
    # Optional RELATIVE part, a fraction of |course rate|: the course rate
    # nearly doubles from 5 to 15 cm of tape, so one absolute tolerance means
    # a different accuracy at every steering input. The drift is accepted when
    # it is within max(tol, rtol * |course rate|). 0 = absolute only.
    steering_settle_course_rate_rtol = float(
        config["aero_structural_solver"].get("steering_settle_course_rate_rtol", 0.0)
    )
    steering_settle_counter = 0
    course_rate_since_step = None  # None until the first steering step
    course_rate_remaining = float("nan")

    ## track initial state
    # Update unified tracking dataframe (replaces position update)
    tracking.update_tracking_arrays(
        tracking_data,
        0,
        struc_nodes,
        np.zeros(np.shape(struc_nodes.flatten())),
        np.zeros(np.shape(struc_nodes.flatten())),
    )

    ######################################################################
    # Initialization of external forces pre-simulation loop
    ######################################################################

    # Node indices + drag diameters of the VSM's bridle segments, so every aero
    # call can rebuild them on the live structure instead of the built shape.
    bridle_line_specs = _bridle_line_specs_for_vsm(
        body_aero, bridle_connectivity_arr
    )

    ### STRUC --> AERO
    _update = _STRUC_TO_AERO_MAPPER.map(
        struc_nodes,
        struc_node_le_indices,
        struc_node_te_indices,
        config["aerodynamic"]["n_aero_panels_per_struc_section"],
    )
    le_arr, te_arr = _update.leading_edge_points, _update.trailing_edge_points

    cg = calculate_cg(struc_nodes=struc_nodes, m_arr=m_arr)

    ### AERO
    f_aero_wing_vsm_format, body_aero, results_aero = aerodynamic_vsm.run_vsm_package(
        body_aero=body_aero,
        solver=vsm_solver,
        system_model=system_model,
        center_of_gravity=cg,
        le_arr=le_arr,
        te_arr=te_arr,
        aero_input_type="reuse_initial_polar_data",
        initial_polar_data=initial_polar_data,
        include_gravity=config["is_with_gravity"],
        is_with_plot=config["is_with_aero_plot_per_iteration"],
        # Keep the bridle drag the trim charges on the same segments the
        # structure carries it on (see _bridle_line_specs_for_vsm).
        struc_nodes=struc_nodes,
        bridle_line_specs=bridle_line_specs,
    )

    logging.debug(
        f"Aero symmetry check, f_aero_y: {np.sum([force[1] for force in f_aero_wing_vsm_format])}"
    )
    roll, pitch, yaw = results_aero["opt_x"][1:4]
    struc_nodes = rotate_geometry(struc_nodes, angle_deg=[roll, pitch, yaw])
    # The pre-loop rotation is part of the attitude too: positions[0] is the
    # geometry BEFORE it, so composing rows 0..n recovers the kite's attitude.
    tracking_data["trim_state"][0] = np.ravel(results_aero["opt_x"])[:5]
    tracking_data["steering_half_difference"][0] = steering_half_difference

    # # TODO: debuggin here
    # # print out the cp locations as % of le to te for debugging purposes
    # alpha_arr = np.array(results_aero["alpha_at_ac"]).ravel()
    # alpha_arr_geom = np.array(results_aero["alpha_geometric"]).ravel()
    # cl_arr = np.array(results_aero["cl_distribution"]).ravel()
    # for i, (panel, alpha, alpha_geom, cl) in enumerate(
    #     zip(body_aero.panels, alpha_arr, alpha_arr_geom, cl_arr)
    # ):
    #     cp = np.array(results_aero["panel_cp_locations"][i])
    #     le_mid = 0.5 * (panel.LE_point_1 + panel.LE_point_2)
    #     te_mid = 0.5 * (panel.TE_point_1 + panel.TE_point_2)
    #     # chordwise fraction along panel chord axis
    #     cp_rel = np.dot(cp - le_mid, panel.y_airf) / panel.chord
    #     print(
    #         f"i:{i}, CP: {cp_rel:.3f}, alpha_corr: {alpha:.2f}deg, alpha_geom: {alpha_geom:.2f}deg, cl: {cl:.3f}, le: {le_mid}, te: {te_mid}"
    #     )
    #     F = np.array(results_aero["F_distribution"][i])
    #     M = np.array(results_aero["M_distribution"][i])
    #     ac = panel.aerodynamic_center
    #     y_airf = panel.y_airf
    #     z_airf = panel.z_airf
    #     c = panel.chord

    #     r = ac  # reference_point is [0,0,0] in config
    #     M_local = M - np.cross(r, F)
    #     m_pitch = np.dot(M_local, z_airf)
    #     F_perp = F - np.dot(F, z_airf) * z_airf
    #     F_perp_mag = np.linalg.norm(F_perp)
    #     lever_raw = m_pitch / max(F_perp_mag, 1e-12)
    #     lever_clamped = np.clip(lever_raw, -0.25 * c, 0.75 * c)
    #     cp_rel = 0.25 + lever_clamped / c

    #     print(
    #         f"i:{i}, cp_rel:{cp_rel:.3f}, lever_raw:{lever_raw/c:.3f}, F_perp:{F_perp_mag:.3e}"
    #     )

    #     cd, cm = panel.compute_cd_cm(alpha)
    #     print(
    #         f"i:{i}, cm:{cm:.4f}, alpha_corr:{alpha:.2f}, alpha_geom:{alpha_geom:.2f}, cp:{cp_rel:.3f}"
    #     )

    ### AERO --> STRUC
    f_aero_wing, aero_mapping_debug = _map_aero_to_structure(
        config, f_aero_wing_vsm_format, struc_nodes, results_aero, aero2struc_mapping,
        canopy_sections, strut_sections, body_aero.panels, billow_structure,
        is_with_conservation_check=False,
    )

    # Check moment preservation of aero→struc mapping (pre-loop)
    aero2struc.check_moment_preservation(
        f_aero_panel=aero_mapping_debug["forces"],
        panel_cps=aero_mapping_debug["points"],
        f_aero_mapped=f_aero_wing,
        struc_nodes=struc_nodes,
    )

    ### BRIDLE AERO
    # Same shared load as in the loop -- a different bridle drag at iteration 0
    # would be a step the loop then has to absorb.
    f_aero_bridle = _shared_bridle_line_drag(
        config, struc_nodes, bridle_connectivity_arr, body_aero,
        (roll, pitch, yaw),
    )
    if f_aero_bridle is None:
        f_aero_bridle = _bridle_line_drag(
            config, struc_nodes, bridle_connectivity_arr, bridle_diameter_arr,
            results_aero, vel_app,
        )
    inertial_force_total = np.asarray(
        results_aero.get("inertial_force", np.zeros(3)), dtype=float
    )
    f_inertial = distribute_total_force_by_particle_mass(inertial_force_total, m_arr)
    if config["is_with_gravity"] and ("gravity_force" in results_aero):
        gravity_force_total = np.asarray(
            results_aero.get("gravity_force", np.zeros(3)), dtype=float
        )
        f_ext_gravity = distribute_total_force_by_particle_mass(
            gravity_force_total, m_arr
        )
    else:
        f_ext_gravity = np.array(f_ext_gravity_default, copy=True)
    f_aero = f_aero_wing + f_aero_bridle
    ## EXTERNAL FORCE
    f_ext = f_aero + f_inertial + f_ext_gravity
    f_ext = np.round(f_ext, 5)
    f_ext_flat = f_ext.flatten()

    if config.get("aero_structural_solver", {}).get(
        "log_top_external_force_nodes", False
    ):
        log_top_external_force_nodes(
            struc_nodes=struc_nodes,
            m_arr=m_arr,
            f_ext=f_ext,
            f_aero=f_aero,
            f_inertial=f_inertial,
            f_ext_gravity=f_ext_gravity,
            top_n=int(
                config.get("aero_structural_solver", {}).get(
                    "log_top_external_force_nodes_n", 5
                )
            ),
            tag="(pre-loop)",
        )

    ######################################################################
    # SIMULATION LOOP
    ######################################################################
    ## propagating the simulation for each timestep and saving results
    with tqdm(total=max_iter, desc="Simulating", leave=True) as pbar:
        for i in range(max_iter):
            if i > 0:
                struc_nodes_prev = struc_nodes.copy()

            iter_start_time = time.time()

            ########################################################
            ############## INTERNAL FORCE CALCULATION ##############
            ########################################################
            begin_time_f_int = time.time()
            if config["structural_solver"] == "pss":
                psystem, is_structural_converged, struc_nodes, f_int = (
                    structural_pss.run_pss(
                        psystem,
                        f_ext_flat,
                        config["structural_pss"],
                    )
                )
            elif config["structural_solver"] == "kite_fem":
                kite_fem_structure, is_structural_converged, struc_nodes, f_int = (
                    structural_kite_fem.run_kite_fem(
                        kite_fem_structure, f_ext_flat, config["structural_kite_fem"]
                    )
                )
            elif config["structural_solver"] == "billow":
                billow_structure, is_structural_converged, struc_nodes, f_int = (
                    structural_billow.run_billow(
                        billow_structure, f_ext_flat, config.get("structural_billow")
                    )
                )
            end_time_f_int = time.time()

            ### Aitken relaxation of structural nodes
            if struc_nodes_prev is not None:
                r_k = struc_nodes - struc_nodes_prev
                r_k_flat = r_k.flatten()

                if (
                    config["aero_structural_solver"].get(
                        "is_with_aitken_relaxation", True
                    )
                    and r_prev_flat is not None
                ):
                    delta_r = r_k_flat - r_prev_flat
                    denom = np.dot(delta_r, delta_r)
                    if denom > 1e-30:
                        omega_relaxation = -omega_relaxation * (
                            np.dot(r_prev_flat, delta_r) / denom
                        )
                        omega_relaxation = np.clip(omega_relaxation, 0.05, 1.0)

                struc_nodes = struc_nodes_prev + omega_relaxation * r_k
                r_prev_flat = r_k_flat.copy()
                logging.debug(f"Aitken relaxation omega: {omega_relaxation:.4f}")

                # Sync relaxed positions back to structural solver state
                if config["structural_solver"] == "pss":
                    for idx, particle in enumerate(psystem.particles):
                        particle.update_pos(struc_nodes[idx])
                        particle.update_vel(np.zeros(3))
                elif config["structural_solver"] == "kite_fem":
                    # Update kite_fem so the next solve() starts from the
                    # Aitken-relaxed geometry instead of the original construction
                    # geometry.  coords_rotations_init is the reference that
                    # solve() adds displacements to, so moving it here makes the
                    # Newton-Raphson start near the current state.
                    flat_xyz = struc_nodes.flatten()
                    kite_fem_structure.coords_current = flat_xyz.copy()
                    # Build the 6-DOF vector [x,y,z, 0,0,0] per node
                    n_nodes = len(struc_nodes)
                    coords_rot = np.zeros(n_nodes * 6)
                    for ni in range(n_nodes):
                        coords_rot[6 * ni : 6 * ni + 3] = struc_nodes[ni]
                    kite_fem_structure.coords_rotations_init = coords_rot.copy()
                    kite_fem_structure.coords_rotations_current = coords_rot.copy()
                elif config["structural_solver"] == "billow":
                    # Seed the next minimum-energy solve from the relaxed
                    # geometry. The reference FRAMES are kept: they carry the
                    # absorbed rotation increments, and the element reference
                    # strains are measured against them, so replacing them with
                    # the relaxed positions alone would re-zero every beam.
                    billow_structure.state = dataclasses.replace(
                        billow_structure.state, positions=struc_nodes.copy()
                    )

            ### PLOT per iteration
            if config["is_with_struc_plot_per_iteration"]:
                if config["structural_solver"] == "pss":
                    rest_lengths = psystem.extract_rest_length
                elif config["structural_solver"] == "kite_fem":
                    rest_lengths = structural_kite_fem.get_rest_lengths(
                        kite_fem_structure, kite_connectivity_arr
                    )
                    # kite_fem_structure.plot_convergence()  # not available in kite_fem
                elif config["structural_solver"] == "billow":
                    rest_lengths = structural_billow.get_rest_lengths(
                        billow_structure, kite_connectivity_arr
                    )

                plotting.main(
                    struc_nodes,
                    kite_connectivity_arr,
                    rest_lengths,
                    f_ext=f_ext,
                    f_inertial=f_inertial,
                    title=f"i: {i}",
                    body_aero=body_aero,
                    is_with_node_indices=False,
                    pulley_line_indices=pulley_line_indices,
                    pulley_line_to_other_node_pair_dict=pulley_line_to_other_node_pair_dict,
                )

            ## external force
            begin_time_f_ext = time.time()

            ### STRUC --> AERO
            _update = _STRUC_TO_AERO_MAPPER.map(
                struc_nodes,
                struc_node_le_indices,
                struc_node_te_indices,
                config["aerodynamic"]["n_aero_panels_per_struc_section"],
            )
            le_arr, te_arr = _update.leading_edge_points, _update.trailing_edge_points

            cg = calculate_cg(struc_nodes=struc_nodes, m_arr=m_arr)

            ### AERO
            begin_time_aero_model = time.time()
            f_aero_wing_vsm_format, body_aero, results_aero = (
                aerodynamic_vsm.run_vsm_package(
                    body_aero=body_aero,
                    solver=vsm_solver,
                    system_model=system_model,
                    center_of_gravity=cg,
                    le_arr=le_arr,
                    te_arr=te_arr,
                    current_guess=[
                        results_aero["opt_x"][0],
                        0,
                        0,
                        0,
                        results_aero["opt_x"][4],
                    ],
                    aero_input_type="reuse_initial_polar_data",
                    initial_polar_data=initial_polar_data,
                    include_gravity=config["is_with_gravity"],
                    is_with_plot=config["is_with_aero_plot_per_iteration"],
                    # The bridle deforms with the kite, and on a steered one it
                    # is asymmetric: rebuild the VSM's segments on the current
                    # nodes so the trim charges the drag the structure carries.
                    struc_nodes=struc_nodes,
                    bridle_line_specs=bridle_line_specs,
                )
            )
            end_time_aero_model = time.time()
            logging.debug(
                f"Aero symmetry check, f_aero_y: {np.sum([force[1] for force in f_aero_wing_vsm_format])}"
            )
            print("Quasi-steady state solver info:")
            print(f"  Kite_speed: {results_aero['opt_x'][0]:.2f} m/s")
            print(f"  Roll: {results_aero['opt_x'][1]:.2f} deg")
            print(f"  Pitch: {results_aero['opt_x'][2]:.2f} deg")
            print(f"  Yaw: {results_aero['opt_x'][3]:.2f} deg")
            print(f"  Course rate: {results_aero['opt_x'][4]:.2f} rad/s")
            roll, pitch, yaw = results_aero["opt_x"][1:4]
            struc_nodes = rotate_geometry(struc_nodes, angle_deg=[roll, pitch, yaw])
            ### AERO --> STRUC
            f_aero_wing, aero_mapping_debug = _map_aero_to_structure(
                config, f_aero_wing_vsm_format, struc_nodes, results_aero,
                aero2struc_mapping, canopy_sections, strut_sections,
                body_aero.panels, billow_structure,
                is_with_conservation_check=(i == 0),
            )

            # Check moment preservation (only first coupling iteration to limit log spam)
            if i == 1:
                aero2struc.check_moment_preservation(
                    f_aero_panel=aero_mapping_debug["forces"],
                    panel_cps=aero_mapping_debug["points"],
                    f_aero_mapped=f_aero_wing,
                    struc_nodes=struc_nodes,
                )

            ### BRIDLE AERO
            # The load the trim already charged, not a second opinion on it.
            f_aero_bridle = _shared_bridle_line_drag(
                config, struc_nodes, bridle_connectivity_arr, body_aero,
                (roll, pitch, yaw),
            )
            if f_aero_bridle is None:
                f_aero_bridle = _bridle_line_drag(
                    config, struc_nodes, bridle_connectivity_arr,
                    bridle_diameter_arr, results_aero, vel_app,
                )
            inertial_force_total = np.asarray(
                results_aero.get("inertial_force", np.zeros(3)), dtype=float
            )
            f_inertial = distribute_total_force_by_particle_mass(
                inertial_force_total,
                m_arr,
            )
            if config["is_with_gravity"] and ("gravity_force" in results_aero):
                gravity_force_total = np.asarray(
                    results_aero.get("gravity_force", np.zeros(3)), dtype=float
                )
                f_ext_gravity = distribute_total_force_by_particle_mass(
                    gravity_force_total,
                    m_arr,
                )
            else:
                f_ext_gravity = np.array(f_ext_gravity_default, copy=True)
            f_aero = f_aero_wing + f_aero_bridle

            ## EXTERNAL FORCE
            f_ext = f_aero + f_inertial + f_ext_gravity
            f_ext = np.round(f_ext, 5)
            f_ext_flat = f_ext.flatten()
            end_time_f_ext = time.time()

            if config.get("aero_structural_solver", {}).get(
                "log_top_external_force_nodes", False
            ):
                log_top_external_force_nodes(
                    struc_nodes=struc_nodes,
                    m_arr=m_arr,
                    f_ext=f_ext,
                    f_aero=f_aero,
                    f_inertial=f_inertial,
                    f_ext_gravity=f_ext_gravity,
                    top_n=int(
                        config.get("aero_structural_solver", {}).get(
                            "log_top_external_force_nodes_n", 5
                        )
                    ),
                    tag=f"(iter={i})",
                )

            ### FORCING SYMMETRY
            if config["is_with_forcing_symmetry"]:
                logging.info("Forcing symmetry in y-direction")
                struc_nodes = forcing_symmetry(struc_nodes, symmetry_mapping)

            ### RESIDUAL
            f_residual = f_int + f_ext_flat

            # Zero out residual at fixed (constrained) nodes — their imbalance
            # is carried by the constraint reaction force, not by f_int.
            # Without this, the residual includes e.g. the weight of node 0
            # (~92 N) which can never converge to zero.
            if config["structural_solver"] == "pss":
                for fix_idx in config["structural_pss"]["fixed_point_indices"]:
                    f_residual[3 * fix_idx : 3 * fix_idx + 3] = 0.0

            f_residual_list.append(np.linalg.norm(np.abs(f_residual)))
            if config["structural_solver"] == "pss":
                logging.debug(
                    f"residual force in y-direction: {np.sum([f_residual[1::3]]):.3f}N"
                )

            ### TRACKING
            # Update unified tracking dataframe (replaces position update)
            # Use i+1 so that positions[0] retains the true initial geometry
            # stored in the pre-loop call.
            # Read the tape length from the STRUCTURAL BACKEND, not from the
            # `rest_lengths` local: that is only refreshed inside the
            # per-iteration plot block, so it is stale on a normal run and would
            # silently label every actuation step with the built length.
            _tape = _current_power_tape_length(
                config, power_tape_index, psystem, kite_fem_structure,
                billow_structure, kite_connectivity_arr,
            )
            tracking.update_tracking_arrays(
                tracking_data,
                i + 1,
                struc_nodes,
                f_ext_flat,
                f_residual,
                frames=(
                    billow_structure.last_solution.state.frames
                    if _billow_solved(config, billow_structure)
                    else None
                ),
                solved_positions=(
                    billow_structure.last_solution.state.positions
                    if _billow_solved(config, billow_structure)
                    else None
                ),
                tape_length=_tape,
                speed_apparent=float(
                    np.linalg.norm(
                        np.asarray(
                            results_aero.get("va_vel_world", [np.nan] * 3), dtype=float
                        )
                    )
                ),
                steering_half_difference=steering_half_difference,
                trim_state=results_aero["opt_x"],
            )

            ### PROGRESS BAR
            pbar.set_postfix(
                {
                    "res": f"{np.linalg.norm(f_residual):.3f}N",
                    "aero_model": f"{end_time_aero_model-begin_time_aero_model:.2f}s",
                    "struc_model": f"{end_time_f_int-begin_time_f_int:.2f}s",
                    "ext_total": f"{end_time_f_ext-begin_time_f_ext:.2f}s",
                    "iter": f"{time.time()-iter_start_time:.2f}s",
                }
            )
            pbar.update(1)

            ### CHECK CONVERGENCE
            is_convergence, should_break, is_stagnated = check_convergence(
                i=i,
                f_residual=f_residual,
                f_residual_list=f_residual_list,
                f_aero_wing_vsm_format=f_aero_wing_vsm_format,
                config=config,
                stagnation_check_start=stagnation_check_start,
            )

            # Two-phase regularization: on stagnation in phase 1, disable
            # pseudo_dt and continue to let the solver polish to true equilibrium.
            if is_stagnated:
                if reg_phase == 1 and config["structural_solver"] == "kite_fem":
                    reg_phase = 2
                    stagnation_check_start = i  # reset stagnation window
                    config["structural_kite_fem"]["pseudo_dt"] = None
                    logging.info(
                        f"Phase 1 stagnated at iter {i} "
                        f"(res={np.linalg.norm(f_residual):.1f}N). "
                        f"Switching to phase 2: pseudo_dt=None (no regularization)."
                    )
                else:
                    logging.info(
                        "Classic PS non-converging - residual no longer changes"
                    )
                    should_break = True

            ### STEERING SETTLE (see steering_settle_course_rate_tol)
            if course_rate_since_step is not None:
                course_rate_since_step.append(float(results_aero["opt_x"][4]))
                if steering_settle_counter > 0:
                    steering_settle_counter -= 1
                drift = remaining_drift(course_rate_since_step)
                drift_tolerance = max(
                    steering_settle_course_rate_tol,
                    steering_settle_course_rate_rtol * abs(course_rate_since_step[-1]),
                )
                is_steering_settled = (
                    steering_settle_counter == 0 and drift <= drift_tolerance
                )
                if is_convergence and not is_steering_settled:
                    logging.info(
                        "residual below the gate %d iteration(s) after a steering "
                        "step; held, course rate %.4f rad/s with %.2e rad/s "
                        "still to come (tolerance %.1e)",
                        len(course_rate_since_step),
                        course_rate_since_step[-1],
                        drift,
                        drift_tolerance,
                    )
                    is_convergence = False
                # What the rule accepted on, for the record: an early stop on a
                # slow tail should carry its own estimate of the distance left.
                course_rate_remaining = drift

            ### ACTUATION (only when converged)
            if is_convergence:
                # Update residual flag for actuation function
                is_residual_below_tol = is_convergence

                delta_power_tape, is_actuation_finalized = update_power_tape_actuation(
                    config=config,
                    psystem=psystem,
                    kite_fem_structure=kite_fem_structure,
                    billow_structure=billow_structure,
                    kite_connectivity_arr=kite_connectivity_arr,
                    power_tape_index=power_tape_index,
                    power_tape_extension_step=power_tape_extension_step,
                    initial_length_power_tape=initial_length_power_tape,
                    power_tape_final_extension=power_tape_final_extension,
                    is_residual_below_tol=is_residual_below_tol,
                    n_power_tape_steps=n_power_tape_steps,
                    rest_lengths=(
                        rest_lengths
                        if config["structural_solver"] in ("kite_fem", "billow")
                        else None
                    ),
                )

                # If actuation not finalized, continue to next iteration
                if not is_actuation_finalized:
                    # ACTUATION PHASE: Continue until power tape reaches final extension
                    continue

                # Steering walks once the depower has arrived, one step per
                # converged state, so each step starts from an equilibrium.
                if is_steering_active and not is_steering_finalized:
                    (
                        steering_half_difference,
                        is_steering_finalized,
                        did_update_steering,
                    ) = update_steering_tape_actuation(
                        config=config,
                        psystem=psystem,
                        kite_fem_structure=kite_fem_structure,
                        billow_structure=billow_structure,
                        kite_connectivity_arr=kite_connectivity_arr,
                        steering_tape_indices=steering_tape_indices,
                        initial_lengths_steering=initial_lengths_steering,
                        steering_tape_final_extension=steering_tape_final_extension,
                        steering_tape_extension_step=config.get(
                            "steering_tape_extension_step", 0.0
                        ),
                    )
                    if did_update_steering:
                        steering_settle_counter = steering_settle_iterations
                        course_rate_since_step = []
                        # The residuals before the step belong to another
                        # equilibrium; comparing across it can call the
                        # growing steered state "stagnated".
                        stagnation_check_start = i + 1
                        continue

            # Check if we should exit the loop
            if should_break or (
                is_convergence and is_actuation_finalized and is_steering_finalized
            ):
                break
    ######################################################################
    ## END OF SIMULATION FOR LOOP
    ######################################################################
    # print out the geometric angle of attack of the mid panel
    panels = body_aero.panels

    # Select middle panel
    mid_idx = len(panels) // 2
    panel = panels[mid_idx]

    # Midpoints of leading and trailing edges
    le_mid = 0.5 * (panel.LE_point_1 + panel.LE_point_2)
    te_mid = 0.5 * (panel.TE_point_1 + panel.TE_point_2)

    # Chord direction vector
    vec_chord = te_mid - le_mid
    vec_chord /= np.linalg.norm(vec_chord)

    # Apparent wind direction (normalize)
    vec_wind = vel_app / np.linalg.norm(vel_app)

    # Project onto plane of interest (optional: usually x-z plane)
    # Remove spanwise component if needed
    vec_chord_2d = np.array([vec_chord[0], vec_chord[2]])
    vec_wind_2d = np.array([vec_wind[0], vec_wind[2]])

    vec_chord_2d /= np.linalg.norm(vec_chord_2d)
    vec_wind_2d /= np.linalg.norm(vec_wind_2d)

    # Angle between vectors (signed)
    dot = np.clip(np.dot(vec_chord_2d, vec_wind_2d), -1.0, 1.0)
    cross = np.cross(vec_chord_2d, vec_wind_2d)

    angle = np.arctan2(cross, dot)

    print(f"alpha = {np.degrees(angle):.2f}° (va vs mid-span chord)")
    print(
        f'alpha = {float(np.rad2deg(results_aero["alpha_at_ac"][mid_idx])):.2f}° (incl. induced velocity, from results_aero["alpha_at_ac"])'
    )

    if config["structural_solver"] == "pss":
        rest_lengths = psystem.extract_rest_length
    elif config["structural_solver"] == "kite_fem":
        rest_lengths = structural_kite_fem.get_rest_lengths(
            kite_fem_structure, kite_connectivity_arr
        )
    elif config["structural_solver"] == "billow":
        rest_lengths = structural_billow.get_rest_lengths(
            billow_structure, kite_connectivity_arr
        )

    if config["is_with_final_plot"]:
        plotting.main(
            struc_nodes,
            kite_connectivity_arr,
            f_ext=f_ext,
            f_inertial=f_inertial,
            rest_lengths=rest_lengths,
            struc_nodes_initial=struc_nodes_initial,
            title="Initial vs final",
            pulley_line_indices=pulley_line_indices,
            pulley_line_to_other_node_pair_dict=pulley_line_to_other_node_pair_dict,
        )
    # Trim state of the converged solve. Without these a saved run cannot be
    # asked what flight condition it is, which a sweep targeting an apparent
    # speed needs: opt_x[0] is the tangential (kite) speed and opt_x[4] the
    # course rate, the same entries vsm_quasi_steady reports them as.
    _opt_x = np.asarray(results_aero.get("opt_x", []), dtype=float).ravel()
    meta = {
        "speed_tangential": float(_opt_x[0]) if _opt_x.size else float("nan"),
        "roll_body_deg": float(_opt_x[1]) if _opt_x.size > 1 else float("nan"),
        "course_rate": float(_opt_x[4]) if _opt_x.size > 4 else float("nan"),
        # NaN when the run did not steer (no tape indices passed).
        "steering_half_difference": float(steering_half_difference),
        # remaining_drift of the course rate at exit [rad/s]: the settle
        # rule's own estimate of how far the course rate still had to go.
        "course_rate_remaining": float(course_rate_remaining),
        "tether_force": float(results_aero.get("tether_force", float("nan"))),
        # va_vel_world is the apparent wind vector on the kite; "va" is a
        # nested payload key, not a result key, and reads back NaN.
        "speed_apparent": float(
            np.linalg.norm(
                np.asarray(
                    results_aero.get("va_vel_world", [np.nan] * 3), dtype=float
                )
            )
        ),
        "total_time_s": time.time() - start_time,
        # +1 for the pre-loop initial state at tracking index 0.
        # Each loop iteration appends one entry, so total stored frames are:
        # initial + number of completed loop iterations.
        "n_iter": len(f_residual_list) + 1,
        "converged": is_convergence,
        "rest_lengths": rest_lengths,  # ensure numeric array
        # Convert kite_connectivity to a numeric array for HDF5 compatibility
        "kite_connectivity": np.array(
            [[int(row[0]), int(row[1])] for row in np.array(kite_connectivity_arr)],
            dtype=np.int32,
        ),
    }

    return tracking_data, meta

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

import time
from tqdm import tqdm
import numpy as np
import logging
import matplotlib.pyplot as plt
from . import structural_pss
from .. import aerodynamic_vsm, aerodynamic_bridle_line_drag, tracking
from awetrim import plotting
from .actuation import (
    update_power_tape_actuation,
    update_steering_tape_actuation_progressive,
)
from ..convergence import (
    check_convergence,
    compute_adaptive_dt,
    element_elongations,
    max_element_elongation,
    relative_residual_norm,
    resolve_residual_tolerances,
    resultant_tether_force,
)
from ..forces import distribute_total_force_by_particle_mass
from ..mapping import (
    BilinearAeroToStructuralLoadMapper,
    LinearStructuralToAeroMapper,
    check_moment_preservation,
)
from ..protocols import AeroToStructureMap
from ..utils import (
    calculate_cg,
    rotate_geometry,
)


STRUCTURAL_TO_AERO_MAPPER = LinearStructuralToAeroMapper()
AERO_TO_STRUCTURAL_LOAD_MAPPER = BilinearAeroToStructuralLoadMapper()


def _map_structural_edges_to_aero(
    struc_nodes,
    struc_node_le_indices,
    struc_node_te_indices,
    n_aero_panels_per_struc_section,
):
    """Return aerodynamic leading/trailing-edge arrays from structural nodes."""
    update = STRUCTURAL_TO_AERO_MAPPER.map(
        struc_nodes,
        struc_node_le_indices,
        struc_node_te_indices,
        n_aero_panels_per_struc_section,
    )
    return update.leading_edge_points, update.trailing_edge_points


def _map_aero_loads_to_structure(
    f_aero_wing_vsm_format,
    struc_nodes,
    panel_cp_locations,
    aero2struc_mapping,
):
    """Return nodal aerodynamic loads from panel loads and a corner map."""
    mapping = AeroToStructureMap(panel_corner_map=np.asarray(aero2struc_mapping))
    return AERO_TO_STRUCTURAL_LOAD_MAPPER.map_loads(
        f_aero_wing_vsm_format,
        panel_cp_locations,
        struc_nodes,
        mapping,
    )


# Remove hardcoded values, when changing away from V3
def forcing_symmetry(struc_nodes):
    """
    Forcing symmetry in the y-direction for the kite structure nodes.
    This is a temporary solution to ensure symmetry in the simulation.
    """
    symmetry_pairs_dict = {
        1: 19,
        2: 20,
        3: 17,
        4: 18,
        5: 15,
        6: 16,
        7: 13,
        8: 14,
        9: 11,
        10: 12,
        # bridles
        21: 24,
        22: 23,
        25: 26,
        27: 30,
        28: 29,
        31: 32,
        33: 35,
        36: 37,
    }

    for key, value in symmetry_pairs_dict.items():
        struc_nodes[value] = np.array(
            [struc_nodes[key][0], -struc_nodes[key][1], struc_nodes[key][2]]
        )
    struc_nodes[34][1] = 0
    return struc_nodes


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
    initial_length_steering_left=None,
    initial_length_steering_right=None,
    steering_tape_indices=None,
    steering_tape_final_extension=None,
    steering_tape_extension_step=None,
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
    # (node_i, node_j, diameter) segments of the VSM bridle-line drag system
    # (aerodynamic_vsm.parse_bridle_line_specs); when given, the bridle drag
    # inside the VSM solve tracks the deforming structure instead of keeping
    # the initial geometry (update_from_points refreshes wings only).
    bridle_line_specs=None,
    ### AERO --> STRUC
    aero2struc_mapping=None,
    power_tape_index=None,
    ### STRUC
    psystem=None,
):
    """
    Runs the aero-structural solver for the given input parameters.

    Args:
        config (dict): Main configuration dictionary.
        PROJECT_DIR (Path): Path to the project directory.
        results_dir (Path): Path to the results directory.

    Returns:
        tracking_data (dict): Dictionary containing time histories of positions, forces, etc.
        meta (dict): Dictionary with meta information about the simulation (timing, convergence, etc).
    """
    print("--> Running structural solver: pss")

    ## PRELOOP
    f_ext_gravity = np.zeros(struc_nodes.shape)

    max_iter = config["aero_structural_solver"]["max_iter"]
    # Keep index 0 for the pre-loop initial state and reserve max_iter loop slots.
    t_vector = np.linspace(0, max_iter, max_iter + 1)
    n_panels = len(body_aero.panels) if body_aero is not None else 0
    tracking_data = tracking.setup_tracking_arrays(len(struc_nodes), t_vector, n_panels=n_panels)
    is_convergence = False
    # Coupled convergence is measured on the RELATIVE nodal force residual
    # ||f_int + f_ext|| / F_tether (see aerostructural/convergence.py), so this
    # history holds dimensionless norms. Resolving the tolerances here fails
    # fast on a config that still carries the legacy absolute [N] keys.
    residual_tol_relative, _ = resolve_residual_tolerances(
        config["aero_structural_solver"]
    )
    f_residual_list = []
    f_residual_relative = float("nan")
    f_tether_resultant = float("nan")
    f_tether_drag = np.zeros(3)
    # WING element stiffnesses are not known: following Poland & Schmehl they
    # are chosen so that no element elongates by more than `max_elongation`.
    # The loop enforces that -- an element over the bound is stiffened and the
    # coupled iteration continues, so a converged shape satisfies BOTH the
    # force residual and the elongation bound.
    #
    # Scope: WING elements only by default. A bridle line's stiffness is not a
    # free parameter (k = E*A/l0 from the line material and diameter), so
    # stiffening it to suppress its stretch would falsify the bridle; set
    # `stiffness_update_scope: all` to include them anyway.
    is_with_stiffness_update = bool(
        config["structural_pss"].get("update_stiffness", True)
    )
    stiffness_update_scope = str(
        config["structural_pss"].get("stiffness_update_scope", "wing")
    ).lower()
    elongation_bound = float(config["structural_pss"].get("max_elongation", 0.01))
    stiffness_update_factor = float(
        config["structural_pss"].get("stiffness_update_factor", 2.0)
    )
    max_stiffness = float(config["structural_pss"].get("max_stiffness", 5.0e5))
    # Wing elements are parsed first, bridle lines appended after them.
    n_wing_elements = len(kite_connectivity_arr) - (
        0 if bridle_connectivity_arr is None else len(bridle_connectivity_arr)
    )
    stiffness_update_indices = (
        None if stiffness_update_scope == "all" else range(n_wing_elements)
    )
    n_stiffness_updates = 0
    max_elongation_seen = float("nan")
    max_wing_elongation = float("nan")
    max_bridle_elongation = float("nan")
    # Stiffness CONTINUATION. A bridle imposed at its full (stiff) value in one
    # step moves the wing by centimetres of line length at once, which throws
    # the trim onto the stalled branch and never recovers. Ramping the target
    # stiffness in from `stiffness_ramp_start_factor` over
    # `stiffness_ramp_iterations` lets the shape re-equilibrate at each step.
    # `k_target` is what the elongation update adapts; the particle system runs
    # at `ramp * k_target` until the ramp completes.
    # Default ON: the shipped geometries carry the SK75 bridle, which cannot be
    # imposed in one step (it stalls the wing). Set 0 to disable.
    stiffness_ramp_iterations = int(
        config["structural_pss"].get("stiffness_ramp_iterations", 20)
    )
    stiffness_ramp_start_factor = float(
        config["structural_pss"].get("stiffness_ramp_start_factor", 0.1)
    )
    stiffness_ramp_scope = str(
        config["structural_pss"].get("stiffness_ramp_scope", "bridle")
    ).lower()
    k_target = structural_pss.get_stiffnesses(psystem)
    ramp_mask = np.zeros(len(k_target), dtype=bool)
    if stiffness_ramp_iterations > 0:
        if stiffness_ramp_scope == "all":
            ramp_mask[:] = True
        else:
            ramp_mask[n_wing_elements:] = True
        logging.info(
            "Stiffness continuation: %s elements ramped from %.3g to 1.0 over "
            "%s iterations",
            int(ramp_mask.sum()),
            stiffness_ramp_start_factor,
            stiffness_ramp_iterations,
        )
    stiffness_ramp_active = bool(ramp_mask.any())
    is_actuation_finalized = True
    is_steering_finalized = True
    struc_nodes_prev = None  # Initialize previous points for tracking
    start_time = time.time()
    plotting.set_plot_style()

    stagnation_check_start = 0  # iteration at which current phase started

    # Adaptive dt for PSS solver
    dt_initial = config["structural_pss"]["dt"]
    dt_max = config["structural_pss"].get(
        "dt_max", dt_initial * 10.0
    )  # Default to 10x initial dt

    # Quasi-steady stagnation stop: if rounded opt_x stops changing for N iterations.
    qs_stag_decimals = int(
        config["aero_structural_solver"].get("qs_state_stagnation_decimals", 3)
    )
    qs_stag_n_iter = int(
        config["aero_structural_solver"].get("qs_state_stagnation_n_iter", 0)
    )
    steering_actuation_interval_iters = int(
        config["aero_structural_solver"].get("steering_actuation_interval_iters", 5)
    )
    steering_actuation_interval_iters = max(1, steering_actuation_interval_iters)
    power_tape_actuation_interval_iters = int(
        config["aero_structural_solver"].get(
            "power_tape_actuation_interval_iters", steering_actuation_interval_iters
        )
    )
    power_tape_actuation_interval_iters = max(1, power_tape_actuation_interval_iters)
    depower_settle_iterations_after_update = int(
        config["aero_structural_solver"].get(
            "depower_settle_iterations_after_update", 2
        )
    )
    depower_settle_iterations_after_update = max(
        0, depower_settle_iterations_after_update
    )
    qs_opt_prev_rounded = None
    qs_stag_counter = 0
    qs_state_should_break = False
    depower_settle_counter = 0
    logging.info(
        "Steering actuation interval: every %s iterations",
        steering_actuation_interval_iters,
    )
    logging.info(
        "Depower actuation interval: every %s iterations",
        power_tape_actuation_interval_iters,
    )
    logging.info(
        "Depower settle iterations after update: %s",
        depower_settle_iterations_after_update,
    )

    bridle_node_pairs = None

    if config["is_with_aero_bridle"]:
        bridle_node_pairs = (
            aerodynamic_bridle_line_drag.build_bridle_node_pairs_from_line_system(
                struc_nodes,
                getattr(body_aero, "_bridle_line_system", None),
            )
        )

    # Aitken relaxation state
    omega_relaxation = config["aero_structural_solver"].get("relaxation_factor", 0.3)
    r_prev_flat = None

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

    ### STRUC --> AERO
    le_arr, te_arr = _map_structural_edges_to_aero(
        struc_nodes,
        struc_node_le_indices,
        struc_node_te_indices,
        config["aerodynamic"]["n_aero_panels_per_struc_section"],
    )

    cg = calculate_cg(struc_nodes=struc_nodes, m_arr=m_arr)
    ### AERO
    # Warm gamma continuation across the coupling iterations (see
    # run_vsm_package): the pre-loop solve is cold, every later trim seeds
    # from the previous iteration's converged circulation.
    gamma_seed_prev = None
    f_aero_wing_vsm_format, body_aero, results_aero = aerodynamic_vsm.run_vsm_package(
        body_aero=body_aero,
        solver=vsm_solver,
        system_model=system_model,
        center_of_gravity=cg,
        struc_nodes=struc_nodes,
        bridle_line_specs=bridle_line_specs,
        gamma_seed=gamma_seed_prev,
        le_arr=le_arr,
        te_arr=te_arr,
        # va_vector=vel_app,
        aero_input_type="reuse_initial_polar_data",
        initial_polar_data=initial_polar_data,
        include_gravity=config["is_with_gravity"],
        is_with_plot=config["is_with_aero_plot_per_iteration"],
        # Lets the trim carry the tether when
        # config["tether"]["include_in_trim"] is set.
        config=config,
    )
    logging.debug(
        f"Aero symmetry check, f_aero_y: {np.sum([force[1] for force in f_aero_wing_vsm_format])}"
    )
    tracking.update_aero_tracking(
        tracking_data, 0,
        results_aero.get("alpha_at_ac"),
        results_aero.get("stall_mask"),
    )
    gamma_seed_prev = results_aero.get("gamma_distribution", gamma_seed_prev)
    roll, pitch, yaw = results_aero["opt_x"][1:4]
    struc_nodes = rotate_geometry(struc_nodes, angle_deg=[roll, pitch, yaw])
    ### AERO --> STRUC
    f_aero_wing = _map_aero_loads_to_structure(
        f_aero_wing_vsm_format,
        struc_nodes,
        np.array(results_aero["panel_cp_locations"]),
        aero2struc_mapping,
    )

    # Check moment preservation of aero→struc mapping (pre-loop)
    check_moment_preservation(
        panel_forces=f_aero_wing_vsm_format,
        panel_points=np.array(results_aero["panel_cp_locations"]),
        nodal_forces=f_aero_wing,
        nodes=struc_nodes,
    )

    ### BRIDLE AERO
    # f_aero_bridle = aerodynamic_bridle_line_drag.main(
    #     struc_nodes,
    #     bridle_connectivity_arr,
    #     bridle_diameter_arr,
    #     vel_app,
    #     config["rho"],
    #     config["aerodynamic_bridle"]["cd_cable"],
    #     config["aerodynamic_bridle"]["cf_cable"],
    # )
    f_aero_bridle = np.zeros((len(struc_nodes), 3))
    f_inertial = distribute_total_force_by_particle_mass(
        results_aero.get("inertial_force", np.zeros(3)),
        m_arr,
    )
    f_ext_gravity = distribute_total_force_by_particle_mass(
        results_aero.get("gravity_force", np.zeros(3)),
        m_arr,
    )
    f_aero = f_aero_wing + f_aero_bridle
    ## EXTERNAL FORCE
    f_ext = f_aero + f_inertial + f_ext_gravity
    f_ext = np.round(f_ext, 5)
    f_ext_flat = f_ext.flatten()

    ######################################################################
    # SIMULATION LOOP
    ######################################################################
    ## propagating the simulation for each timestep and saving results
    with tqdm(total=max_iter, desc="Simulating", leave=True) as pbar:
        for i in range(max_iter):
            if i > 0:
                struc_nodes_prev = struc_nodes.copy()

            ########################################################
            ############## INTERNAL FORCE CALCULATION ##############
            ########################################################
            begin_time_f_int = time.time()
            # Apply adaptive dt based on convergence progress
            if len(f_residual_list) > 0:
                adaptive_dt = compute_adaptive_dt(
                    f_residual_list,
                    dt_initial,
                    dt_max,
                    residual_tol_relative,
                )
                config["structural_pss"]["dt"] = adaptive_dt
                logging.debug(
                    f"Adaptive dt updated: {adaptive_dt:.6f} "
                    f"(relative residual: {f_residual_list[-1]:.3e})"
                )
                print(f"Adaptive dt: {adaptive_dt:.6f} s at iteration {i}")
            if stiffness_ramp_active:
                ramp = structural_pss.stiffness_ramp_factor(
                    i, stiffness_ramp_iterations, stiffness_ramp_start_factor
                )
                structural_pss.set_stiffnesses(
                    psystem, np.where(ramp_mask, ramp * k_target, k_target)
                )
                if ramp >= 1.0:
                    stiffness_ramp_active = False
                    logging.info("Stiffness continuation complete at iteration %s", i)
            psystem, is_structural_converged, struc_nodes, f_int = (
                structural_pss.run_pss(
                    psystem,
                    f_ext_flat,
                    config["structural_pss"],
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

                # Sync relaxed positions back to structural solver state.
                for idx, particle in enumerate(psystem.particles):
                    particle.update_pos(struc_nodes[idx])
                    particle.update_vel(np.zeros(3))

            ### PLOT per iteration
            if config["is_with_struc_plot_per_iteration"]:
                rest_lengths = psystem.extract_rest_length

                plotting.main(
                    struc_nodes,
                    kite_connectivity_arr,
                    rest_lengths,
                    f_ext=f_ext,
                    f_bridle=f_aero_bridle if config["is_with_aero_bridle"] else None,
                    f_inertial=f_inertial,
                    title=f"i: {i}",
                    body_aero=body_aero,
                    is_with_node_indices=False,
                    pulley_line_indices=pulley_line_indices,
                    pulley_line_to_other_node_pair_dict=pulley_line_to_other_node_pair_dict,
                )

            ########################################################
            ############## INTERNAL FORCE CALCULATION ##############
            ########################################################
            begin_time_f_ext = time.time()

            ### STRUC --> AERO
            le_arr, te_arr = _map_structural_edges_to_aero(
                struc_nodes,
                struc_node_le_indices,
                struc_node_te_indices,
                config["aerodynamic"]["n_aero_panels_per_struc_section"],
            )

            cg = calculate_cg(struc_nodes=struc_nodes, m_arr=m_arr)
            ### AERO
            f_aero_wing_vsm_format, body_aero, results_aero = (
                aerodynamic_vsm.run_vsm_package(
                    body_aero=body_aero,
                    solver=vsm_solver,
                    system_model=system_model,
                    center_of_gravity=cg,
                    struc_nodes=struc_nodes,
                    bridle_line_specs=bridle_line_specs,
                    gamma_seed=gamma_seed_prev,
                    le_arr=le_arr,
                    te_arr=te_arr,
                    current_guess=[
                        results_aero["opt_x"][0],
                        0,
                        0,
                        0,
                        results_aero["opt_x"][4],
                    ],
                    # va_vector=vel_app,
                    aero_input_type="reuse_initial_polar_data",
                    initial_polar_data=initial_polar_data,
                    include_gravity=config["is_with_gravity"],
                    is_with_plot=config["is_with_aero_plot_per_iteration"],
                    # Lets the trim carry the tether when
                    # config["tether"]["include_in_trim"] is set.
                    config=config,
                )
            )
            logging.debug(
                f"Aero symmetry check, f_aero_y: {np.sum([force[1] for force in f_aero_wing_vsm_format])}"
            )
            print("Quasi-steady state solver info:")
            # print(f"  Converged: {results_aero['is_converged']}")
            print(f"  Kite_speed: {results_aero['opt_x'][0]:.2f} m/s")
            print(f"  Roll: {results_aero['opt_x'][1]:.2f} deg")
            print(f"  Pitch: {results_aero['opt_x'][2]:.2f} deg")
            print(f"  Yaw: {results_aero['opt_x'][3]:.2f} deg")
            print(f"  Course rate: {results_aero['opt_x'][4]:.2f} rad/s")

            # Stop if quasi-steady state vector has effectively frozen.
            if qs_stag_n_iter > 0:
                qs_opt_current = np.asarray(results_aero.get("opt_x", []), dtype=float)
                if qs_opt_current.size > 0:
                    qs_opt_current_rounded = np.round(qs_opt_current, qs_stag_decimals)
                    if (
                        qs_opt_prev_rounded is not None
                        and qs_opt_prev_rounded.shape == qs_opt_current_rounded.shape
                        and np.array_equal(qs_opt_prev_rounded, qs_opt_current_rounded)
                    ):
                        qs_stag_counter += 1
                    else:
                        qs_stag_counter = 0

                    qs_opt_prev_rounded = qs_opt_current_rounded.copy()

                    if qs_stag_counter >= qs_stag_n_iter:
                        qs_state_should_break = True
                        logging.info(
                            "Stopping: quasi-steady opt_x unchanged up to %s decimals for %s consecutive iterations. "
                            "opt_x_rounded=%s",
                            qs_stag_decimals,
                            qs_stag_n_iter,
                            qs_opt_current_rounded,
                        )
            gamma_seed_prev = results_aero.get("gamma_distribution", gamma_seed_prev)
            roll, pitch, yaw = results_aero["opt_x"][1:4]
            struc_nodes = rotate_geometry(struc_nodes, angle_deg=[roll, pitch, yaw])
            ### AERO --> STRUC
            f_aero_wing = _map_aero_loads_to_structure(
                f_aero_wing_vsm_format,
                struc_nodes,
                np.array(results_aero["panel_cp_locations"]),
                aero2struc_mapping,
            )

            # Check moment preservation (only first coupling iteration to limit log spam)
            if i == 1:
                check_moment_preservation(
                    panel_forces=f_aero_wing_vsm_format,
                    panel_points=np.array(results_aero["panel_cp_locations"]),
                    nodal_forces=f_aero_wing,
                    nodes=struc_nodes,
                )

            ### BRIDLE AERO
            if config["is_with_aero_bridle"]:
                f_aero_bridle = aerodynamic_bridle_line_drag.main(
                    struc_nodes,
                    bridle_connectivity_arr,
                    bridle_diameter_arr,
                    vel_app,
                    config["rho"],
                    config["aerodynamic_bridle"]["cd_cable"],
                    config["aerodynamic_bridle"]["cf_cable"],
                    body_aero=body_aero,
                    bridle_node_pairs=bridle_node_pairs,
                )
                bridle_force_total = np.sum(f_aero_bridle, axis=0)
                bridle_force_total_norm = np.linalg.norm(bridle_force_total)
                bridle_force_nodal_max = np.max(np.linalg.norm(f_aero_bridle, axis=1))
                print(
                    "Bridle aero force: "
                    f"|sum|={bridle_force_total_norm:.3f}N "
                    f"sum=[{bridle_force_total[0]:.3f}, {bridle_force_total[1]:.3f}, {bridle_force_total[2]:.3f}]N "
                    f"max_node={bridle_force_nodal_max:.3f}N"
                )
                logging.info(
                    "Bridle aero force iter %s: |sum|=%.3fN, sum=(%.3f, %.3f, %.3f)N, max_node=%.3fN",
                    i,
                    bridle_force_total_norm,
                    bridle_force_total[0],
                    bridle_force_total[1],
                    bridle_force_total[2],
                    bridle_force_nodal_max,
                )
            else:
                f_aero_bridle = np.zeros((len(struc_nodes), 3))

            inertial_force_total = np.asarray(
                results_aero.get("inertial_force", np.zeros(3)), dtype=float
            )
            f_inertial = distribute_total_force_by_particle_mass(
                inertial_force_total,
                m_arr,
            )
            gravity_force_total = np.asarray(
                results_aero.get("gravity_force", np.zeros(3)), dtype=float
            )
            f_ext_gravity = distribute_total_force_by_particle_mass(
                gravity_force_total,
                m_arr,
            )
            inertial_force_total_norm = np.linalg.norm(inertial_force_total)
            print(
                "Inertial force (QSM): "
                f"|sum|={inertial_force_total_norm:.3f}N "
                f"sum=[{inertial_force_total[0]:.3f}, {inertial_force_total[1]:.3f}, {inertial_force_total[2]:.3f}]N"
            )
            gravity_force_total_norm = np.linalg.norm(gravity_force_total)
            print(
                "Gravity force (QSM): "
                f"|sum|={gravity_force_total_norm:.3f}N "
                f"sum=[{gravity_force_total[0]:.3f}, {gravity_force_total[1]:.3f}, {gravity_force_total[2]:.3f}]N"
            )
            logging.info(
                "Inertial force iter %s: |sum|=%.3fN, sum=(%.3f, %.3f, %.3f)N",
                i,
                inertial_force_total_norm,
                inertial_force_total[0],
                inertial_force_total[1],
                inertial_force_total[2],
            )
            logging.info(
                "Gravity force iter %s: |sum|=%.3fN, sum=(%.3f, %.3f, %.3f)N",
                i,
                gravity_force_total_norm,
                gravity_force_total[0],
                gravity_force_total[1],
                gravity_force_total[2],
            )
            f_aero = f_aero_wing + f_aero_bridle

            ## EXTERNAL FORCE
            f_ext = f_aero + f_inertial + f_ext_gravity
            f_ext = np.round(f_ext, 5)
            f_ext_flat = f_ext.flatten()
            end_time_f_ext = time.time()

            ### FORCING SYMMETRY
            if config["is_with_forcing_symmetry"]:
                logging.info("Forcing symmetry in y-direction")
                struc_nodes = forcing_symmetry(struc_nodes)

            ### RESIDUAL
            f_residual = f_int + f_ext_flat

            # Zero out residual at fixed (constrained) nodes — their imbalance
            # is carried by the constraint reaction force, not by f_int.
            # Without this, the residual includes e.g. the weight of node 0
            # (~92 N) which can never converge to zero.
            for fix_idx in config["structural_pss"]["fixed_point_indices"]:
                f_residual[3 * fix_idx : 3 * fix_idx + 3] = 0.0

            # Convergence measure: the Euclidean norm of the nodal force
            # residual normalised by the resultant tether force, i.e. the load
            # the whole structure hangs from. Dimensionless, so the same
            # threshold holds across wind speeds and depower settings.
            f_tether_resultant = resultant_tether_force(f_ext)
            f_residual_relative = relative_residual_norm(f_residual, f_tether_resultant)
            f_residual_list.append(f_residual_relative)
            logging.debug(
                f"residual force in y-direction: {np.sum([f_residual[1::3]]):.3f}N"
            )
            logging.debug(
                "residual %.4fN / tether force %.1fN = %.3e (tol %.1e)",
                np.linalg.norm(f_residual),
                f_tether_resultant,
                f_residual_relative,
                residual_tol_relative,
            )

            ### TRACKING
            # Update unified tracking dataframe (replaces position update)
            # Use i+1 so that positions[0] retains the true initial geometry
            # stored in the pre-loop call.
            tracking.update_tracking_arrays(
                tracking_data,
                i + 1,
                struc_nodes,
                f_ext_flat,
                f_residual,
            )
            tracking.update_aero_tracking(
                tracking_data,
                i + 1,
                results_aero.get("alpha_at_ac"),
                results_aero.get("stall_mask"),
            )

            ### PROGRESS BAR
            pbar.set_postfix(
                {
                    "res/Ft": f"{f_residual_relative:.2e}",
                    "res": f"{np.linalg.norm(f_residual):.3f}N",
                    "aero": f"{end_time_f_ext-begin_time_f_ext:.2f}s",
                    "struc": f"{end_time_f_int-begin_time_f_int:.2f}s",
                }
            )
            pbar.update(1)

            ### CHECK CONVERGENCE
            is_convergence, should_break, is_stagnated = check_convergence(
                iteration=i,
                residual=f_residual,
                residual_norm_history=f_residual_list,
                force_reference=f_tether_resultant,
                aero_forces_vsm_format=f_aero_wing_vsm_format,
                solver_config=config["aero_structural_solver"],
                is_run_only_1_time_step=config["is_run_only_1_time_step"],
                stagnation_check_start=stagnation_check_start,
            )

            ### ELEMENT ELONGATION BOUND
            stiffness_updated_now = False
            # Stiffen whatever exceeds the bound and keep iterating: the
            # elongation bound is part of the convergence criterion, not a
            # post-check. Once every offending element sits at max_stiffness
            # nothing is updated any more and the loop is free to converge.
            if is_with_stiffness_update:
                elongations = element_elongations(
                    struc_nodes,
                    kite_connectivity_arr,
                    psystem.extract_rest_length,
                    pulley_pairs=pulley_line_to_other_node_pair_dict,
                )
                k_target, n_stiffened, max_elongation_seen = (
                    structural_pss.adapt_stiffnesses(
                        k_target,
                        elongations,
                        element_indices=stiffness_update_indices,
                        max_elongation=elongation_bound,
                        factor=stiffness_update_factor,
                        max_stiffness=max_stiffness,
                    )
                )
                if n_stiffened > 0 and not stiffness_ramp_active:
                    structural_pss.set_stiffnesses(psystem, k_target)
                if n_wing_elements > 0:
                    max_wing_elongation = float(
                        np.nanmax(elongations[:n_wing_elements])
                    )
                    if len(elongations) > n_wing_elements:
                        max_bridle_elongation = float(
                            np.nanmax(elongations[n_wing_elements:])
                        )
                if stiffness_ramp_active:
                    # The structure has not reached its target stiffness yet.
                    stiffness_updated_now = True
                    is_convergence = False
                    is_stagnated = False
                    stagnation_check_start = i + 1
                if n_stiffened > 0:
                    stiffness_updated_now = True
                    n_stiffness_updates += n_stiffened
                    is_convergence = False
                    is_stagnated = False
                    # The residual history is no longer comparable across a
                    # stiffness change, so restart the stagnation window.
                    stagnation_check_start = i + 1
                    logging.info(
                        "Stiffened %s element(s) over the %.2f%% elongation "
                        "bound (max %.2f%%) at iteration %s",
                        n_stiffened,
                        100.0 * elongation_bound,
                        100.0 * max_elongation_seen,
                        i,
                    )

            should_apply_steering_now = (
                steering_tape_extension_step != 0
                and steering_tape_final_extension != 0
                and ((i + 1) % steering_actuation_interval_iters == 0)
            )

            delta_steering, is_steering_finalized, did_update_steering = (
                update_steering_tape_actuation_progressive(
                    psystem=psystem,
                    steering_tape_indices=steering_tape_indices,
                    steering_tape_extension_step=steering_tape_extension_step,
                    initial_length_steering_left=initial_length_steering_left,
                    initial_length_steering_right=initial_length_steering_right,
                    steering_tape_final_extension=steering_tape_final_extension,
                    should_apply_update=should_apply_steering_now,
                )
            )

            # Two-phase regularization: on stagnation in phase 1, disable
            # pseudo_dt and continue to let the solver polish to true equilibrium.
            if is_stagnated:
                logging.info("Classic PS non-converging - residual no longer changes")
                should_break = True

            if qs_state_should_break:
                # Do not allow quasi-steady stagnation stopping to interrupt
                # progressive actuation before targets are fully applied, nor a
                # stiffness update whose shape change has not been solved yet.
                if (
                    is_actuation_finalized
                    and is_steering_finalized
                    and depower_settle_counter == 0
                    and not stiffness_updated_now
                ):
                    break
                qs_state_should_break = False

            ### ACTUATION (depower & steering checked every iteration; applied at enforced cadence)
            should_apply_depower_now = (
                power_tape_extension_step != 0
                and power_tape_final_extension != 0
                and ((i + 1) % power_tape_actuation_interval_iters == 0)
            )
            (
                delta_power_tape,
                is_actuation_finalized,
                did_update_depower,
            ) = update_power_tape_actuation(
                psystem=psystem,
                power_tape_index=power_tape_index,
                power_tape_extension_step=power_tape_extension_step,
                initial_length_power_tape=initial_length_power_tape,
                power_tape_final_extension=power_tape_final_extension,
                should_apply_update=should_apply_depower_now,
                n_power_tape_steps=n_power_tape_steps,
            )

            if did_update_depower:
                depower_settle_counter = depower_settle_iterations_after_update
            elif depower_settle_counter > 0:
                depower_settle_counter -= 1

            # If either actuation is not finalized, continue actuation phase.
            if (
                (not is_actuation_finalized)
                or (not is_steering_finalized)
                or (depower_settle_counter > 0)
            ):
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
    vec_wind = body_aero.va
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
    alpha_at_ac_mid = np.ravel(results_aero["alpha_at_ac"])[mid_idx]

    print(f"alpha = {np.degrees(angle):.2f}° (va vs mid-span chord)")
    print(
        f'alpha = {np.rad2deg(alpha_at_ac_mid):.2f}° (incl. induced velocity, from results_aero["alpha_at_ac"])'
    )
    # print(
    #     f'results_aero["alpha_uncorrected"]: {float(np.rad2deg(results_aero["alpha_uncorrected"][mid_idx])):.2f}°'
    # )
    # print(
    #     f'results_aero["alpha_geometric"]: wrt horizontal {results_aero["alpha_geometric"][mid_idx]:.2f}°'
    # )

    rest_lengths = psystem.extract_rest_length

    if config["is_with_final_plot"]:
        plotting.main(
            struc_nodes,
            kite_connectivity_arr,
            f_ext=f_ext,
            f_bridle=f_aero_bridle if config["is_with_aero_bridle"] else None,
            f_inertial=f_inertial,
            rest_lengths=rest_lengths,
            struc_nodes_initial=struc_nodes_initial,
            title="Initial vs final",
            pulley_line_indices=pulley_line_indices,
            pulley_line_to_other_node_pair_dict=pulley_line_to_other_node_pair_dict,
            vel_app=vel_app,
        )

    if config.get("is_with_aero_frame_final_plot", False):
        panel_cp = np.asarray(results_aero["panel_cp_locations"])
        aerodynamic_vsm.plot_aero_forces_with_frames(
            struc_nodes=struc_nodes,
            kite_connectivity_arr=kite_connectivity_arr,
            m_arr=m_arr,
            panel_cp_locations=panel_cp,
            f_aero_panel=np.asarray(f_aero_wing_vsm_format),
            struc_node_le_indices=struc_node_le_indices,
            struc_node_te_indices=struc_node_te_indices,
            title="Aero forces, body frame and course frame",
        )
        plt.show()

    # Calculate cl, cd, tether force, and va for output
    opt_x = np.asarray(results_aero.get("opt_x", np.full(5, np.nan)), dtype=float)
    kite_speed = opt_x[0] if opt_x.size > 0 else np.nan

    # Apparent wind speed from quasi-steady kinematics (opt_x[0]).
    # Fallback to body_aero.va for backward compatibility.
    if opt_x.size > 0 and np.isfinite(opt_x[0]):
        va = float(opt_x[0])
    else:
        try:
            va_vec = np.asarray(body_aero.va, dtype=float)
            va = float(np.linalg.norm(va_vec))
        except Exception:
            va = np.nan

    # Extract Cl and Cd from VSM solution
    # Try from results_aero dict first (cl_distribution, cd_distribution)
    cl = np.nan
    cd = np.nan

    try:
        cl_dist = np.asarray(results_aero.get("cl", []), dtype=float).ravel()
        cd_dist = np.asarray(results_aero.get("cd", []), dtype=float).ravel()
        if cl_dist.size > 0 and np.isfinite(cl_dist).any():
            cl = float(np.nanmean(cl_dist))  # Wing-averaged Cl
        if cd_dist.size > 0 and np.isfinite(cd_dist).any():
            cd = float(np.nanmean(cd_dist))  # Wing-averaged Cd
    except Exception as e:
        logging.warning(
            f"Could not extract cl/cd from cl_distribution/cd_distribution: {e}"
        )

    # Fallback: try to extract from body_aero panels after solve
    if np.isnan(cl) or np.isnan(cd):
        try:
            panels = body_aero.panels
            if panels and len(panels) > 0:
                # Try to access cl/cd attributes from panels
                cl_vals = [
                    p.cl for p in panels if hasattr(p, "cl") and np.isfinite(p.cl)
                ]
                cd_vals = [
                    p.cd for p in panels if hasattr(p, "cd") and np.isfinite(p.cd)
                ]
                if cl_vals:
                    cl = float(np.mean(cl_vals))
                if cd_vals:
                    cd = float(np.mean(cd_vals))
                logging.info(f"Extracted Cl={cl:.4f}, Cd={cd:.4f} from panel objects")
        except Exception as e:
            logging.warning(f"Could not extract cl/cd from panel objects: {e}")

    # Also debug: log what's in results_aero
    logging.debug(f"results_aero keys: {list(results_aero.keys())}")

    # Tether force: sum of all forces in Z direction at equilibrium
    try:
        f_aero_total = np.sum(f_aero_wing, axis=0)  # f_aero_wing is shape (n_nodes, 3)
        tether_force = float(
            f_aero_total[2] + gravity_force_total[2] + inertial_force_total[2]
        )
    except Exception:
        tether_force = np.nan

    meta = {
        "total_time_s": time.time() - start_time,
        "n_iter": i + 2,  # +2: 1 for pre-loop initial state + (i+1) loop entries
        "converged": is_convergence,
        # What "converged" actually means for this solve: the nodal force
        # residual ||f_int + f_ext||, the resultant tether force it was
        # normalised by, their ratio, and the threshold it was tested against.
        "residual_force_n": float(np.linalg.norm(f_residual)),
        "residual_relative": float(f_residual_relative),
        "residual_tol_relative": float(residual_tol_relative),
        "tether_force_resultant": float(f_tether_resultant),
        # Largest element elongation of the converged shape: the stiffness
        # update keeps this at or below `elongation_bound` unless offending
        # elements hit `max_stiffness`. Pulley arms carry the ROPE's strain
        # (both arms combined), not their individual length change.
        "max_element_elongation": max_element_elongation(
            struc_nodes,
            kite_connectivity_arr,
            rest_lengths,
            pulley_pairs=pulley_line_to_other_node_pair_dict,
        ),
        "max_wing_element_elongation": float(max_wing_elongation),
        "max_bridle_element_elongation": float(max_bridle_elongation),
        "elongation_bound": float(elongation_bound),
        "is_with_stiffness_update": bool(is_with_stiffness_update),
        "stiffness_update_scope": str(stiffness_update_scope),
        "n_stiffness_updates": int(n_stiffness_updates),
        "stiffness_ramp_iterations": int(stiffness_ramp_iterations),
        "stiffness_ramp_start_factor": float(stiffness_ramp_start_factor),
        # Stiffnesses the converged shape actually ran with. The deformed
        # struc_geometry.yaml snapshot still carries the ORIGINAL k (the
        # update is a solver device, and a re-solve re-derives it from the
        # same rule), so this is the only record of what was used.
        "final_stiffnesses": np.array(
            [float(link.k) for link in psystem.springdampers], dtype=float
        ),
        "qs_success": bool(results_aero.get("success", False)),
        "opt_x": opt_x,
        # The gravity flag this solve actually ran with: it gates BOTH the
        # structural weight and the internal trim, so callers audit it rather
        # than trust whatever the kite's as_config.yaml said at launch time.
        "is_with_gravity": bool(config["is_with_gravity"]),
        "is_with_kcu_drag": bool(config.get("is_with_kcu_drag", True)),
        # Kite-side KCU drag the trim actually applied (0.0 when off).
        "kcu_drag_coefficient": float(
            results_aero.get("kcu_drag_coefficient", 0.0) or 0.0
        ),
        # The internal trim's own moment residuals at opt_x. In deep stall
        # least_squares can park on a residual plateau (|cmx| ~ 5e-2 observed)
        # while still reporting success -- downstream re-solves of the same
        # model converge tighter, so keep the evidence of how converged THIS
        # trim actually was.
        "qs_cm": np.asarray(results_aero.get("cm", [np.nan] * 3), dtype=float),
        "aero_roll_deg": float(results_aero.get("aero_roll_deg", np.nan)),
        "aoa_deg": float(results_aero.get("aoa_deg", np.nan)),
        # Freestream angle of attack atan2(va_z, va_x). This is the value the trim
        # feeds to va_initialize, so it is the correct centre for a frozen alpha
        # sweep around this anchor (the center-chord ``aoa_deg`` differs by the
        # induced angle). Falls back to aoa_deg if the trim did not report it.
        "aoa_course_deg": float(
            results_aero.get("aoa_course_deg", results_aero.get("aoa_deg", np.nan))
        ),
        "side_slip_deg": float(results_aero.get("side_slip_deg", np.nan)),
        # Apparent-wind magnitude used by the trim (Umag), distinct from the
        # tangential/kite speed stored in ``va``; the frozen sweep reuses it.
        "Umag": float(results_aero.get("Umag", va)),
        "va": va,
        "cl": float(cl),
        "cd": float(cd),
        "tether_force": float(tether_force),
        "rest_lengths": rest_lengths,
        # Converged circulation of the FINAL coupled trim. Downstream re-solves
        # of the snapshot seed their gamma loop with it (branch selection near
        # stall); it cannot be reconstructed later -- it is the product of the
        # warm continuation along the coupled iterations.
        "gamma_distribution": np.asarray(
            results_aero.get("gamma_distribution", []), dtype=float
        ),
        "panel_cp_locations": np.asarray(results_aero.get("panel_cp_locations", []), dtype=float),
        "f_aero_panel": np.asarray(f_aero_wing_vsm_format, dtype=float),
        # Convert kite_connectivity to a numeric array for HDF5 compatibility
        "kite_connectivity": np.array(
            [[int(row[0]), int(row[1])] for row in np.array(kite_connectivity_arr)],
            dtype=np.int32,
        ),
    }

    # Summary stall flag: which panels stalled at least once across all iterations.
    if "stall_mask" in tracking_data:
        meta["panels_ever_stalled"] = tracking_data["stall_mask"].any(axis=0)
        n_ever_stalled = int(meta["panels_ever_stalled"].sum())
        if n_ever_stalled > 0:
            logging.warning(
                "STALL summary: %d/%d panels stalled at least once during the simulation.",
                n_ever_stalled,
                tracking_data["stall_mask"].shape[1],
            )

    return tracking_data, meta


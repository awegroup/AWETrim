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
import math
import matplotlib.pyplot as plt
from . import structural_pss
from . import structural_nlp
from .. import aerodynamic_vsm, aerodynamic_bridle_line_drag, tracking
from awetrim import plotting
from .actuation import (
    update_power_tape_actuation,
    update_steering_tape_actuation_progressive,
)
from ..convergence import (
    check_convergence,
    compute_adaptive_dt,
    relative_residual_norm,
    resolve_fallback_tolerance,
    resolve_residual_tolerances,
    resultant_tether_force,
    element_elongations,
    max_element_elongation,
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
    # Inner structural solver: PSS kinetic damping (default) or the exact
    # minimum-energy NLP (structural_pss.solver: nlp). Both share the same
    # call contract and operate on the same particle-system state, so
    # actuation, the stiffness ramp and handover work identically.
    structural_solver_name, structural_solve = (
        structural_nlp.resolve_structural_solver(
            config["structural_pss"],
            psystem,
            kite_connectivity_arr,
            pulley_line_indices,
        )
    )
    print(f"--> Running structural solver: {structural_solver_name}")

    ## PRELOOP
    f_ext_gravity = np.zeros(struc_nodes.shape)

    max_iter = config["aero_structural_solver"]["max_iter"]
    # Keep index 0 for the pre-loop initial state and reserve max_iter loop slots.
    t_vector = np.linspace(0, max_iter, max_iter + 1)
    n_panels = len(body_aero.panels) if body_aero is not None else 0
    tracking_data = tracking.setup_tracking_arrays(len(struc_nodes), t_vector, n_panels=n_panels)
    is_convergence = False
    f_residual_list = []
    #: Same iterations as ``f_residual_list`` but always in newtons, so the
    #: report stays readable whichever criterion is active.
    f_residual_n_list = []
    f_tether_drag = np.zeros(3)

    ### ELEMENT ELONGATION BOUND
    # The WING element stiffnesses are not known: following Poland & Schmehl
    # they are chosen so that no element elongates by more than
    # `max_elongation`. Enforced HERE rather than as a post-check, so a
    # converged shape satisfies both the force residual and the bound.
    #
    # Scope defaults to the wing. A bridle line's stiffness is not free -- it is
    # E*A/l0 from the line material and diameter -- so stiffening it to suppress
    # its (real) stretch falsifies the bridle unless the ceiling keeps it
    # physical, which is what the MODULUS cap below is for. Off by default:
    # this changes the model, so it is opted into, never inherited.
    is_with_stiffness_update = bool(
        config["structural_pss"].get("update_stiffness", True)
    )
    stiffness_update_scope = str(
        config["structural_pss"].get("stiffness_update_scope", "all")
    ).lower()
    elongation_bound = float(config["structural_pss"].get("max_elongation", 0.01))
    stiffness_update_factor = float(
        config["structural_pss"].get("stiffness_update_factor", 1.5)
    )
    # The ceiling is a MODULUS, expressed as a multiple of the geometry's own.
    # 10.9 takes a 10 GPa base to the 109 GPa SK75 fibre datasheet value.
    max_modulus_factor = float(
        config["structural_pss"].get("max_stiffness_factor", 9.0)
    )
    # Minimum coupled iterations between two stiffening events. A stiffness
    # jump has to be ABSORBED before the next one is judged -- the residual
    # spikes on the iteration after an update, and stiffening again off that
    # spike is the open-loop ramp this design exists to avoid.
    stiffness_settle_iters = int(
        config["structural_pss"].get("stiffness_settle_iters", 5)
    )
    # Wing elements are parsed first, bridle lines appended after them.
    n_wing_elements = len(kite_connectivity_arr) - (
        0 if bridle_connectivity_arr is None else len(bridle_connectivity_arr)
    )
    if stiffness_update_scope == "all":
        stiffness_update_indices = None
    elif stiffness_update_scope == "bridle":
        stiffness_update_indices = range(n_wing_elements, len(kite_connectivity_arr))
    else:
        stiffness_update_indices = range(n_wing_elements)
    #: A stiffness step may only be taken from a state at or under this
    #: residual [N] -- the solver's own convergence tolerance by default, so
    #: "converged" means the same thing to the ramp as to the solve. Loosen it
    #: only deliberately: every multiple of tol is a step taken from a state
    #: that is not actually an equilibrium.
    # ONE resolution of the convergence measure for the whole solve, so the
    # loop, the adaptive dt, the stagnation window and the stiffness trigger
    # can never disagree about what "converged" means or what units it is in.
    residual_tol_active, _stagnation_tol_active, is_residual_relative = (
        resolve_residual_tolerances(config["aero_structural_solver"])
    )
    #: Suffix for printed residuals. The relative measure is dimensionless,
    #: so it gets no unit rather than a trailing "-".
    residual_unit = "" if is_residual_relative else " N"
    residual_tol_fallback = resolve_fallback_tolerance(
        config["aero_structural_solver"], residual_tol_active
    )
    #: Iterations without a NEW BEST residual before a solve that is already
    #: inside the fallback is declared plateaued and accepted. A limit cycle
    #: is not flat, so the stagnation window (which compares a span against
    #: `stagnation_tol_relative`) never fires on one: measured at
    #: u_dp 0.18 / u_s 0.1 the cycle spans 2.7e-4 against a 4e-5 stagnation
    #: tolerance, so the loop ran 87 iterations past its last improvement to
    #: reach max_iter. 0 disables.
    plateau_patience = int(
        config["aero_structural_solver"].get("residual_plateau_patience", 20)
    )
    is_converged_at_fallback = False
    residual_best = float("inf")
    residual_best_iteration = 0
    stiffness_trigger_residual = residual_tol_active * float(
        config["structural_pss"].get("stiffness_trigger_factor", 1.0)
    )

    # k_base is the GEOMETRY's own stiffness (E*A/l0 straight from the YAML) and
    # must be read before any handover is applied: it is the reference the
    # modulus ceiling is measured against. Seed the ceiling off a stiffened
    # state instead and it ratchets -- 9x of an already-9x run is 81x, and the
    # bound silently stops bounding anything.
    k_base = structural_pss.get_stiffnesses(psystem)
    k_ceiling = structural_pss.modulus_stiffness_ceiling(k_base, max_modulus_factor)

    # The stiffnesses the elongation bound EARNED, if a neighbour is handing
    # them over. They are per-element and cannot be written back into
    # struc_geometry.yaml (which stores one modulus per material), so without
    # this a restart re-derives every one of them: at u_dp 0.18 that is 19-29
    # stiffening events, each costing `stiffness_settle_iters` before the next
    # is allowed -- most of the solve. Clipped to this geometry's own ceiling,
    # so a handover cannot import a stiffness the bound would not have granted.
    k_target = k_base.copy()
    handover_block = config.get("solver_state_handover") or {}
    handover_k = handover_block.get("stiffnesses")

    # A per-element MODULUS FACTOR k/k_base, which is what a stiffness should
    # travel as. `stiffnesses` carries absolute N/m, and k = E*A/l0 depends on
    # the REST LENGTH, so an absolute value handed to a point whose tapes are
    # actuated differently silently changes that element's modulus. Factors
    # rebuild k against each point's OWN k_base and so describe one kite.
    #
    # This is also the only order-independent way to pin a stiffness across a
    # sweep. Chaining `stiffnesses` from row to row freezes every row at the
    # FIRST row's structure (measured: 0 stiffening events and an identical
    # sum k for every subsequent row, against 7-29 events solving cold), so
    # which structure the sweep reports depends on which point happened to be
    # solved first -- i.e. on sharding and ordering.
    handover_factors = handover_block.get("stiffness_factors")
    if handover_factors is not None:
        factors = np.asarray(handover_factors, dtype=float).ravel()
        if factors.size != k_base.size:
            logging.warning(
                "Ignoring handover stiffness factors: %s values for %s elements.",
                factors.size,
                k_base.size,
            )
        elif not np.all(np.isfinite(factors)) or np.any(factors <= 0.0):
            logging.warning(
                "Ignoring handover stiffness factors: non-finite or <= 0."
            )
        else:
            if handover_k is not None:
                logging.warning(
                    "Both stiffnesses and stiffness_factors handed over; using "
                    "the factors, which are rest-length independent."
                )
                handover_k = None
            k_target = np.clip(factors * k_base, k_base, k_ceiling)
            structural_pss.set_stiffnesses(psystem, k_target)
            logging.info(
                "Stiffness modulus pinned: %s/%s element(s) above the "
                "geometry's own (max %.2fx, ceiling %.2fx).",
                int(np.sum(k_target > k_base * 1.000001)),
                k_base.size,
                float(np.max(k_target / k_base)),
                max_modulus_factor,
            )

    if handover_k is not None:
        handover_k = np.asarray(handover_k, dtype=float).ravel()
        if handover_k.size != k_base.size:
            logging.warning(
                "Ignoring handover stiffnesses: %s values for %s elements.",
                handover_k.size,
                k_base.size,
            )
        elif not np.all(np.isfinite(handover_k)) or np.any(handover_k <= 0.0):
            logging.warning("Ignoring handover stiffnesses: non-finite or <= 0.")
        else:
            k_target = np.clip(handover_k, k_base, k_ceiling)
            structural_pss.set_stiffnesses(psystem, k_target)
            n_raised = int(np.sum(k_target > k_base * 1.000001))
            logging.info(
                "Stiffnesses handed over: %s/%s element(s) above the "
                "geometry's own (max %.2fx, ceiling %.2fx).",
                n_raised,
                k_base.size,
                float(np.max(k_target / k_base)),
                max_modulus_factor,
            )

    ### MODULUS CONTINUATION, INSIDE THE COUPLED LOOP
    # Walk the bridle modulus up to `modulus_ramp_target_factor` times the
    # geometry's own, in `modulus_ramp_step_factor` steps, each taken only once
    # the previous one has been ABSORBED (same residual gate as the elongation
    # bound below).
    #
    # Why in the loop rather than by re-solving from a converged snapshot: a
    # restart can only carry what is serialisable -- node positions, rest
    # lengths, stiffnesses, the trim, the circulation. It cannot carry the
    # Aitken relaxation state, the adapted dt (which drifts from 0.005 to
    # ~0.009 over a solve), or the phase of the loop's cumulative trim
    # rotation. Measured 2026-08-28: re-solving at the SAME modulus from a
    # converged 3.327 N state restarts at 191 N against only 288 N of
    # free-node load, i.e. the aero lands somewhere else entirely. Continuing
    # in place has no handover, so none of that can be dropped.
    modulus_ramp_target = float(
        config["structural_pss"].get("modulus_ramp_target_factor", 1.0)
    )
    modulus_ramp_step = float(
        config["structural_pss"].get("modulus_ramp_step_factor", 1.4)
    )
    modulus_ramp_scope = str(
        config["structural_pss"].get("modulus_ramp_scope", "bridle")
    ).lower()
    is_with_modulus_ramp = modulus_ramp_target > 1.0 and modulus_ramp_step > 1.0
    #: Where the ramp currently sits, as a multiple of the initial modulus.
    modulus_factor = 1.0
    modulus_ramp_history = []
    is_modulus_ramp_finalized = not is_with_modulus_ramp
    #: The last stiffness state that actually CONVERGED, and its factor. A step
    #: that cannot be absorbed is reverted to this, because the previous level
    #: is a perfectly good answer -- it is a converged aerostructural state at a
    #: known modulus. Failing the whole solve because the ramp could not reach
    #: its target throws that away and reports NaN for a case that had a valid
    #: result in hand. The stall point IS the deliverable: the highest modulus
    #: this case supports, measured rather than assumed.
    k_last_converged = None
    #: The converged SHAPE that went with those stiffnesses. Restoring k alone
    #: is useless: by the time a bad step is detected the geometry has already
    #: run away (1.9 MN residual, 460% elongation, v_tau pinned at its bound
    #: was measured at u_dp 0.32), and putting the old stiffness back onto
    #: wrecked positions recovers nothing. The state has to come back too.
    struc_nodes_last_converged = None
    modulus_factor_last_converged = 1.0
    modulus_step_current = modulus_ramp_step
    n_modulus_halvings = 0
    is_modulus_ramp_stalled = False
    #: Iterations after a step before it is declared unabsorbable.
    modulus_stall_iters = int(
        config["structural_pss"].get("modulus_stall_iters", 25)
    )
    #: Abandon a step the moment the residual exceeds this multiple of the one
    #: it was taken from. Waiting the full modulus_stall_iters lets a diverging
    #: step run to 1e6 N, and nothing is recoverable from there -- the point of
    #: the fallback is to catch it while the shape is still worth reverting to.
    modulus_divergence_factor = float(
        config["structural_pss"].get("modulus_divergence_factor", 50.0)
    )
    residual_at_last_step = float("inf")
    max_modulus_halvings = int(
        config["structural_pss"].get("modulus_max_halvings", 2)
    )
    if modulus_ramp_scope == "all":
        modulus_ramp_mask = np.ones(len(k_base), dtype=bool)
    elif modulus_ramp_scope == "wing":
        modulus_ramp_mask = np.zeros(len(k_base), dtype=bool)
        modulus_ramp_mask[:n_wing_elements] = True
    else:
        modulus_ramp_mask = np.zeros(len(k_base), dtype=bool)
        modulus_ramp_mask[n_wing_elements:] = True
        if modulus_ramp_scope == "bridle_lines":
            # The power and steering tapes are flat 12x1.5 mm webbing, not
            # braided rope -- their `d` in the geometry is an AREA-equivalent
            # diameter, so giving them a rope's modulus is not the same
            # physical claim it is for a round line. The depower tape is also
            # the actuator: at 90 GPa its k is ~1.1e6 N/m, which turns a 5 cm
            # rest-length step into a ~54 kN swing on one element, the largest
            # single perturbation anywhere in the solve. Exclude them and ramp
            # the round lines only.
            for tape_index in (
                [power_tape_index] if power_tape_index is not None else []
            ) + list(steering_tape_indices or []):
                index = int(tape_index)
                if 0 <= index < len(modulus_ramp_mask):
                    modulus_ramp_mask[index] = False
    if is_with_modulus_ramp:
        logging.info(
            "Modulus continuation: %s elements (%s) from 1.0x to %.2fx in "
            "%.2fx steps, each taken only from a converged state "
            f"(residual <= {stiffness_trigger_residual:.3e}{residual_unit}, "
            "actuation done)",
            int(modulus_ramp_mask.sum()),
            modulus_ramp_scope,
            modulus_ramp_target,
            modulus_ramp_step,
        )

    n_stiffness_updates = 0
    #: Does the CURRENT shape satisfy the elongation bound? Either nothing
    #: eligible exceeds it, or everything that does is already pinned at its
    #: modulus ceiling and cannot be stiffened further. This joins the break
    #: condition, which is what makes the bound part of the convergence
    #: criterion rather than a post-check: a small FORCE residual on a shape
    #: whose lines are stretched 2-3% is not a converged answer, it is a
    #: converged answer to the wrong problem.
    is_elongation_satisfied = not bool(
        config["structural_pss"].get("update_stiffness", True)
    )
    max_elongation_seen = float("nan")
    max_wing_elongation = float("nan")
    max_bridle_elongation = float("nan")
    #: Iteration of the last stiffening, and the residual just before it. The
    #: iteration spaces successive steps; the residual is recorded for the log
    #: and the ramp history, NOT used as the trigger (see the gate below).
    last_stiffening_iteration = -(10**6)
    residual_before_stiffening = float("inf")
    if is_with_stiffness_update:
        logging.info(
            "Elongation bound active: scope=%s, bound=%.2f%%, factor=%.2f, "
            "modulus ceiling=%.1fx initial, settle=%s iterations",
            stiffness_update_scope,
            100.0 * elongation_bound,
            stiffness_update_factor,
            max_modulus_factor,
            stiffness_settle_iters,
        )
    is_actuation_finalized = True
    is_steering_finalized = True
    struc_nodes_prev = None  # Initialize previous points for tracking
    #: Set below from the handover: with it, iteration 0 is relaxed against the
    #: shape that was handed over instead of accepting the structural step
    #: whole. A cold solve keeps None -- its first step SHOULD be unrelaxed,
    #: since it is walking in from the undeformed geometry.
    is_continuation = False
    start_time = time.time()
    plotting.set_plot_style()

    stagnation_check_start = 0  # iteration at which current phase started

    # Adaptive dt for PSS solver
    # NOT handed over from a continuation, deliberately. dt is the PSS inner
    # solve's continuation parameter and dt_initial is the FLOOR of the
    # adaptive range (compute_adaptive_dt interpolates dt_initial -> dt_max on
    # the residual). Seeding it with the dt a converged neighbour earned raises
    # that floor -- measured 0.00895 against a 0.005 config -- so the moment
    # the new point's residual goes large, the solve cannot take the small step
    # it needs. That is exactly a coupled wind step: handing 4 m/s to 5 m/s
    # raises the load 56%, and the structural solve then failed to converge in
    # 1500 inner iterations, 55 times in one point. The adaptive rule recovers
    # the right dt from the residual within one iteration anyway, so there was
    # nothing to gain and a floor to lose.
    dt_initial = config["structural_pss"]["dt"]
    #: Last value compute_adaptive_dt returned, so the solve can hand the dt it
    #: EARNED to the next one instead of making it re-derive it from scratch.
    adaptive_dt = dt_initial
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
    #: Runaway stop. A trim pinned on its kite-speed bound is a constrained
    #: optimum, not an equilibrium (aerodynamic_vsm._warn_if_trim_on_bounds):
    #: the force balance wants a speed outside the box, and a coupled loop
    #: fed such a trim only drives the shape further from anything flyable.
    #: Measured 2026-09-01 on a handover-seeded deep-depower point (u_dp
    #: 0.37, V_w 7 m/s): pinned at 40 m/s from iteration 1, residual 5.9 kN,
    #: then 100 iterations of nothing before the caller's cold retry. This
    #: many CONSECUTIVE pinned iterations end the solve as not converged
    #: (meta["stop_reason"] = "trim_speed_bound"); 0 disables.
    qs_speed_bound_patience = int(
        config["aero_structural_solver"].get("qs_speed_bound_patience", 3)
    )
    speed_bound_counter = 0
    runaway_should_break = False
    stop_reason = None
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
    # Steering needs a LONGER settle than depower. A tape half-difference
    # changes the geometry by millimetres at first; the steered equilibrium
    # (rolled wing, turning trim) is reached by the coupled fixed-point
    # iteration AMPLIFYING that asymmetry over several iterations, and the
    # residual it carries meanwhile is far below the convergence gate. With
    # no settle the u_s 0.025 / 0.05 rows of 2026-09-01 "converged" in 3
    # iterations on the still-symmetric state (roll 0.00 deg, chi_dot 0),
    # while the 0.075 row -- whose residual also dipped to 0.37 N at the same
    # point -- went on to 2.1, 17, 21 N before settling on the rolled state.
    steering_settle_iterations_after_update = int(
        config["aero_structural_solver"].get(
            "steering_settle_iterations_after_update", 6
        )
    )
    steering_settle_iterations_after_update = max(
        0, steering_settle_iterations_after_update
    )
    qs_opt_prev_rounded = None
    qs_stag_counter = 0
    qs_state_should_break = False
    depower_settle_counter = 0
    steering_settle_counter = 0
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
    logging.info(
        "Steering settle iterations after update: %s",
        steering_settle_iterations_after_update,
    )

    bridle_node_pairs = None

    if config["is_with_aero_bridle"]:
        bridle_node_pairs = (
            aerodynamic_bridle_line_drag.build_bridle_node_pairs_from_line_system(
                struc_nodes,
                getattr(body_aero, "_bridle_line_system", None),
            )
        )

    # Aitken relaxation state.
    #
    # This is SOLVER state, not model state, and it is what a snapshot restart
    # silently throws away. A cold solve walks omega down from 0.3 towards the
    # 0.05 floor as the coupling stiffens; a restart from a converged shape
    # starts again at 0.3 and takes a step several times larger than the one
    # its own base had settled on. Together with an empty residual history
    # (which resets the adaptive dt to dt_initial) and a missing previous
    # increment (which leaves Aitken unable to adapt for two more iterations),
    # that is enough to knock a converged shape off its fixed point on the
    # first iteration -- the handover then looks like an aero problem when it
    # is really the loop restarting undamped.
    #
    # ``solver_state_handover`` carries all three. It is optional and additive:
    # absent, every value below is exactly what a cold solve would use, so a
    # cold run is unchanged.
    handover = config.get("solver_state_handover") or {}
    is_continuation = bool(handover)
    #: Floor on the Aitken factor. A solve that ends pinned AT this floor is
    #: telling you the clip is what stopped it, not the physics: Aitken asked
    #: for a smaller step than it was allowed and the coupling limit-cycles
    #: instead of converging. Measured at u_dp 0.32, where the loop ends on
    #: omega = 0.0500 exactly and the residual bounces 3-170 N for twenty
    #: iterations before exiting on whichever one dips under tol.
    relaxation_min = float(
        config["aero_structural_solver"].get("relaxation_min", 0.05)
    )
    # ADAPTIVE floor (opt-in). The 0.05 floor above is what a NOISY inner
    # solve needs near the fixed point, but far from it the same floor is a
    # crawl: measured with the exact NLP inner solve, a 0.3 floor reaches the
    # 1e-4 neighbourhood 4.5x faster and THEN limit-cycles because Aitken is
    # not allowed the small step it asks for. ``relaxation_min_far`` (> min)
    # is the floor used while the residual is still above
    # ``relaxation_release_factor`` x tolerance; once the residual first drops
    # below that threshold the floor RELEASES to ``relaxation_min`` and stays
    # released (latched -- bouncing across the threshold would re-widen the
    # floor mid-damping). Default far == min keeps the historical behaviour.
    relaxation_min_far = float(
        config["aero_structural_solver"].get("relaxation_min_far", relaxation_min)
    )
    relaxation_release_factor = float(
        config["aero_structural_solver"].get("relaxation_release_factor", 30.0)
    )
    relaxation_released = relaxation_min_far <= relaxation_min
    # Outer update scheme (opt-in). "aitken" is the historical scalar-relaxed
    # fixed point. "anderson" replaces it with Anderson acceleration on the
    # NODE-POSITION fixed point: a vector extrapolation over the last
    # ``anderson_outer_depth`` iterates that can damp the oscillatory
    # (period-4) coupling mode no scalar omega can -- the same acceleration
    # idea the VSM gamma loop uses. Only sensible with an EXACT inner solve
    # (structural_pss.solver: nlp): PSS's ~1 N inner noise corrupts the
    # residual differences the extrapolation is built from. The history is
    # reset whenever the fixed-point MAP changes (tape actuation or a
    # stiffness event), and a step whose largest node move exceeds
    # ``anderson_outer_max_step_m`` falls back to the plain beta-relaxed step.
    outer_acceleration = str(
        config["aero_structural_solver"].get("outer_acceleration", "aitken")
    ).lower()
    if outer_acceleration not in ("aitken", "anderson"):
        raise ValueError(
            f"Unknown outer_acceleration {outer_acceleration!r}; "
            "expected 'aitken' or 'anderson'."
        )
    anderson_outer_depth = int(
        config["aero_structural_solver"].get("anderson_outer_depth", 4)
    )
    anderson_outer_beta = float(
        config["aero_structural_solver"].get("anderson_outer_beta", 0.5)
    )
    anderson_outer_reg = float(
        config["aero_structural_solver"].get("anderson_outer_reg", 1e-8)
    )
    anderson_outer_max_step = float(
        config["aero_structural_solver"].get("anderson_outer_max_step_m", 0.5)
    )
    anderson_xs: list = []
    anderson_rs: list = []
    anderson_map_signature = None
    omega_relaxation = float(
        handover.get(
            "relaxation_factor",
            config["aero_structural_solver"].get("relaxation_factor", 0.3),
        )
    )
    r_prev_flat = handover.get("node_increment")
    if r_prev_flat is not None:
        r_prev_flat = np.asarray(r_prev_flat, dtype=float).reshape(-1)
        if r_prev_flat.size != struc_nodes.size:
            logging.warning(
                "Ignoring handover node_increment: %s values for %s DOF.",
                r_prev_flat.size,
                struc_nodes.size,
            )
            r_prev_flat = None
    if handover:
        logging.info(
            "Solver state handed over: omega=%.4f, increment=%s.",
            omega_relaxation,
            "yes" if r_prev_flat is not None else "no",
        )

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
    #
    # ...unless a continuation hands over its neighbour's converged
    # circulation. Geometry and trim are only two thirds of the state: near
    # stall the gamma loop has more than one attracting branch, and a cold
    # pre-loop solve is free to pick a different one from the converged
    # neighbour whose SHAPE it was just handed. That is the same "warm shape,
    # cold aero" defect the trim seed fixes, one level down.
    #
    # Length must match the current panel count -- a re-panelled or re-meshed
    # geometry invalidates the seed, so it is dropped rather than reshaped.
    gamma_seed_prev = None
    initial_gamma = config.get("initial_gamma_distribution")
    if initial_gamma is not None:
        initial_gamma = np.asarray(initial_gamma, dtype=float).reshape(-1)
        if body_aero is None:
            logging.warning("initial_gamma_distribution given but no body_aero.")
        elif initial_gamma.size != len(body_aero.panels):
            logging.warning(
                "Ignoring initial_gamma_distribution: %s values for %s panels.",
                initial_gamma.size,
                len(body_aero.panels),
            )
        elif not np.all(np.isfinite(initial_gamma)):
            logging.warning("Ignoring initial_gamma_distribution: non-finite.")
        else:
            gamma_seed_prev = initial_gamma
            logging.info(
                "Circulation seeded from a continuation neighbour "
                "(%s panels, |gamma|_max %.4f).",
                initial_gamma.size,
                float(np.max(np.abs(initial_gamma))),
            )
    # Trim seed for the PRE-LOOP solve. Every later iteration seeds from the
    # previous one's opt_x, but this first one falls back to DEFAULT_GUESS_QS
    # (30 m/s, level) unless told otherwise. In a CONTINUATION -- stepping the
    # stiffness or the depower from an already-converged neighbour -- handing
    # over the deformed shape with a cold trim is worse than useless: the first
    # coupled step re-trims from 30 m/s and throws the warm shape away. This
    # key carries the neighbour's converged trim across with the geometry.
    # [kite_speed, roll, pitch, yaw, course_rate_body], as opt_x.
    quasi_steady_initial_guess = config.get("quasi_steady_initial_guess")
    if quasi_steady_initial_guess is not None:
        quasi_steady_initial_guess = np.asarray(
            quasi_steady_initial_guess, dtype=float
        ).reshape(5)
        logging.info(
            "Quasi-steady trim seeded from a continuation neighbour: %s",
            quasi_steady_initial_guess,
        )
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
        current_guess=quasi_steady_initial_guess,
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

    if is_continuation:
        # Iteration 0 is otherwise the ONE unrelaxed step in the whole solve:
        # `struc_nodes_prev` is None, so whatever the structural step returns is
        # accepted whole. Walking in from the undeformed geometry that is right
        # -- there is nothing to damp towards. Continuing from a neighbour's
        # converged shape it is exactly wrong: the step lands before Aitken has
        # any say, and the shape that was handed over is gone by iteration 1.
        # Anchoring on the handover shape makes iteration 0 relax like the rest.
        struc_nodes_prev = struc_nodes.copy()

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
                    residual_tol_active,
                )
                config["structural_pss"]["dt"] = adaptive_dt
                logging.debug(
                    f"Adaptive dt updated: {adaptive_dt:.6f} "
                    f"(residual: {f_residual_list[-1]:.3e}{residual_unit})"
                )
                print(f"Adaptive dt: {adaptive_dt:.6f} s at iteration {i}")
            psystem, is_structural_converged, struc_nodes, f_int = (
                structural_solve(
                    psystem,
                    f_ext_flat,
                    config["structural_pss"],
                )
            )
            end_time_f_int = time.time()

            ### Outer update: Aitken relaxation (default) or Anderson
            if struc_nodes_prev is not None and outer_acceleration == "anderson":
                r_k = struc_nodes - struc_nodes_prev
                r_k_flat = r_k.flatten()
                x_k = struc_nodes_prev.flatten()

                # The stored iterates describe ONE fixed-point map; actuation
                # or a stiffness event redefines it, so extrapolating across
                # the change would mix two maps.
                signature = (
                    np.asarray(psystem.extract_rest_length, dtype=float).tobytes()
                    + structural_pss.get_stiffnesses(psystem).tobytes()
                )
                if signature != anderson_map_signature:
                    anderson_xs.clear()
                    anderson_rs.clear()
                    anderson_map_signature = signature

                anderson_xs.append(x_k.copy())
                anderson_rs.append(r_k_flat.copy())
                if len(anderson_xs) > anderson_outer_depth + 1:
                    anderson_xs.pop(0)
                    anderson_rs.pop(0)

                beta = anderson_outer_beta
                x_next = x_k + beta * r_k_flat
                if len(anderson_xs) >= 2:
                    dX = np.diff(np.column_stack(anderson_xs), axis=1)
                    dR = np.diff(np.column_stack(anderson_rs), axis=1)
                    n_cols = dR.shape[1]
                    gram = dR.T @ dR
                    ridge = anderson_outer_reg * max(
                        float(np.trace(gram)) / max(n_cols, 1), 1e-30
                    )
                    try:
                        gamma = np.linalg.solve(
                            gram + ridge * np.eye(n_cols), dR.T @ r_k_flat
                        )
                        x_next = (
                            x_k + beta * r_k_flat - (dX + beta * dR) @ gamma
                        )
                    except np.linalg.LinAlgError:
                        pass  # keep the plain relaxed step

                max_move = float(
                    np.max(
                        np.linalg.norm((x_next - x_k).reshape(-1, 3), axis=1)
                    )
                )
                if not np.isfinite(max_move) or max_move > anderson_outer_max_step:
                    logging.info(
                        "Anderson outer step rejected (max node move %.3f m); "
                        "plain beta step, history reset.",
                        max_move,
                    )
                    x_next = x_k + beta * r_k_flat
                    anderson_xs[:] = anderson_xs[-1:]
                    anderson_rs[:] = anderson_rs[-1:]

                struc_nodes = x_next.reshape(-1, 3)
                r_prev_flat = r_k_flat.copy()

                # Sync accepted positions back to structural solver state.
                for idx, particle in enumerate(psystem.particles):
                    particle.update_pos(struc_nodes[idx])
                    particle.update_vel(np.zeros(3))

            ### Aitken relaxation of structural nodes
            elif struc_nodes_prev is not None:
                r_k = struc_nodes - struc_nodes_prev
                r_k_flat = r_k.flatten()

                if (
                    config["aero_structural_solver"].get(
                        "is_with_aitken_relaxation", True
                    )
                    and r_prev_flat is not None
                ):
                    if not relaxation_released and f_residual_list:
                        if (
                            f_residual_list[-1]
                            <= relaxation_release_factor * residual_tol_active
                        ):
                            relaxation_released = True
                            logging.info(
                                "Aitken floor released: residual %.3e <= %g x tol, "
                                "floor %.3g -> %.3g",
                                f_residual_list[-1],
                                relaxation_release_factor,
                                relaxation_min_far,
                                relaxation_min,
                            )
                    relaxation_min_active = (
                        relaxation_min if relaxation_released else relaxation_min_far
                    )
                    delta_r = r_k_flat - r_prev_flat
                    denom = np.dot(delta_r, delta_r)
                    if denom > 1e-30:
                        omega_relaxation = -omega_relaxation * (
                            np.dot(r_prev_flat, delta_r) / denom
                        )
                        omega_relaxation = np.clip(
                            omega_relaxation, relaxation_min_active, 1.0
                        )

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

            # Runaway stop (see qs_speed_bound_patience above).
            if qs_speed_bound_patience > 0:
                if "kite_speed" in (results_aero.get("trim_on_bounds") or []):
                    speed_bound_counter += 1
                else:
                    speed_bound_counter = 0
                if speed_bound_counter >= qs_speed_bound_patience:
                    runaway_should_break = True
                    stop_reason = "trim_speed_bound"
                    logging.warning(
                        "Stopping: quasi-steady trim pinned on its kite-speed "
                        "bound (%.6g m/s) for %d consecutive coupled "
                        "iterations -- no equilibrium inside the trim box; "
                        "the solve is NOT converged.",
                        float(results_aero["opt_x"][0]),
                        speed_bound_counter,
                    )

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

            # The history feeds the convergence test, the stagnation window,
            # the adaptive dt AND the stiffness trigger, so it must be in the
            # same measure as the tolerance those compare against.
            f_residual_norm_n = float(np.linalg.norm(f_residual))
            force_reference = resultant_tether_force(f_ext)
            f_residual_relative = relative_residual_norm(
                f_residual, force_reference
            )
            f_residual_list.append(
                f_residual_relative if is_residual_relative else f_residual_norm_n
            )
            f_residual_n_list.append(f_residual_norm_n)
            logging.debug(
                f"residual force in y-direction: {np.sum([f_residual[1::3]]):.3f}N"
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
                    "res": (
                        f"{f_residual_norm_n:.3f}N"
                        f" ({f_residual_relative:.2e})"
                    ),
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
                aero_forces_vsm_format=f_aero_wing_vsm_format,
                solver_config=config["aero_structural_solver"],
                is_run_only_1_time_step=config["is_run_only_1_time_step"],
                stagnation_check_start=stagnation_check_start,
                force_reference=force_reference,
            )

            ### ELEMENT ELONGATION BOUND
            # Stiffen whatever exceeds the bound and keep iterating: the bound
            # is part of the convergence criterion, not a post-check. Once
            # every offending element sits at its modulus ceiling nothing is
            # updated any more and the loop is free to converge -- and the
            # report then says honestly which elements are pinned.
            stiffness_updated_now = False
            # ONE shared gate for both stiffness mechanisms, so at most one
            # perturbation is injected per iteration and each is judged from a
            # state that has actually recovered.
            #
            # The rule is the plan's: only raise the stiffness FROM A CONVERGED
            # STATE. That means the residual is at or under the solver's own
            # tolerance -- not merely "no worse than before the last step",
            # which is vacuous on the first step (nothing precedes it) and
            # admits steps taken at 6.4 N against a 5 N criterion.
            #
            # It also means waiting for the tape. Ramping the modulus while the
            # power tape is still being walked out runs two continuations at
            # once and neither result means anything.
            residual_now = f_residual_list[-1] if f_residual_list else float("inf")
            is_settled = (
                is_actuation_finalized
                and is_steering_finalized
                and depower_settle_counter == 0
                    and steering_settle_counter == 0
                and i - last_stiffening_iteration >= stiffness_settle_iters
                and np.isfinite(residual_now)
                and residual_now <= stiffness_trigger_residual
            )

            ### MODULUS CONTINUATION
            if is_with_modulus_ramp and not is_modulus_ramp_finalized and is_settled:
                # This state converged, so it becomes the fallback -- both
                # the stiffnesses AND the shape they were converged with.
                k_last_converged = k_target.copy()
                struc_nodes_last_converged = np.array(struc_nodes, copy=True)
                modulus_factor_last_converged = modulus_factor
                residual_at_last_step = residual_now
                previous_factor = modulus_factor
                modulus_factor = min(
                    modulus_factor * modulus_step_current, modulus_ramp_target
                )
                # A FLOOR, not an assignment. The elongation bound may already
                # have stiffened some of these elements past the ramp's current
                # level; recomputing from k_base would silently undo that work
                # every step, so the two mechanisms would fight. Taking the
                # maximum lets the ramp lift everything toward the target
                # modulus while whatever the bound raised further stays raised.
                k_target = np.where(
                    modulus_ramp_mask,
                    np.maximum(k_target, k_base * modulus_factor),
                    k_target,
                )
                # The elongation bound may not push an element past the modulus
                # ceiling, but the ramp itself defines a floor that rises with
                # it -- keep the ceiling at least at the ramp's own level.
                k_ceiling = np.maximum(k_ceiling, k_base * modulus_factor)
                structural_pss.set_stiffnesses(psystem, k_target)
                stiffness_updated_now = True
                is_convergence = False
                is_stagnated = False
                stagnation_check_start = i + 1
                last_stiffening_iteration = i
                residual_before_stiffening = residual_now
                # Plain numbers, not dicts: this goes into the h5 as an
                # attribute, and a list of dicts becomes an object-dtype array
                # that h5py cannot store -- which threw away a completed
                # 337 s solve at the very last step.
                modulus_ramp_history.append(
                    (int(i), float(modulus_factor), float(residual_now))
                )
                if modulus_factor >= modulus_ramp_target - 1e-12:
                    is_modulus_ramp_finalized = True
                logging.info(
                    "Modulus continuation: %.3fx -> %.3fx at iteration %s "
                    "(residual %.3f N)%s",
                    previous_factor,
                    modulus_factor,
                    i,
                    residual_now,
                    " -- ramp complete" if is_modulus_ramp_finalized else "",
                )

            ### MODULUS RAMP STALL / FALLBACK
            # A step the solve cannot absorb within modulus_stall_iters is
            # reverted. Halve the step and try again from the same converged
            # state; after max_modulus_halvings, stop and let the solve settle
            # at the last converged modulus, which is then reported as the
            # highest this case supports.
            is_diverging = k_last_converged is not None and (
                not np.isfinite(residual_now)
                or residual_now
                > modulus_divergence_factor * max(residual_at_last_step, 1e-9)
            )
            if (
                is_with_modulus_ramp
                and not is_modulus_ramp_finalized
                and k_last_converged is not None
                and i > last_stiffening_iteration
                and (
                    is_diverging
                    or (
                        i - last_stiffening_iteration >= modulus_stall_iters
                        and residual_now > stiffness_trigger_residual
                    )
                )
            ):
                # Restore the whole state, not just the stiffnesses.
                structural_pss.set_stiffnesses(psystem, k_last_converged)
                k_target = k_last_converged.copy()
                struc_nodes = np.array(struc_nodes_last_converged, copy=True)
                for particle_index, particle in enumerate(psystem.particles):
                    particle.update_pos(struc_nodes[particle_index])
                    particle.update_vel(np.zeros(3))
                modulus_factor = modulus_factor_last_converged
                n_modulus_halvings += 1
                if n_modulus_halvings > max_modulus_halvings:
                    is_modulus_ramp_finalized = True
                    is_modulus_ramp_stalled = True
                    logging.warning(
                        "Modulus continuation STALLED at %.3fx (%s halvings "
                        "exhausted). Reverting to the last converged stiffness "
                        "and settling there; that factor is the highest this "
                        "case supports.",
                        modulus_factor,
                        n_modulus_halvings - 1,
                    )
                else:
                    modulus_step_current = math.sqrt(modulus_step_current)
                    logging.info(
                        "Step to %.3fx abandoned after %s iterations "
                        "(residual %.3g N); shape and stiffness reverted to "
                        "%.3fx, step now %.4fx",
                        previous_factor * modulus_step_current**2,
                        i - last_stiffening_iteration,
                        residual_now,
                        modulus_factor,
                        modulus_step_current,
                    )
                last_stiffening_iteration = i
                stagnation_check_start = i + 1
                is_convergence = False
                is_stagnated = False
                stiffness_updated_now = True

            ### ELEMENT ELONGATION BOUND
            if is_with_stiffness_update:
                elongations = element_elongations(
                    struc_nodes,
                    kite_connectivity_arr,
                    psystem.extract_rest_length,
                    pulley_pairs=pulley_line_to_other_node_pair_dict,
                )
                if n_wing_elements > 0 and elongations.size:
                    max_wing_elongation = float(
                        np.nanmax(elongations[:n_wing_elements])
                    )
                    if elongations.size > n_wing_elements:
                        max_bridle_elongation = float(
                            np.nanmax(elongations[n_wing_elements:])
                        )
                # Evaluated every iteration, independently of the settle
                # gate -- the gate decides WHEN to act, never whether the
                # answer is acceptable. Doing this only on settled iterations
                # is how a fast solve exits having never once looked.
                eligible = (
                    np.arange(len(elongations))
                    if stiffness_update_indices is None
                    else np.fromiter(stiffness_update_indices, dtype=int)
                )
                eligible = eligible[eligible < len(elongations)]
                over = eligible[
                    np.nan_to_num(elongations[eligible], nan=-np.inf)
                    > elongation_bound
                ]
                # An element pinned at its ceiling cannot be helped; requiring
                # it to come under the bound would spin the loop forever.
                is_elongation_satisfied = bool(
                    over.size == 0 or np.all(k_target[over] >= k_ceiling[over] - 1e-9)
                )
                if is_settled and not stiffness_updated_now:
                    k_target, n_stiffened, max_elongation_seen, n_pinned = (
                        structural_pss.adapt_stiffnesses(
                            k_target,
                            elongations,
                            element_indices=stiffness_update_indices,
                            max_elongation=elongation_bound,
                            factor=stiffness_update_factor,
                            max_stiffness=k_ceiling,
                        )
                    )
                    if n_stiffened > 0:
                        structural_pss.set_stiffnesses(psystem, k_target)
                        stiffness_updated_now = True
                        n_stiffness_updates += n_stiffened
                        is_convergence = False
                        is_stagnated = False
                        # Neither the residual history nor the stagnation
                        # window is comparable across a stiffness change.
                        stagnation_check_start = i + 1
                        last_stiffening_iteration = i
                        residual_before_stiffening = residual_now
                        logging.info(
                            "Stiffened %s element(s) over the %.2f%% bound "
                            "(max %.2f%%, %s already at ceiling) at iteration %s",
                            n_stiffened,
                            100.0 * elongation_bound,
                            100.0 * max_elongation_seen,
                            n_pinned,
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

            if runaway_should_break:
                # Unlike the stagnation stop this does not wait for the
                # actuation to finish: nothing downstream of a bound-pinned
                # trim is worth another iteration.
                is_convergence = False
                break

            if qs_state_should_break:
                # Do not allow quasi-steady stagnation stopping to interrupt
                # progressive actuation before targets are fully applied.
                if (
                    is_actuation_finalized
                    and is_steering_finalized
                    and depower_settle_counter == 0
                    and steering_settle_counter == 0
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
            if did_update_steering:
                steering_settle_counter = steering_settle_iterations_after_update
            elif steering_settle_counter > 0:
                steering_settle_counter -= 1

            # If either actuation is not finalized, continue actuation phase.
            if (
                (not is_actuation_finalized)
                or (not is_steering_finalized)
                or (depower_settle_counter > 0)
                or (steering_settle_counter > 0)
            ):
                continue

            ### PLATEAU DETECTION
            # Accept a solve that has stopped improving and is already inside
            # the fallback, rather than spending the rest of max_iter proving
            # the cycle is a cycle. Everything the post-loop acceptance
            # requires is required here too, so this only changes WHEN such a
            # point is accepted, never WHETHER.
            if plateau_patience > 0 and not is_convergence and f_residual_list:
                residual_now_plateau = float(f_residual_list[-1])
                if stiffness_updated_now:
                    # The structure just changed, so the old best belongs to a
                    # different problem and the patience clock restarts.
                    residual_best = float("inf")
                    residual_best_iteration = i
                if (
                    np.isfinite(residual_now_plateau)
                    and residual_now_plateau < residual_best
                ):
                    residual_best = residual_now_plateau
                    residual_best_iteration = i
                elif (
                    is_modulus_ramp_finalized
                    and is_elongation_satisfied
                    and residual_tol_fallback > residual_tol_active
                    and np.isfinite(residual_now_plateau)
                    and residual_now_plateau <= residual_tol_fallback
                    and i - residual_best_iteration >= plateau_patience
                ):
                    is_convergence = True
                    is_converged_at_fallback = True
                    logging.warning(
                        "Coupled solve PLATEAUED at %.3e%s after %s iterations "
                        "with no improvement on %.3e%s (best at iteration %s). "
                        "Above the %.3e%s criterion but within the %.3e%s "
                        "fallback, and every physical gate is satisfied, so it "
                        "is accepted here rather than at max_iter.",
                        residual_now_plateau,
                        residual_unit,
                        i - residual_best_iteration,
                        residual_best,
                        residual_unit,
                        residual_best_iteration,
                        residual_tol_active,
                        residual_unit,
                        residual_tol_fallback,
                        residual_unit,
                    )

            # Check if we should exit the loop
            # A solve is not converged while the modulus is still being walked
            # up: the residual being small at 1.0x says nothing about 10.9x.
            if should_break or (
                is_convergence
                and is_actuation_finalized
                and is_steering_finalized
                and is_modulus_ramp_finalized
                and is_elongation_satisfied
            ):
                break
    ######################################################################
    ## END OF SIMULATION FOR LOOP
    ######################################################################

    ### PLATEAU ACCEPTANCE
    # A solve that ran out of iterations or stagnated has not necessarily
    # failed: steered, near-stall and heavily loaded cases settle into a LIMIT
    # CYCLE rather than a fixed point, with the Aitken factor already pinned at
    # its floor, so neither more iterations nor more damping can close them.
    # Accept those at the looser `residual_tol_relative_fallback` -- but only
    # once every PHYSICAL gate is satisfied, so this can never rescue a run
    # that stopped mid-actuation or with elements still over the elongation
    # bound. `residual_cycle_span` reports the orbit the point sits on, which
    # is the honest uncertainty to quote for it.
    residual_cycle_span = float("nan")
    if len(f_residual_list) >= 2:
        window = np.asarray(f_residual_list[-10:], dtype=float)
        window = window[np.isfinite(window)]
        if window.size:
            residual_cycle_span = float(window.max() - window.min())
    if not is_convergence and f_residual_list and stop_reason is None:
        residual_final = float(f_residual_list[-1])
        physical_gates_met = (
            is_actuation_finalized
            and is_steering_finalized
            and is_modulus_ramp_finalized
            and is_elongation_satisfied
        )
        if (
            physical_gates_met
            and np.isfinite(residual_final)
            and residual_tol_fallback > residual_tol_active
            and residual_final <= residual_tol_fallback
        ):
            is_convergence = True
            is_converged_at_fallback = True
            logging.warning(
                "Coupled solve PLATEAUED at %.3e%s, above the %.3e%s "
                "criterion but within the %.3e%s fallback; accepted, cycle "
                "span over the last 10 iterations %.3e. Every physical gate "
                "(actuation, steering, modulus ramp, elongation bound) is "
                "satisfied.",
                residual_final,
                residual_unit,
                residual_tol_active,
                residual_unit,
                residual_tol_fallback,
                residual_unit,
                residual_cycle_span,
            )

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
        # Why the loop ended early, when it did for a reason other than
        # convergence / stagnation / max_iter: "trim_speed_bound" (runaway
        # stop, see qs_speed_bound_patience). None otherwise.
        "stop_reason": stop_reason,
        # The nodal force residual the run ENDED on, and the threshold it was
        # tested against. Reporting only; the criterion itself is unchanged
        # (check_convergence still owns it). A continuation needs this per
        # level to tell a solve that closed from one that merely ran out of
        # iterations near the bound. `residual_force_n` stays in newtons
        # whichever criterion is active; `residual_relative` is the
        # dimensionless measure, and `residual_tol_active` is in whichever
        # unit `residual_tol_is_relative` says.
        "residual_force_n": (
            float(f_residual_n_list[-1]) if f_residual_n_list else float("nan")
        ),
        "residual_relative": (
            float(f_residual_relative) if f_residual_n_list else float("nan")
        ),
        # Per-element k/k_base of the converged structure. This, not the
        # absolute stiffnesses, is what another solve should be pinned to:
        # it is independent of that point's tape rest lengths.
        "final_stiffness_factors": np.asarray(
            k_target / np.where(k_base > 0.0, k_base, np.nan), dtype=float
        ),
        "residual_tol_active": float(residual_tol_active),
        # True when the solve was accepted on the LOOSER fallback because it
        # plateaued on a limit cycle rather than reaching a fixed point.
        # `residual_cycle_span` is that orbit's width in the active measure.
        "converged_at_fallback": bool(is_converged_at_fallback),
        "residual_tol_fallback": float(residual_tol_fallback),
        "residual_cycle_span": float(residual_cycle_span),
        "residual_best": float(residual_best),
        "residual_best_iteration": int(residual_best_iteration),
        "plateau_patience": int(plateau_patience),
        "residual_tol_is_relative": bool(is_residual_relative),
        # Largest element elongation of the converged shape. With the bound
        # active this sits at or below `elongation_bound` unless offending
        # elements hit their modulus ceiling. Pulley arms carry the ROPE's
        # strain (both arms combined), not their individual length change.
        "max_element_elongation": max_element_elongation(
            struc_nodes,
            kite_connectivity_arr,
            rest_lengths,
            pulley_pairs=pulley_line_to_other_node_pair_dict,
        ),
        "max_wing_element_elongation": float(max_wing_elongation),
        "max_bridle_element_elongation": float(max_bridle_elongation),
        "elongation_bound": float(elongation_bound),
        "elongation_bound_satisfied": bool(is_elongation_satisfied),
        "is_with_stiffness_update": bool(is_with_stiffness_update),
        "stiffness_update_scope": str(stiffness_update_scope),
        "max_stiffness_factor": float(max_modulus_factor),
        "n_stiffness_updates": int(n_stiffness_updates),
        # Where the in-loop modulus continuation got to, and how it got there.
        # `modulus_factor_reached` is a MULTIPLE of the geometry's own modulus,
        # so 10.9 from a 10 GPa base means the solve closed at 109 GPa.
        "modulus_factor_reached": float(modulus_factor),
        "modulus_ramp_target_factor": float(modulus_ramp_target),
        "modulus_ramp_finalized": bool(is_modulus_ramp_finalized),
        # True when the ramp could not reach its target and fell back. The
        # solve is still VALID -- it converged at modulus_factor_reached.
        "modulus_ramp_stalled": bool(is_modulus_ramp_stalled),
        "modulus_ramp_halvings": int(n_modulus_halvings),
        "stiffness_trigger_residual": float(stiffness_trigger_residual),
        # (iteration, factor, residual_before_n) per step, as a float array.
        "modulus_ramp_history": np.asarray(
            modulus_ramp_history, dtype=float
        ).reshape(-1, 3),
        # Stiffnesses the converged shape actually ran with -- the deformed
        # geometry snapshot still carries the material table's k, so this is
        # the only record of what the solve used.
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
        # --- solver state, for a continuation to pick up where this left off --
        # None of this is model state, and none of it can be reconstructed from
        # the snapshot: it is what the coupled loop LEARNED about its own
        # conditioning. Handing it over is the difference between a restart
        # resuming and a restart re-learning (and taking a 0.3 step at a shape
        # its base had settled to 0.05 on). Read back as
        # ``config["solver_state_handover"]``; see the block where that is
        # parsed for what each one does.
        "relaxation_factor_final": float(omega_relaxation),
        "relaxation_min": float(relaxation_min),
        #: True when the solve ended clipped. Read it before trusting a shape:
        #: a pinned run did not converge, it ran out of allowed damping.
        "relaxation_pinned": bool(omega_relaxation <= relaxation_min * 1.000001),
        "dt_final": float(adaptive_dt),
        # Last accepted node increment, flat. Aitken needs a previous residual
        # to adapt at all -- without it a restart runs two iterations blind.
        "node_increment_final": (
            np.zeros(0) if r_prev_flat is None
            else np.asarray(r_prev_flat, dtype=float)
        ),
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


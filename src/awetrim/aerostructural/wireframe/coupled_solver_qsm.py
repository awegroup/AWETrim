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
from . import structural_wireframe
from . import structural_wireframe
from .. import aerodynamic_vsm, aerodynamic_bridle_line_drag, tracking
from awetrim import plotting
from awetrim.aerostructural.case import air_density_at_case
from .actuation import (
    steering_half_differences,
    update_power_tape_actuation,
    update_steering_tape_actuation,
    update_steering_tape_actuation_progressive,
)
from ..convergence import (
    remaining_drift,
    check_convergence,
    force_settle_state,
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
    StrutLeverAeroToStructuralLoadMapper,
    check_moment_preservation,
)
from ..protocols import AeroToStructureMap
from ..utils import (
    calculate_cg,
    rotate_geometry,
    rotation_matrix_from_angles,
)


STRUCTURAL_TO_AERO_MAPPER = LinearStructuralToAeroMapper()
AERO_TO_STRUCTURAL_LOAD_MAPPER = BilinearAeroToStructuralLoadMapper()
STRUT_LEVER_LOAD_MAPPER = StrutLeverAeroToStructuralLoadMapper()


def _map_structural_edges_to_aero(
    struc_nodes,
    struc_node_le_indices,
    struc_node_te_indices,
    n_aero_panels_per_struc_section,
    mapper=None,
):
    """Return aerodynamic leading/trailing-edge arrays from structural nodes.

    ``mapper`` replaces the default node-to-edge mapping, e.g. a
    ``SkinStructuralToAeroMapper`` that flies a finer aerodynamic mesh than
    the structure has ribs.
    """
    update = (mapper or STRUCTURAL_TO_AERO_MAPPER).map(
        struc_nodes,
        struc_node_le_indices,
        struc_node_te_indices,
        n_aero_panels_per_struc_section,
    )
    return update.leading_edge_points, update.trailing_edge_points


def _bridle_node_pairs(bridle_line_specs, struc_nodes, body_aero):
    """Which structural nodes each VSM bridle segment hangs between.

    ``bridle_line_specs`` already IS the answer: it carries the node indices the
    reader parsed, which is why ``rebuild_bridle_line_system`` can move the
    segments with the structure in the first place. Recovering them instead by
    matching the VSM's BUILT coordinates to the nearest node only holds while
    the structure is still at its built shape. On a converged one it does not:
    with 0.571 m of node movement on the LEI-V3 PSM geometry, 2 of 45 segments
    snap BOTH ends onto the same node -- zero length, NaN force, NaN loads --
    and 8 of 45 pair differently. ``wireframe.driver.solve_deformation``'s
    ``starting_from_sim_subdir`` continuation starts exactly there.

    The geometric match stays as the fallback for a caller that passes no specs.
    """
    if bridle_line_specs:
        return np.asarray(
            [[int(i), int(j)] for i, j, _diameter in bridle_line_specs], dtype=int
        )
    return aerodynamic_bridle_line_drag.build_bridle_node_pairs_from_line_system(
        struc_nodes, getattr(body_aero, "_bridle_line_system", None)
    )


def load_transfer_of(config) -> str:
    """``aerodynamic.load_transfer``: ``bilinear`` (default) or ``strut_lever``."""
    name = str((config.get("aerodynamic") or {}).get("load_transfer", "bilinear")).lower()
    if name not in ("bilinear", "strut_lever"):
        raise ValueError(f"aerodynamic.load_transfer must be bilinear or strut_lever, got {name!r}")
    return name


def _map_aero_loads_to_structure(
    f_aero_wing_vsm_format,
    struc_nodes,
    results_aero,
    aero2struc_mapping,
    config,
    check=False,
):
    """Return nodal aerodynamic loads from panel loads and a corner map.

    ``bilinear``: each panel force at VSM's per-panel centre of pressure,
    bilinear weights onto its corner nodes (moment not conserved).
    ``strut_lever``: the same force with its residual moment
    ``M_i - (cp_i - ref) x F_i`` (what VSM's clamped centre of pressure leaves
    out; ``ref`` = ``aerodynamic.reference_point``, about which VSM gives
    ``M_distribution``), rigid-body onto the bay's struts -- force and moment
    exact. ``check`` logs :func:`check_moment_preservation`.
    """
    mapping = AeroToStructureMap(panel_corner_map=np.asarray(aero2struc_mapping))
    forces = np.asarray(f_aero_wing_vsm_format, dtype=float)
    cp = np.asarray(results_aero["panel_cp_locations"], dtype=float)
    moments = None
    if load_transfer_of(config) == "strut_lever":
        if results_aero.get("M_distribution") is None:
            raise ValueError(
                "load_transfer strut_lever needs the trim's per-panel moments (M_distribution)"
            )
        ref = np.asarray(config["aerodynamic"].get("reference_point", [0.0, 0.0, 0.0]), dtype=float)
        moments = np.asarray(results_aero["M_distribution"], dtype=float) - np.cross(cp - ref, forces)
        nodal = STRUT_LEVER_LOAD_MAPPER.map_loads(forces, cp, struc_nodes, mapping, panel_moments=moments)
    else:
        nodal = AERO_TO_STRUCTURAL_LOAD_MAPPER.map_loads(forces, cp, struc_nodes, mapping)
    if check:
        check_moment_preservation(
            panel_forces=forces,
            panel_points=cp,
            nodal_forces=nodal,
            nodes=struc_nodes,
            panel_moments=moments,
        )
    return nodal


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


def _trim_results_summary(results_aero):
    """The final trim's storable outputs; one implementation for both backends."""
    from ..coupled.coupled_solver import _trim_results_summary as summary

    return summary(results_aero)


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
    # Structure -> aero edge mapper; None = the rib LE/TE nodes ARE the
    # aerodynamic edges (the PSM convention). A SkinStructuralToAeroMapper
    # flies a finer aerodynamic mesh on the same ribs.
    structural_to_aero=None,
    ### AERO --> STRUC
    aero2struc_mapping=None,
    power_tape_index=None,
    ### STRUC
    psystem=None,
    final_state=None,
):
    """
    Runs the aero-structural solver for the given input parameters.

    Args:
        config (dict): Main configuration dictionary.
        PROJECT_DIR (Path): Path to the project directory.
        results_dir (Path): Path to the results directory.
        final_state (dict or None): When given, filled with the FINAL trim's
            ``body_aero`` (trimmed attitude, trim inflow set) and
            ``results_aero``, as in ``coupled.coupled_solver.main``.

    Returns:
        tracking_data (dict): Dictionary containing time histories of positions, forces, etc.
        meta (dict): Dictionary with meta information about the simulation (timing, convergence, etc).
    """
    # The inner structural solve is Billow's line system, minimising the total
    # potential energy. It used to be a choice between PSS kinetic damping and
    # an equivalent NLP; the relaxation is gone and the NLP is now Billow's, so
    # there is nothing left to dispatch over.
    structural_solve = structural_wireframe.run_wireframe
    print("--> Running structural solver: billow wireframe (minimum energy)")

    # DIRECT steering preset (opt-in, ``steering_tape_preset_extension``):
    # set the asymmetric tape rest lengths (left = initial - delta, right =
    # initial + delta) BEFORE the loop. The geometry file format carries one
    # symmetric row per tape, so a snapshot of a STEERED state cannot store
    # its own actuation -- without this every restart silently re-walks the
    # steering from zero through the progressive ramp, which re-selects the
    # solution family instead of resuming the state. With the preset equal
    # to ``steering_tape_final_extension`` the progressive update sees
    # current == target at iteration 0 and no walk happens; with a different
    # final target only the difference is walked. The ramp remains the right
    # tool for COLD solves (it doubles as the load continuation) -- preset
    # only from a deformed snapshot equilibrated at these lengths. Snapshots
    # written since 2026-10-27 DO store their actuation (per-connection
    # bridle_connection_rest_lengths): starting from one of those, the
    # geometry already carries the steering and a preset would add it twice.
    steering_preset = config.get("steering_tape_preset_extension")
    if (
        steering_preset is not None
        and steering_tape_indices is not None
        and len(steering_tape_indices) >= 2
        and abs(float(steering_preset)) > 1e-12
    ):
        update_steering_tape_actuation(
            psystem=psystem,
            steering_tape_indices=steering_tape_indices,
            steering_tape_extension_step=float(steering_preset),
            initial_length_steering_left=initial_length_steering_left,
            initial_length_steering_right=initial_length_steering_right,
            steering_tape_final_extension=float(steering_preset),
        )
        logging.info(
            "Steering preset applied directly: tape delta %+.4f m "
            "(left %.4f, right %.4f)",
            float(steering_preset),
            float(initial_length_steering_left) - float(steering_preset),
            float(initial_length_steering_right) + float(steering_preset),
        )

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

    # Wing elements are parsed first, bridle lines appended after them.
    n_wing_elements = len(kite_connectivity_arr) - (
        0 if bridle_connectivity_arr is None else len(bridle_connectivity_arr)
    )
    # ONE resolution of the convergence measure for the whole solve, so the
    # loop, the adaptive dt and the stagnation window can never disagree about
    # what "converged" means or what units it is in.
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

    # The structure flies the stiffnesses of its file (E*A/l0 from each line's
    # material and diameter). The elongation-bound stiffening that raised them
    # in the loop was removed on 2026-11-02: it existed for the PSM file's
    # ten-times-too-soft lines and never fires on the datasheet moduli
    # (0.13 % elongation against its 1 % bound on the LEI-V3).
    handover_block = config.get("solver_state_handover") or {}
    for key in ("stiffnesses", "stiffness_factors"):
        if handover_block.get(key) is not None:
            logging.warning(
                "Ignoring handover %s: the wireframe flies its structure file's "
                "own stiffnesses.", key,
            )
    #: Largest element elongation (l - l0) / l0 of the latest shape, wing and
    #: bridle: a diagnostic, reported in the meta.
    max_wing_elongation = float("nan")
    max_bridle_elongation = float("nan")
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
    # Why the loop ended, the same values as the Billow driver:
    # converged | stagnated | diverged | stopped | max_iter.
    exit_reason = "max_iter"
    # Second exit condition: the tether force must have settled since the last
    # state change (convergence.force_settle_state). 0 disables. Gates the
    # main tier only; the plateau fallback keeps its own acceptance.
    force_settle_rtol = float(
        config["aero_structural_solver"].get("force_settle_rtol", 0.0) or 0.0
    )
    force_since_change = []
    force_remaining = float("nan")
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
    # REALISED-ACTUATION GATE (2026-09-23). The settle counter is a minimum
    # hold; it cannot tell whether the structure has ARRIVED at its rest
    # lengths. With Aitken at its floor the tape knots creep toward the
    # commanded lengths ~1 mm per iteration while the force residual sits
    # under the gate, so the 2026-09-13 Billow steering chains exited rows
    # u_s 0.075-0.125 with the realised half-difference 15-25 mm short of the
    # command and still moving 0.8-1.3 mm per iteration -- a lower steering
    # gain that read as a second equilibrium (the turn-rate kink at 0.150).
    # An exit on a steered solve therefore also needs the REALISED
    # half-difference (tape endpoint distances, `steering_half_differences`)
    # within steering_realised_tol_m of the command AND the creep still to
    # come -- the geometric tail of its history since the last steering step
    # (`remaining_drift`; Aitken at its floor gives a ratio ~0.95, so a
    # per-iteration threshold alone leaves 2-3 mm on the table) -- within
    # steering_realised_remaining_tol_m. Rows that finished their walk sit
    # within 0.4 mm and move < 0.1 mm per iteration.
    steering_realised_tol_m = float(
        config["aero_structural_solver"].get("steering_realised_tol_m", 5e-3)
    )
    steering_realised_remaining_tol_m = float(
        config["aero_structural_solver"].get(
            "steering_realised_remaining_tol_m", 1e-3
        )
    )
    # A value that has STOPPED moving is realised whatever the tail estimator
    # says: on a period-2 residual cycle the last changes are nanometres of
    # noise with ratios >= 1, which `remaining_drift` reads as "infinitely far"
    # (va21/va25 u_s 0.025 held 200 iterations on 0.0249940 m, 2026-09-23).
    steering_realised_rate_floor_m = float(
        config["aero_structural_solver"].get("steering_realised_rate_floor_m", 2e-5)
    )
    is_steering_gate_active = (
        steering_tape_indices is not None
        and len(steering_tape_indices) >= 2
        and initial_length_steering_left is not None
        and initial_length_steering_right is not None
        and abs(float(steering_tape_final_extension or 0.0)) > 1e-12
        and steering_realised_tol_m > 0.0
    )
    if is_steering_gate_active:
        # Mapping check: the tapes must measure about their initial lengths
        # on the starting geometry, else the gate would hold every solve.
        _cmd0, _real0 = steering_half_differences(
            psystem.extract_rest_length, struc_nodes, kite_connectivity_arr,
            steering_tape_indices, initial_length_steering_left,
            initial_length_steering_right,
        )
        if not np.isfinite(_real0) or abs(_real0 - _cmd0) > 0.25 * max(
            float(initial_length_steering_left), 1e-6
        ):
            logging.warning(
                "Realised-steering gate DISABLED: tape endpoints read a "
                "half-difference of %.3f m against a command of %.3f m on the "
                "starting geometry (tape indices %s do not address the tape "
                "nodes?)",
                _real0, _cmd0, list(steering_tape_indices[:2]),
            )
            is_steering_gate_active = False
    steering_commanded_m = float("nan")
    steering_realised_m = float("nan")
    steering_realised_rate_m = float("nan")
    steering_realised_remaining_m = float("nan")
    steering_realised_history: list = []  # since the last steering step
    is_steering_realised = True
    qs_opt_prev_rounded = None
    qs_stag_counter = 0
    qs_state_should_break = False
    #: Per-iteration trim state [kite_speed, roll, pitch, yaw, course_rate]
    #: and the largest |course_rate| seen on iterations where the steering
    #: actuation was fully applied and settled -- see the in-loop comment.
    opt_x_history: list = []
    course_rate_max_settled = 0.0
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
        bridle_node_pairs = _bridle_node_pairs(
            bridle_line_specs, struc_nodes, body_aero
        )

    # Aitken relaxation state.
    #
    # This is SOLVER state, not model state, and it is what a snapshot restart
    # silently throws away. A cold solve walks omega down from 0.3 towards the
    # 0.05 floor as the coupling stiffens; a restart from a converged shape
    # starts again at 0.3 and takes a step several times larger than the one
    # its own base had settled on. Together with a missing previous increment
    # (which leaves Aitken unable to adapt for two more iterations), that is
    # enough to knock a converged shape off its fixed point on the first
    # iteration -- the handover then looks like an aero problem when it is
    # really the loop restarting undamped.
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
    # (structural_wireframe.solver: nlp): PSS's ~1 N inner noise corrupts the
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
    # Defaults are the NOISE-ROBUST setting measured 2026-09-10 on the
    # frame-consistent history (actuated case, exact NLP inner solve, full
    # trim): depth 2 / ridge 1e-4 / beta 0.2 / max step 0.1 m contracts
    # monotonically ~1e-3 -> 1.2e-4 in ~25 post-actuation iterations and then
    # floors at the outer map's own evaluation noise. The original aggressive
    # setting (depth 4, ridge 1e-8, beta 0.5) BOUNCES 2e-4..1e-2 even with
    # the rotated history -- near the fixed point the iterate differences
    # approach the noise floor and the barely-regularized extrapolation
    # amplifies them; beta 1.0 diverges onto a wrong branch.
    anderson_outer_depth = int(
        config["aero_structural_solver"].get("anderson_outer_depth", 2)
    )
    anderson_outer_beta = float(
        config["aero_structural_solver"].get("anderson_outer_beta", 0.2)
    )
    anderson_outer_reg = float(
        config["aero_structural_solver"].get("anderson_outer_reg", 1e-4)
    )
    anderson_outer_max_step = float(
        config["aero_structural_solver"].get("anderson_outer_max_step_m", 0.1)
    )
    anderson_xs: list = []
    anderson_rs: list = []
    anderson_map_signature = None

    def _rotate_anderson_history(roll_deg, pitch_deg, yaw_deg):
        """Carry the Anderson history into the frame the geometry just rotated to.

        The driver applies the solved trim attitude to ``struc_nodes`` every
        iteration, so consecutive outer iterates live in DIFFERENT frames.
        Anderson extrapolates over differences of stored iterates; left
        unrotated, those differences are dominated by the attitude increment
        (a 1 deg rotation moves outboard nodes centimetres, against
        millimetre structural increments near the fixed point), which is the
        measured 2026-09-10 failure of the outer acceleration. Rotating the
        stored positions and residuals with exactly the geometry's rotation
        keeps the whole history in the current frame; in de-rotated
        coordinates the outer map is smooth again. Aitken is deliberately
        untouched -- its scalar update only spans two consecutive frames and
        its historical behaviour is the campaign baseline.
        """
        nonlocal r_prev_flat
        if not anderson_xs and r_prev_flat is None:
            return
        rotation = rotation_matrix_from_angles(
            angle_deg=[roll_deg, pitch_deg, yaw_deg]
        )
        anderson_xs[:] = [
            (x.reshape(-1, 3) @ rotation.T).reshape(-1) for x in anderson_xs
        ]
        anderson_rs[:] = [
            (r.reshape(-1, 3) @ rotation.T).reshape(-1) for r in anderson_rs
        ]
        if r_prev_flat is not None:
            r_prev_flat = (
                (r_prev_flat.reshape(-1, 3) @ rotation.T).reshape(-1)
            )
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
        mapper=structural_to_aero,
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
        # config["tether"]["include_in_trim"] is set. The pre-loop trim
        # sets the whole attitude, so it takes quasi_steady_trim.pre_loop.
        config=aerodynamic_vsm.pre_loop_trim_config(config),
    )
    logging.debug(
        f"Aero symmetry check, f_aero_y: {np.sum([force[1] for force in f_aero_wing_vsm_format])}"
    )
    tracking.update_aero_tracking(
        tracking_data, 0,
        results_aero.get("alpha_at_ac"),
        results_aero.get("stall_mask"),
    )
    # Wall time spent inside the trim, summed over the coupled iterations
    # (quasi_steady_trim.solver A/B: the trim is the coupled loop's cost).
    trim_time_total_s = float(results_aero.get("trim_time_s", 0.0) or 0.0)
    print(
        f"  Trim solver: {results_aero.get('trim_solver', '?')}, "
        f"{trim_time_total_s:.2f} s"
    )
    gamma_seed_prev = results_aero.get("gamma_distribution", gamma_seed_prev)
    roll, pitch, yaw = results_aero["opt_x"][1:4]
    struc_nodes = rotate_geometry(struc_nodes, angle_deg=[roll, pitch, yaw])
    # As in coupled_solver: positions[0] is the geometry BEFORE this rotation,
    # so composing trim_state rows 0..n recovers the kite's attitude.
    tracking_data["trim_state"][0] = np.ravel(results_aero["opt_x"])[:5]
    if outer_acceleration == "anderson":
        _rotate_anderson_history(roll, pitch, yaw)
    ### AERO --> STRUC (and the moment-preservation check, pre-loop)
    f_aero_wing = _map_aero_loads_to_structure(
        f_aero_wing_vsm_format,
        struc_nodes,
        results_aero,
        aero2struc_mapping,
        config,
        check=True,
    )

    ### BRIDLE AERO
    # Not optional, and not zero: the trim has already balanced this drag
    # (the VSM body is built with the bridle lines, so `calculate_results` puts
    # their force and their moment about the reference point into the balance
    # the attitude comes out of). Zeroing it here left the FIRST structural
    # solve 274.7 N m of pitch out of balance about the bridle point -- which
    # it is PINNED at, so it can only shed that by swinging 0.886 deg. The
    # fixed point does not move, but the loop is seeded off it, and this map
    # has more than one branch to be seeded onto.
    if config["is_with_aero_bridle"]:
        f_aero_bridle = aerodynamic_bridle_line_drag.main(
            struc_nodes,
            bridle_connectivity_arr,
            bridle_diameter_arr,
            vel_app,
            air_density_at_case(config),
            config["aerodynamic_bridle"]["cd_cable"],
            config["aerodynamic_bridle"]["cf_cable"],
            # With the body, the drag is taken at the TRIM's apparent wind
            # (`body_aero.va`) on the segments it just rebuilt -- `vel_app`,
            # the freestream, is then only a fallback.
            body_aero=body_aero,
            bridle_node_pairs=bridle_node_pairs,
        )
    else:
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
            # There is no adaptive time step any more. `dt` was the
            # kinetic-damping solver's continuation parameter; energy
            # minimisation has no time step -- stiffness is just a coefficient
            # of its objective -- so the schedule that used to live here
            # computed a value nothing read.
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
                    + structural_wireframe.get_stiffnesses(psystem).tobytes()
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

            ### REALISED STEERING (see steering_realised_tol_m above)
            if is_steering_gate_active:
                steering_commanded_m, realised_now = steering_half_differences(
                    psystem.extract_rest_length, struc_nodes,
                    kite_connectivity_arr, steering_tape_indices,
                    initial_length_steering_left, initial_length_steering_right,
                )
                steering_realised_rate_m = (
                    abs(realised_now - steering_realised_m)
                    if np.isfinite(steering_realised_m)
                    else float("inf")
                )
                steering_realised_m = realised_now
                steering_realised_history.append(realised_now)
                steering_realised_remaining_m = (
                    0.0
                    if steering_realised_rate_m <= steering_realised_rate_floor_m
                    else remaining_drift(steering_realised_history)
                )
                is_steering_realised = bool(
                    abs(steering_realised_m - steering_commanded_m)
                    <= steering_realised_tol_m
                    and steering_realised_remaining_m
                    <= steering_realised_remaining_tol_m
                )

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
                mapper=structural_to_aero,
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
            trim_time_total_s += float(results_aero.get("trim_time_s", 0.0) or 0.0)
            print(
                f"  Trim solver: {results_aero.get('trim_solver', '?')}, "
                f"{float(results_aero.get('trim_time_s', float('nan')) or float('nan')):.2f} s"
                f" (av_stage {results_aero.get('av_stage')})"
            )

            # Trim-state history across the coupled iterations, and the
            # largest settled course rate. The actuation ramp makes every
            # cold solve its own continuation in the tape length, so a final
            # |course rate| far below the maximum it reached AFTER the
            # steering was fully applied marks a solve that fell off the
            # loaded steering branch onto the unloaded (slack-path) attached
            # equilibrium -- a donor-free branch check for sweep callers.
            opt_x_history.append(
                np.asarray(results_aero.get("opt_x", np.full(5, np.nan)),
                           dtype=float).ravel()
            )
            if (
                is_steering_finalized
                and steering_settle_counter == 0
                and opt_x_history[-1].size == 5
                and np.isfinite(opt_x_history[-1][4])
            ):
                course_rate_max_settled = max(
                    course_rate_max_settled, abs(float(opt_x_history[-1][4]))
                )

            # Runaway stop (see qs_speed_bound_patience above). Counted only
            # once the tape walk has finished and settled: mid-walk the kite
            # is not the one being solved for (part-way down the depower walk
            # it is still powered, and faster), so a pinned trim then says
            # nothing about the target state. Counting through the walk ended
            # a reel-in grid point (u_dp 0.277) at iteration 5 as a fully
            # powered 25.7 kN kite whose neighbours converge at 6.5 kN
            # (2026-09-13). The flags are the previous iteration's walk
            # update, which is what this iteration's shape was solved with.
            if qs_speed_bound_patience > 0:
                is_walk_settled = (
                    is_actuation_finalized
                    and is_steering_finalized
                    and depower_settle_counter == 0
                    and steering_settle_counter == 0
                    and is_steering_realised
                )
                if is_walk_settled and "kite_speed" in (
                    results_aero.get("trim_on_bounds") or []
                ):
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
            if outer_acceleration == "anderson":
                _rotate_anderson_history(roll, pitch, yaw)
            ### AERO --> STRUC (moment check on the first coupling iteration
            ### only, to limit log spam)
            f_aero_wing = _map_aero_loads_to_structure(
                f_aero_wing_vsm_format,
                struc_nodes,
                results_aero,
                aero2struc_mapping,
                config,
                check=(i == 1),
            )

            ### BRIDLE AERO
            if config["is_with_aero_bridle"]:
                f_aero_bridle = aerodynamic_bridle_line_drag.main(
                    struc_nodes,
                    bridle_connectivity_arr,
                    bridle_diameter_arr,
                    vel_app,
                    air_density_at_case(config),
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
                trim_state=results_aero["opt_x"],
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

            ### FORCE SETTLE (see force_settle_rtol)
            force_since_change.append(force_reference)
            is_force_settled, force_remaining = force_settle_state(
                force_since_change, force_settle_rtol
            )
            if is_convergence and not is_force_settled:
                logging.info(
                    "residual below the gate at iteration %d but the tether "
                    "force has not settled (%d value(s) since the last change, "
                    "%.1e of it still moving, tolerance %.1e); held",
                    i, len(force_since_change), force_remaining, force_settle_rtol,
                )
                is_convergence = False

            ### ELEMENT ELONGATION (a diagnostic; nothing is stiffened)
            elongations = element_elongations(
                struc_nodes,
                kite_connectivity_arr,
                psystem.extract_rest_length,
                pulley_pairs=pulley_line_to_other_node_pair_dict,
            )
            if n_wing_elements > 0 and elongations.size:
                max_wing_elongation = float(np.nanmax(elongations[:n_wing_elements]))
                if elongations.size > n_wing_elements:
                    max_bridle_elongation = float(
                        np.nanmax(elongations[n_wing_elements:])
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
                    and is_steering_realised
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

            if did_update_depower or did_update_steering:
                force_since_change = []
            if did_update_depower:
                depower_settle_counter = depower_settle_iterations_after_update
            elif depower_settle_counter > 0:
                depower_settle_counter -= 1
            if did_update_steering:
                steering_settle_counter = steering_settle_iterations_after_update
                steering_realised_history = []
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
            # Walk applied and settled, but the structure has not arrived at
            # the commanded tape lengths yet: hold (see the gate above).
            if not is_steering_realised:
                if is_convergence:
                    logging.info(
                        "residual below the gate but the steering is not "
                        "realised: half-difference %.4f m of %.4f m commanded, "
                        "moving %.2e m/iteration with %.2e m still to come "
                        "(tol %.1e m, %.1e m); held",
                        steering_realised_m, steering_commanded_m,
                        steering_realised_rate_m, steering_realised_remaining_m,
                        steering_realised_tol_m,
                        steering_realised_remaining_tol_m,
                    )
                continue

            ### PLATEAU DETECTION
            # Accept a solve that has stopped improving and is already inside
            # the fallback, rather than spending the rest of max_iter proving
            # the cycle is a cycle. Everything the post-loop acceptance
            # requires is required here too, so this only changes WHEN such a
            # point is accepted, never WHETHER.
            if plateau_patience > 0 and not is_convergence and f_residual_list:
                residual_now_plateau = float(f_residual_list[-1])
                if (
                    np.isfinite(residual_now_plateau)
                    and residual_now_plateau < residual_best
                ):
                    residual_best = residual_now_plateau
                    residual_best_iteration = i
                elif (
                    residual_tol_fallback > residual_tol_active
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
            if should_break or (
                is_convergence
                and is_actuation_finalized
                and is_steering_finalized
                and is_steering_realised
            ):
                if is_convergence:
                    exit_reason = "converged"
                elif is_stagnated:
                    exit_reason = "stagnated"
                elif not np.all(np.isfinite(np.asarray(f_residual, dtype=float))):
                    exit_reason = "diverged"
                else:
                    exit_reason = "stopped"
                break
    ######################################################################
    ## END OF SIMULATION FOR LOOP
    ######################################################################
    if is_steering_gate_active:
        logging.info(
            "Steering at exit: half-difference %.4f m realised of %.4f m "
            "commanded, last change %.2e m/iteration, %.2e m still to come "
            "(%s)",
            steering_realised_m, steering_commanded_m, steering_realised_rate_m,
            steering_realised_remaining_m,
            "realised" if is_steering_realised else "NOT REALISED",
        )

    ### PLATEAU ACCEPTANCE
    # A solve that ran out of iterations or stagnated has not necessarily
    # failed: steered, near-stall and heavily loaded cases settle into a LIMIT
    # CYCLE rather than a fixed point, with the Aitken factor already pinned at
    # its floor, so neither more iterations nor more damping can close them.
    # Accept those at the looser `residual_tol_relative_fallback` -- but only
    # once every PHYSICAL gate is satisfied, so this can never rescue a run
    # that stopped mid-actuation. `residual_cycle_span` reports the orbit the point sits on, which
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
            and is_steering_realised
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
                "(actuation, steering) is "
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
    mid_idx = len(body_aero.panels) // 2
    angle = aerodynamic_vsm.mid_span_chord_angle_of_attack(body_aero, body_aero.va)
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

    # Final word on why the run ended: the plateau acceptance above can still
    # turn a max_iter / stagnated run into a converged one (see
    # converged_at_fallback), and a trim_speed_bound stop leaves the loop
    # without passing the exit check.
    if is_convergence:
        exit_reason = "converged"
    elif stop_reason is not None:
        exit_reason = "stopped"

    meta = {
        "total_time_s": time.time() - start_time,
        "n_iter": i + 2,  # +2: 1 for pre-loop initial state + (i+1) loop entries
        "converged": is_convergence,
        # Why the loop ended early, when it did for a reason other than
        # convergence / stagnation / max_iter: "trim_speed_bound" (runaway
        # stop, see qs_speed_bound_patience). None otherwise.
        "stop_reason": stop_reason,
        # Why the loop ended, the same values as the Billow driver:
        # converged | stagnated | diverged | stopped | max_iter.
        "exit_reason": exit_reason,
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
        "residual_tol_active": float(residual_tol_active),
        # True when the solve was accepted on the LOOSER fallback because it
        # plateaued on a limit cycle rather than reaching a fixed point.
        # `residual_cycle_span` is that orbit's width in the active measure.
        "converged_at_fallback": bool(is_converged_at_fallback),
        "force_remaining_relative": float(force_remaining),
        "force_settle_rtol": float(force_settle_rtol),
        "residual_tol_fallback": float(residual_tol_fallback),
        "residual_cycle_span": float(residual_cycle_span),
        "residual_best": float(residual_best),
        "residual_best_iteration": int(residual_best_iteration),
        "plateau_patience": int(plateau_patience),
        "residual_tol_is_relative": bool(is_residual_relative),
        # Largest element elongation of the converged shape (a diagnostic).
        # Pulley arms carry the ROPE's strain (both arms combined), not their
        # individual length change.
        "max_element_elongation": max_element_elongation(
            struc_nodes,
            kite_connectivity_arr,
            rest_lengths,
            pulley_pairs=pulley_line_to_other_node_pair_dict,
        ),
        "max_wing_element_elongation": float(max_wing_elongation),
        "max_bridle_element_elongation": float(max_bridle_elongation),
        # Stiffnesses the converged shape ran with (the structure file's own;
        # ``PreparedWireframe.restore`` puts them back).
        "final_stiffnesses": np.array(
            [float(link.k) for link in psystem.springdampers], dtype=float
        ),
        "qs_success": bool(results_aero.get("success", False)),
        # Which trim solver the coupled loop ran (quasi_steady_trim.solver)
        # and the wall time it spent inside it, all iterations summed.
        "trim_solver": str(results_aero.get("trim_solver", "least_squares")),
        "trim_time_total_s": float(trim_time_total_s),
        "opt_x": opt_x,
        # The final circulation: with opt_x, what a continuation resumed from
        # this file starts from (``PreparedWireframe.restore``).
        "gamma_distribution": np.asarray(
            results_aero.get("gamma_distribution", np.full(0, np.nan)), dtype=float
        ),
        # Trim state per coupled iteration, (n, 5). With the largest settled
        # |course_rate| below it gives sweep callers a donor-free branch
        # check: a final |opt_x[4]| far under course_rate_max_settled means
        # the solve fell off the loaded steering branch during its own
        # actuation ramp (unloaded slack-path attached equilibrium).
        "opt_x_history": (
            np.vstack(opt_x_history) if opt_x_history else np.zeros((0, 5))
        ),
        "course_rate_max_settled": float(course_rate_max_settled),
        # Realised-actuation gate at exit (NaN when the gate was inactive):
        # the steering half-difference the STRUCTURE shows against the one
        # the rest lengths command, and how fast it was still moving.
        "steering_commanded_m": float(steering_commanded_m),
        "steering_realised_m": float(steering_realised_m),
        "steering_realised_rate_m": float(steering_realised_rate_m),
        "steering_realised_remaining_m": float(steering_realised_remaining_m),
        "is_steering_realised": bool(is_steering_realised),
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
        # Wing-averaged SECTIONAL cl/cd (kept for stored-result compatibility);
        # the kite's own coefficients are in trim_results["cl"/"cd"].
        "cl": float(cl),
        "cd": float(cd),
        # The final trim's scalars and 3-vectors (total aero force, apparent
        # wind, CL/CD on the VSM projected area, ...), the same block the
        # Billow driver stores; read by awetrim.aerostructural.case_view.
        "trim_results": _trim_results_summary(results_aero),
        "tether_force": float(tether_force),
        "rest_lengths": rest_lengths,
        # The structure's own node positions at exit: what the next solve on
        # the same structure starts from. NOT tracking positions[-1], which is
        # that shape after the final trim rotation (a fraction of a degree, a
        # few cm at the tips) -- the structure never receives it.
        "structure_positions": np.asarray(psystem.positions, dtype=float).copy(),
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

    if final_state is not None:
        final_state.update(body_aero=body_aero, results_aero=results_aero)

    return tracking_data, meta


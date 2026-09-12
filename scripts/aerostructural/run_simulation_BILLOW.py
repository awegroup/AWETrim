"""Run a single Billow/QSM aerostructural simulation.

The structural side is the minimum-energy model in ``billow``:
cables, frictionless pulleys, inflatable Timoshenko tube beams and a wrinkling
CST membrane canopy, all posed as one total potential energy and solved with
IPOPT. The aerodynamic side, the load mapping and the outer fixed-point loop
are the same ones the FEM path uses.

Geometry comes from ``struc_geometry_FEM_full.yaml`` -- the only format that
carries strut tubes, leading-edge tubes and inflation pressure, which is what
the beams and the canopy mesh need. Config is shared with the PSS and FEM
scripts (``as_config.yaml``); ``structural_solver`` is forced to ``billow`` and
the numerics live under the ``structural_billow`` block.

Two differences from the FEM path are worth knowing before reading results:

* **The bridle is relaxed first.** The YAML stores measured rest lengths
  against measured node positions and the two disagree (``Br_main_1`` is 11.5%
  long, i.e. 88 kN at the real ``EA/l0``). ``structural_billow.instantiate``
  settles the bridle onto the held wing before the loop starts, which is the
  same role ``structural_kite_fem.relaxbridles`` plays on the FEM path, so the
  nodes the loop starts from are *not* the raw YAML nodes.
* **The canopy is a membrane, not a spring net**, so the 693 canopy springs of
  the FEM model are replaced by 378 wrinkling triangles on the same grid.

Usage (from project root):
    python scripts/aerostructural/run_simulation_BILLOW.py
"""

import copy
from pathlib import Path

import numpy as np
import yaml as _yaml

from awetrim.aerostructural.logging_config import *  # noqa: F401,F403
from awetrim.aerostructural.mapping import BilinearAeroToStructuralLoadMapper
from awetrim.aerostructural.results import (
    aerostructural_results_root,
    build_deformed_aero_geometry,
    build_deformed_struc_geometry,
    save_geometry_snapshot,
    save_input_snapshot,
    save_sim_output,
)
from awetrim.aerostructural.utils import load_yaml, rotate_geometry
from awetrim.aerostructural import aerodynamic_vsm
from awetrim.aerostructural.billow import structural_billow
from awetrim.aerostructural.coupled import (
    coupled_solver,
    read_struc_geometry_yaml,
)
from awetrim.system.tether import RigidLumpedTether
from awetrim.utils.system_config import get_tether
from common import (
    DEFAULT_KITE_NAME,
    build_actuation_case_folder,
    build_system_model,
    resolve_initial_geometry_rotation_kwargs,
    resolve_kite_paths,
)
from awesio.validator import validate as awesio_validate

#: Billow needs strut tubes, leading-edge tubes and an inflation pressure, so it
#: reads the same full geometry the FEM path does. The reduced PSM files are
#: bridle-and-spring only and cannot feed beams or a canopy mesh.
STRUC_GEOMETRY_FILENAME = "struc_geometry_FEM_full.yaml"


def main():
    PROJECT_DIR = Path(__file__).resolve().parents[2]
    kite_name = DEFAULT_KITE_NAME

    config_path, aero_geometry_path, _ = resolve_kite_paths(PROJECT_DIR, kite_name)
    struc_geometry_path = PROJECT_DIR / "data" / kite_name / STRUC_GEOMETRY_FILENAME

    system_config_path = PROJECT_DIR / "data" / kite_name / "system.yaml"
    with system_config_path.open("r", encoding="utf-8") as _f:
        system_config = _yaml.safe_load(_f)
    awesio_validate(system_config, restrictive=False)

    config = load_yaml(config_path)
    config["structural_solver"] = "billow"
    # Conserve the panel pitching moment, not just the panel force. A single
    # fixed Delta C_p shape pins every panel's resultant at one chordwise
    # station whatever its own C_m says, and since the wing's trim is a moment
    # balance the coupled loop then re-trims each iteration against a moment
    # the structure never received. See aero2struc.weights_at_centre_of_pressure.
    config.setdefault("aero2struc", {})["chordwise_distribution"] = "moment_matched"
    # Trim the UNACTUATED kite. The shared as_config carries a depower ramp of
    # +0.2 m and a steering ramp of +0.3 m, so a run that reaches the residual
    # gate immediately starts stepping the tapes and the answer is a sequence of
    # actuated states rather than the baseline. The baseline is what a first
    # Billow-vs-FEM comparison needs, so the ramps are switched off here rather
    # than in as_config, which the FEM and PSS scripts share.
    #
    # "Unactuated" means the tape lengths the geometry YAML stores, not zero
    # tape: struc_geometry_FEM_full.yaml has depower_tape l0 = 1.9 m, which
    # through the depower map is u_dp ~ 0.28. Change that l0 to trim at a
    # different depower setting.
    config["power_tape_final_extension"] = 0.0
    config["power_tape_extension_step"] = 0.0
    config["steering_tape_final_extension"] = 0.0
    config["steering_tape_extension_step"] = 0.0

    # Fly the kite at a realistic apparent speed. as_config's state is the dead
    # centre of the window -- zero elevation, zero azimuth, 90 deg course, no
    # gravity -- which is the FASTEST corner there is: at 8 m/s wind and the
    # trimmed L/D of ~4.8 it comes out at v_a ~ 38 m/s, above the whole
    # 13-25 m/s band the wes-quasi-steady campaigns sweep (TARGETS19 in
    # run_center_sweeps_continuation.sh).
    #
    # That matters here in a way it does not for a spring-net canopy. The
    # inflatable tube law SATURATES at moment_max, so the usual "coupled trims
    # are v^2-self-similar" argument does not carry: at 38 m/s every strut sits
    # 3-4.7x past its collapse curvature, in the flat part of the law where the
    # tube has no incremental bending stiffness left, and hinges into ~20% of
    # chord. Load is what decides whether the tubes are inside their calibrated
    # range, so the load has to be the real one.
    #
    # v_a scales with the wind at fixed geometry, so 8 * 20 / 38.19 ~ 4.2 m/s
    # puts it near 20 m/s. The campaigns iterate wind onto a target v_a with a
    # ratio table; this is the one-shot version of the same idea.
    config["wind_speed_wind_ref"] = 4.2

    # Aero panel count. The coupled body is built from the STRUCTURAL sections,
    # not from a panel count directly, so the reachable counts are
    # (n_sections - 1) * n_aero_panels_per_struc_section. The wes-quasi-steady
    # campaigns run 5 per section on the PSM geometry -- 9 sections, 45 panels --
    # but Billow reads struc_geometry_FEM_full.yaml, whose 28 leading-edge nodes
    # give 27 sections, so the same 5 would be 135. The reachable counts here are
    # 27, 54, 81, ...; 2 per section (54) is the closest to the campaigns' 45 and
    # sits above the 27/36/45 band as_config records as mesh converged.
    config["aerodynamic"]["n_aero_panels_per_struc_section"] = 2

    # The VSM inner solver is NOT overridden here. as_config already ships the
    # configuration the wes-quasi-steady campaigns actually run with --
    # gamma_loop_type anderson, allowed_error 1e-8, anderson_max_iterations 1000,
    # no Picard fallback -- so the coupled answer and the simpler single-wing
    # model share one inner solver by reading one file. (The argparse defaults in
    # those scripts say "base" / 1e-6, but no campaign uses them; see
    # run_center_sweeps_continuation.sh and run_steering_continuation.sh.)
    # allowed_error and gamma_loop_type are a MATCHED PAIR: Anderson terminates
    # on a superlinear, non-smooth residual that corrupts the trim's
    # finite-difference Jacobian at a loose tolerance. Change both or neither.

    # Resolve cp_distribution_path relative to project root if given as a relative path.
    cp_rel = config.get("aero2struc", {}).get("cp_distribution_path")
    if cp_rel:
        config["aero2struc"]["cp_distribution_path"] = str(PROJECT_DIR / cp_rel)

    case_folder = build_actuation_case_folder(config)
    results_root = aerostructural_results_root(PROJECT_DIR, kite_name)
    results_dir = results_root / "billow" / case_folder
    struc_geometry = load_yaml(struc_geometry_path)
    aero_geometry = load_yaml(aero_geometry_path)
    results_dir = save_input_snapshot(config=config, results_dir=results_dir)

    ###################
    ### AERODYNAMIC ###
    ###################
    n_wing_struc_nodes = len(struc_geometry["wing_particles"]["data"])
    n_struc_ribs = n_wing_struc_nodes / 2
    n_panels_aero = (n_struc_ribs - 1) * config["aerodynamic"][
        "n_aero_panels_per_struc_section"
    ]
    bridle_path = (
        struc_geometry_path if config.get("is_with_aero_bridle", False) else None
    )
    body_aero, vsm_solver, vel_app, initial_polar_data = aerodynamic_vsm.initialize(
        aero_geometry_path,
        config,
        n_panels_aero,
        bridle_path=bridle_path,
    )

    ##################
    ### STRUCTURAL ###
    ##################
    (
        struc_nodes,
        m_arr,
        struc_node_le_indices,
        struc_node_te_indices,
        power_tape_index,
        steering_tape_indices,
        pulley_node_indices,
        canopy_sections,
        strut_sections,
        simplified_bridle_points,
        kite_connectivity_arr,
        bridle_connectivity_arr,
        bridle_diameter_arr,
        l0_arr,
        k_arr,
        c_arr,
        linktype_arr,
        pulley_line_indices,
        pulley_line_to_other_node_pair_dict,
    ) = read_struc_geometry_yaml.main(
        struc_geometry, config=config, system_config=system_config
    )

    # Rotate before the model is built, so the bridle relaxation and every
    # reference strain are taken in the attitude the run actually starts from.
    struc_nodes = rotate_geometry(
        struc_nodes,
        **resolve_initial_geometry_rotation_kwargs(config),
    )

    billow_structure = structural_billow.instantiate(
        config=config,
        struc_geometry=struc_geometry,
        struc_nodes=struc_nodes,
        kite_connectivity_arr=kite_connectivity_arr,
        l0_arr=l0_arr,
        k_arr=k_arr,
        c_arr=c_arr,
        m_arr=m_arr,
        linktype_arr=linktype_arr,
        pulley_line_indices=pulley_line_indices,
        canopy_sections=canopy_sections,
        strut_sections=strut_sections,
    )
    # instantiate() relaxes the bridle, so the model's nodes are not the ones it
    # was handed. Everything downstream -- the aero mesh, the load mapping, the
    # coupled loop -- has to start from the relaxed shape or the structure and
    # the aerodynamics disagree from the first iteration.
    # instantiate() may ADD nodes (the cross canopy pattern puts one at each
    # quad centre), so the node count downstream is the model's, not the
    # reader's. Take the masses from the structure for the same reason: every
    # array the driver sizes -- external forces, tracking, the CG -- has to
    # agree with model.nodes or they silently misalign.
    struc_nodes = billow_structure.model.nodes.copy()
    m_arr = billow_structure.masses
    # The aerodynamic load is mapped onto these sections. Use the MEMBRANE's
    # grid, not the reader's: with a refined canopy the two differ, and mapping
    # onto the coarse one leaves every added node unloaded. aero2struc's
    # bracketing is symmetry-preserving for any section list, so a finer list
    # costs nothing but resolution gained.
    canopy_sections = [list(map(int, row)) for row in billow_structure.fine_grid]
    strut_sections = []
    struc_nodes_initial = struc_nodes.copy()

    ##################
    ### AERO2STRUC ###
    ##################
    aero2struc_mapping = (
        BilinearAeroToStructuralLoadMapper()
        .initialize(
            body_aero.panels,
            struc_nodes,
            struc_node_le_indices,
            struc_node_te_indices,
        )
        .panel_corner_map
    )

    #################
    ### ACTUATION ###
    #################
    initial_length_power_tape = l0_arr[power_tape_index]
    power_tape_extension_step = config["power_tape_extension_step"]
    power_tape_final_extension = config["power_tape_final_extension"]
    n_power_tape_steps = (
        int(power_tape_final_extension / power_tape_extension_step)
        if power_tape_extension_step != 0
        else 0
    )

    ########################################
    # AWETRIM SYSTEM MODEL
    ########################################
    tether_struct = get_tether(system_config)["structure"]
    tether = RigidLumpedTether(
        diameter=tether_struct["diameter"],
        density=tether_struct.get("density", 970.0),
    )
    print(f"Total structural mass: {float(np.sum(m_arr)):.3f} kg")
    system_model = build_system_model(system_config_path, tether, m_arr, config)

    ########################################
    ### AEROSTRUCTURAL COUPLED SIMULATION ##
    ########################################
    tracking_data, meta = coupled_solver.main(
        m_arr=m_arr,
        struc_nodes=struc_nodes,
        struc_nodes_initial=struc_nodes_initial,
        system_model=system_model,
        config=config,
        ### ACTUATION
        initial_length_power_tape=initial_length_power_tape,
        n_power_tape_steps=n_power_tape_steps,
        power_tape_final_extension=power_tape_final_extension,
        power_tape_extension_step=power_tape_extension_step,
        ### CONNECTIVITY
        kite_connectivity_arr=kite_connectivity_arr,
        bridle_connectivity_arr=bridle_connectivity_arr,
        pulley_line_indices=pulley_line_indices,
        pulley_line_to_other_node_pair_dict=pulley_line_to_other_node_pair_dict,
        ### STRUC --> AERO
        struc_node_le_indices=struc_node_le_indices,
        struc_node_te_indices=struc_node_te_indices,
        ### AERO
        body_aero=copy.deepcopy(body_aero),
        vsm_solver=copy.deepcopy(vsm_solver),
        vel_app=vel_app,
        initial_polar_data=copy.deepcopy(initial_polar_data),
        bridle_diameter_arr=bridle_diameter_arr,
        ### AERO --> STRUC
        aero2struc_mapping=aero2struc_mapping,
        power_tape_index=power_tape_index,
        ### STRUC
        billow_structure=billow_structure,
        canopy_sections=canopy_sections,
        strut_sections=strut_sections,
    )

    save_sim_output(tracking_data, meta, results_dir)
    final_nodes = np.asarray(tracking_data["positions"][meta["n_iter"] - 1])
    save_geometry_snapshot(
        config,
        build_deformed_struc_geometry(struc_geometry, final_nodes),
        build_deformed_aero_geometry(
            aero_geometry, final_nodes, struc_node_le_indices, struc_node_te_indices
        ),
        results_dir,
    )


if __name__ == "__main__":
    main()

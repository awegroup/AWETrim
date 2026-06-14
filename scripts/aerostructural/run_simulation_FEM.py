"""Run a single FEM/QSM aerostructural simulation (kite_fem structural solver).

Geometry is read from struc_geometry_level_2_manual.yaml (FEM level-2 format with
strut tubes and leading-edge tubes). Config is shared with the PSS script
(as_config.yaml); the structural_solver key is forced to 'kite_fem'.

Usage (from project root):
    python scripts/aerostructural/run_simulation_level_2.py
"""

import copy
import sys
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
from awetrim.aerostructural.fem import (
    aerostructural_coupled_solver_level_2,
    read_struc_geometry_yaml_level_2,
    structural_kite_fem_level_2,
)
from awetrim.system.tether import RigidLumpedTether
from common import (
    CONFIG_DEFAULTS,
    DEFAULT_KITE_NAME,
    build_actuation_case_folder,
    build_system_model,
    resolve_initial_geometry_rotation_kwargs,
    resolve_kite_paths,
)
from awesio.validator import validate as awesio_validate

# FEM uses a separate structural geometry file with strut tubes and LE tubes.
STRUC_GEOMETRY_FILENAME = "struc_geometry_level_2_manual.yaml"


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
    config["structural_solver"] = "kite_fem"

    # Resolve cp_distribution_path relative to project root if given as a relative path.
    cp_rel = config.get("aero2struc", {}).get("cp_distribution_path")
    if cp_rel:
        config["aero2struc"]["cp_distribution_path"] = str(PROJECT_DIR / cp_rel)

    case_folder = build_actuation_case_folder(config)
    results_root = aerostructural_results_root(PROJECT_DIR, kite_name)
    results_dir = results_root / "fem" / case_folder
    struc_geometry = load_yaml(struc_geometry_path)
    aero_geometry = load_yaml(aero_geometry_path)
    results_dir = save_input_snapshot(
        config=config,
        results_dir=results_dir,
    )

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
    ) = read_struc_geometry_yaml_level_2.main(struc_geometry, config=config)

    struc_nodes = rotate_geometry(
        struc_nodes,
        **resolve_initial_geometry_rotation_kwargs(config),
    )

    kite_fem_structure, _, _, _, struc_nodes_initial = (
        structural_kite_fem_level_2.instantiate(
            config=config,
            struc_geometry=struc_geometry,
            struc_nodes=struc_nodes,
            kite_connectivity_arr=kite_connectivity_arr,
            l0_arr=l0_arr,
            k_arr=k_arr,
            c_arr=c_arr,
            m_arr=m_arr,
            linktype_arr=linktype_arr,
            pulley_line_to_other_node_pair_dict=pulley_line_to_other_node_pair_dict,
            canopy_sections=canopy_sections,
            strut_sections=strut_sections,
        )
    )

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
    tether_struct = system_config["components"]["tether"]["structure"]
    tether = RigidLumpedTether(
        diameter=tether_struct["diameter"],
        density=tether_struct.get("density", 970.0),
    )
    mass_wing = float(np.sum(m_arr))
    print(f"Total mass of the wing: {mass_wing:.3f} kg")
    system_model = build_system_model(system_config_path, tether, mass_wing, config)

    ########################################
    ### AEROSTRUCTURAL COUPLED SIMULATION ##
    ########################################
    tracking_data, meta = aerostructural_coupled_solver_level_2.main(
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
        kite_fem_structure=kite_fem_structure,
        canopy_sections=canopy_sections,
        strut_sections=strut_sections,
    )

    save_sim_output(tracking_data, meta, results_dir)
    final_nodes = np.asarray(tracking_data["positions"][meta["n_iter"] - 1])
    save_geometry_snapshot(
        config,
        build_deformed_struc_geometry(struc_geometry, final_nodes),
        build_deformed_aero_geometry(aero_geometry, final_nodes, struc_node_le_indices, struc_node_te_indices),
        results_dir,
    )


if __name__ == "__main__":
    main()

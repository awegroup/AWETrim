"""Run one coupled VSM <-> Billow full-fidelity aerostructural case.

The structure is Billow's full model: cables, frictionless pulleys, inflatable
Timoshenko tube beams and a wrinkling CST membrane canopy, posed as one total
potential energy. :func:`solve_deformation` is the whole pipeline -- config,
geometry, VSM body, Billow model, coupled fixed-point loop -- and writes
``sim_output.h5``, the deformed geometry snapshot and ``case.json``.

Moved here from ``scripts/aerostructural/run_simulation_BILLOW.py`` on
2026-10-07; that script now only sets the inputs and calls this function.

Two things to know before reading a result:

* **The bridle is relaxed first.** The geometry stores measured rest lengths
  against measured node positions and the two disagree (``Br_main_1`` is 11.5%
  long, 88 kN at the real ``EA/l0``). ``structural_billow.instantiate`` settles
  the bridle onto the held wing before the loop starts, so the nodes the loop
  starts from are not the raw YAML nodes.
* **The canopy is a membrane, not a spring net.** The wing's canopy springs are
  replaced by wrinkling triangles on the same grid.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml as _yaml

from awesio.validator import validate as awesio_validate

from awetrim.aerostructural import aerodynamic_vsm
from awetrim.aerostructural.billow import structural_billow
from awetrim.aerostructural.case import (
    DEFAULT_KITE_NAME,
    build_actuation_case_folder,
    build_system_model,
    build_tether,
    resolve_initial_geometry_rotation_kwargs,
    resolve_kite_paths,
)
from awetrim.aerostructural.case_view import build_case_view, save_case_view
from awetrim.aerostructural.coupled import coupled_solver, read_struc_geometry_yaml
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
from awetrim.utils.config_paths import REPO_ROOT

#: Billow needs strut tubes, leading-edge tubes and an inflation pressure, so it
#: reads the full geometry. The reduced PSM files are bridle-and-spring only
#: and cannot feed beams or a canopy mesh.
STRUC_GEOMETRY_FILENAME = "struc_geometry_FEM_full.yaml"

#: Settings this driver applies over the kite's ``as_config.yaml`` before the
#: caller's overrides.
BILLOW_CONFIG_DEFAULTS: dict[str, Any] = {
    "structural_solver": "billow",
    # Conserve the panel pitching moment, not just the panel force: the wing's
    # trim is a moment balance. See aero2struc.weights_at_centre_of_pressure.
    "aero2struc": {"chordwise_distribution": "moment_matched"},
    # Trim the tape lengths the geometry stores unless asked otherwise. The
    # shared as_config ramps the depower and steering tapes, which would turn
    # a single trim into a sequence of actuated states.
    "power_tape_final_extension": 0.0,
    "steering_tape_final_extension": 0.0,
    # The coupled body is built from the STRUCTURAL sections: 27 sections on
    # this geometry, so 2 panels per section = 54 aero panels.
    "aerodynamic": {"n_aero_panels_per_struc_section": 2},
}


def merge_config(base: Mapping[str, Any], overrides: Mapping[str, Any] | None):
    """``base`` with ``overrides`` applied, nested dicts merged key by key."""
    merged = copy.deepcopy(dict(base))
    for key, value in (overrides or {}).items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = merge_config(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def line_tensions_by_element(structure, positions, n_elements: int) -> np.ndarray:
    """Tension [N] of every reader element: cables and pulley ropes, NaN elsewhere.

    ``positions`` must be the configuration the solver RETURNED; a relaxed
    shape is millimetres off equilibrium, which on dyneema is hundreds of
    newtons. Both arms of a pulley read the rope's tension.
    """
    from billow.elements import line_tensions

    tension = np.full(int(n_elements), np.nan)
    for name, rows in (
        (structural_billow.CABLES, structure.cable_row),
        (structural_billow.PULLEYS, structure.pulley_row),
    ):
        values = line_tensions(positions, structure.model.element_set(name))
        mask = rows >= 0
        tension[mask] = np.asarray(values, dtype=float)[rows[mask]]
    return tension


def solve_deformation(
    config_overrides: Mapping[str, Any] | None = None,
    *,
    kite_name: str = DEFAULT_KITE_NAME,
    project_dir=None,
    results_dir=None,
    system_config_path=None,
) -> dict:
    """Run one Billow coupled case and snapshot the result.

    ``config_overrides`` is merged (nested dicts key by key) over the kite's
    ``as_config.yaml`` and :data:`BILLOW_CONFIG_DEFAULTS`: set the flight
    condition (``angle_elevation_deg``, ``angle_azimuth_deg``,
    ``angle_course_deg``, ``distance_radial``, ``speed_radial``,
    ``wind_speed_wind_ref``, ``is_with_gravity``) and the actuation
    (``power_tape_final_extension``, ``steering_tape_final_extension`` [m],
    relative to the tape lengths the geometry stores) there.

    ``system_config_path`` defaults to the kite's ``system.yaml``. Returns the
    results directory, config, solver tracking/meta and the final nodes.
    """
    project_dir = Path(project_dir) if project_dir is not None else REPO_ROOT
    config_path, aero_geometry_path, _ = resolve_kite_paths(project_dir, kite_name)
    struc_geometry_path = project_dir / "data" / kite_name / STRUC_GEOMETRY_FILENAME
    system_config_path = Path(
        system_config_path or project_dir / "data" / kite_name / "system.yaml"
    )
    with system_config_path.open("r", encoding="utf-8") as handle:
        system_config = _yaml.safe_load(handle)
    awesio_validate(system_config, restrictive=False)

    config = merge_config(
        merge_config(load_yaml(config_path), BILLOW_CONFIG_DEFAULTS), config_overrides
    )
    cp_rel = config.get("aero2struc", {}).get("cp_distribution_path")
    if cp_rel and not Path(cp_rel).is_absolute():
        config["aero2struc"]["cp_distribution_path"] = str(project_dir / cp_rel)

    results_dir = Path(
        results_dir
        or aerostructural_results_root(project_dir, kite_name)
        / "billow"
        / build_actuation_case_folder(config)
    )
    struc_geometry = load_yaml(struc_geometry_path)
    aero_geometry = load_yaml(aero_geometry_path)
    results_dir = save_input_snapshot(config=config, results_dir=results_dir)

    # -- aerodynamics -------------------------------------------------------
    n_struc_ribs = len(struc_geometry["wing_particles"]["data"]) / 2
    n_panels_aero = (n_struc_ribs - 1) * config["aerodynamic"][
        "n_aero_panels_per_struc_section"
    ]
    bridle_path = (
        struc_geometry_path if config.get("is_with_aero_bridle", False) else None
    )
    body_aero, vsm_solver, vel_app, initial_polar_data = aerodynamic_vsm.initialize(
        aero_geometry_path, config, n_panels_aero, bridle_path=bridle_path
    )

    # -- structure ----------------------------------------------------------
    (
        struc_nodes,
        m_arr,
        struc_node_le_indices,
        struc_node_te_indices,
        power_tape_index,
        steering_tape_indices,
        _pulley_node_indices,
        canopy_sections,
        strut_sections,
        _simplified_bridle_points,
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
    # reference strain are taken in the attitude the run starts from.
    struc_nodes = rotate_geometry(
        struc_nodes, **resolve_initial_geometry_rotation_kwargs(config)
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
    # instantiate() relaxes the bridle and may ADD nodes (quad centres of the
    # cross canopy pattern), so everything downstream -- aero mesh, load
    # mapping, masses, the coupled loop -- starts from the model's nodes.
    struc_nodes = billow_structure.model.nodes.copy()
    m_arr = billow_structure.masses
    # Load the MEMBRANE's grid, not the reader's: with a refined canopy the two
    # differ and mapping onto the coarse one leaves the added nodes unloaded.
    canopy_sections = [list(map(int, row)) for row in billow_structure.fine_grid]
    strut_sections = []
    struc_nodes_initial = struc_nodes.copy()

    aero2struc_mapping = (
        BilinearAeroToStructuralLoadMapper()
        .initialize(
            body_aero.panels, struc_nodes, struc_node_le_indices, struc_node_te_indices
        )
        .panel_corner_map
    )

    # -- actuation ----------------------------------------------------------
    initial_length_power_tape = l0_arr[power_tape_index]
    power_tape_extension_step = config["power_tape_extension_step"]
    power_tape_final_extension = config["power_tape_final_extension"]
    n_power_tape_steps = (
        int(abs(power_tape_final_extension / power_tape_extension_step))
        if power_tape_extension_step
        else 0
    )

    # -- system model -------------------------------------------------------
    # The class config["tether"] asks for: the Williams trim refuses any other.
    tether = build_tether(config, system_config)
    print(f"Total structural mass: {float(np.sum(m_arr)):.3f} kg")
    system_model = build_system_model(system_config_path, tether, m_arr, config)

    # -- coupled solve ------------------------------------------------------
    tracking_data, meta = coupled_solver.main(
        m_arr=m_arr,
        struc_nodes=struc_nodes,
        struc_nodes_initial=struc_nodes_initial,
        system_model=system_model,
        config=config,
        initial_length_power_tape=initial_length_power_tape,
        n_power_tape_steps=n_power_tape_steps,
        power_tape_final_extension=power_tape_final_extension,
        power_tape_extension_step=power_tape_extension_step,
        kite_connectivity_arr=kite_connectivity_arr,
        bridle_connectivity_arr=bridle_connectivity_arr,
        pulley_line_indices=pulley_line_indices,
        pulley_line_to_other_node_pair_dict=pulley_line_to_other_node_pair_dict,
        struc_node_le_indices=struc_node_le_indices,
        struc_node_te_indices=struc_node_te_indices,
        body_aero=copy.deepcopy(body_aero),
        vsm_solver=copy.deepcopy(vsm_solver),
        vel_app=vel_app,
        initial_polar_data=copy.deepcopy(initial_polar_data),
        bridle_diameter_arr=bridle_diameter_arr,
        aero2struc_mapping=aero2struc_mapping,
        power_tape_index=power_tape_index,
        # Steering is actuated only when the tapes are passed; with a zero
        # target the solver leaves them alone.
        steering_tape_indices=steering_tape_indices,
        billow_structure=billow_structure,
        canopy_sections=canopy_sections,
        strut_sections=strut_sections,
    )

    save_sim_output(tracking_data, meta, results_dir)
    last = int(meta["n_iter"]) - 1
    final_nodes = np.asarray(tracking_data["positions"][last])
    save_geometry_snapshot(
        config,
        build_deformed_struc_geometry(struc_geometry, final_nodes),
        build_deformed_aero_geometry(
            aero_geometry, final_nodes, struc_node_le_indices, struc_node_te_indices
        ),
        results_dir,
        # No system.yaml sync: it derives wing statistics from the PSM
        # geometry's wing_elements, which the full geometry does not have.
    )

    solved = (
        np.asarray(tracking_data["solved_positions"][last])
        if "solved_positions" in tracking_data
        else billow_structure.struc_nodes
    )
    n_elements = len(kite_connectivity_arr)
    is_line = (billow_structure.cable_row >= 0) | (billow_structure.pulley_row >= 0)
    tubes = billow_structure.model.element_set(structural_billow.TUBES)
    save_case_view(
        results_dir,
        build_case_view(
            backend="billow",
            kite_name=kite_name,
            config=config,
            system_config_path=system_config_path,
            struc_geometry=struc_geometry,
            nodes=final_nodes,
            masses=m_arr,
            connectivity=kite_connectivity_arr,
            line_tension=line_tensions_by_element(billow_structure, solved, n_elements),
            le_indices=struc_node_le_indices,
            te_indices=struc_node_te_indices,
            power_tape_index=power_tape_index,
            steering_tape_indices=steering_tape_indices,
            pulley_line_indices=pulley_line_indices,
            rest_lengths=meta.get("rest_lengths"),
            fixed=billow_structure.fixed_node_indices,
            triangles=billow_structure.model.element_set(
                structural_billow.CANOPY
            ).connectivity,
            tubes=tubes.connectivity,
            tube_diameter=[float(law.diameter) for law in billow_structure.tube_laws],
            drawn_lines=is_line,
        ),
    )
    return {
        "results_dir": results_dir,
        "config": config,
        "tracking_data": tracking_data,
        "meta": meta,
        "final_nodes": final_nodes,
        "billow_structure": billow_structure,
        "system_config_path": system_config_path,
    }

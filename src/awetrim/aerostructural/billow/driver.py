"""Run coupled VSM <-> Billow full-fidelity aerostructural cases.

The structure is Billow's full model: cables, frictionless pulleys, inflatable
Timoshenko tube beams and a wrinkling CST membrane canopy, posed as one total
potential energy. :func:`solve_deformation` is the whole pipeline -- config,
geometry, VSM body, Billow model, coupled fixed-point loop -- and writes
``sim_output.h5``, the deformed geometry snapshot and ``case.json``.

It is built from three pieces a continuation reuses directly:

* :func:`build_case` -- everything that does not change between solves (the
  merged config, the geometry files, the VSM body and solver, the tether);
* :func:`prepare_structure` -- one live Billow model with its load mapping and
  its own copy of the VSM objects;
* :func:`solve_step` -- one coupled solve on a prepared structure FROM WHEREVER
  IT STANDS: Billow owns positions, frames and rest lengths across solves, so a
  second call warm-starts from the first one's equilibrium with its actuated
  tapes in place. A chain of steps costs about one tape walk.

Moved here from ``examples/2_aerostructural/run_simulation_BILLOW.py`` on
2026-10-07; that script now only sets the inputs and calls
:func:`solve_deformation`.

Two things to know before reading a result:

* **The bridle is relaxed first.** The geometry stores measured rest lengths
  against measured node positions and the two disagree (``Br_main_1`` is 11.5%
  long, 88 kN at the real ``EA/l0``). ``structural_billow.instantiate`` settles
  the bridle onto the held wing before the loop starts, so the nodes the loop
  starts from are not the raw ``structure.yml`` nodes.
* **The canopy is a membrane, not a spring net.** The wing's canopy springs are
  replaced by wrinkling triangles on the same grid.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml as _yaml

from awesio.validator import validate as awesio_validate

from awetrim.aerostructural import aerodynamic_vsm
from awetrim.aerostructural.billow import structural_billow
from awetrim.aerostructural.case import (
    DEFAULT_KITE_NAME,
    DEFAULT_STRUC_GEOMETRY_FILENAME,
    build_actuation_case_folder,
    build_system_model,
    build_tether,
    resolve_config_paths,
    resolve_initial_geometry_rotation_kwargs,
    resolve_kite_paths,
)
from awetrim.aerostructural.case_view import build_case_view, save_case_view
from awetrim.aerostructural.coupled import coupled_solver
from awetrim.aerostructural.coupled.awesio_structure import Structure, load_structure
from awetrim.aerostructural.mapping import BilinearAeroToStructuralLoadMapper
from awetrim.aerostructural.results import (
    aerostructural_results_root,
    build_deformed_aero_geometry,
    build_deformed_struc_geometry,
    save_geometry_snapshot,
    save_input_snapshot,
    save_sim_output,
)
from awetrim.aerostructural.utils import load_yaml, resolve_structure, rotate_geometry
from awetrim.utils.config_paths import REPO_ROOT

#: Billow needs strut tubes, leading-edge tubes and an inflation pressure, so it
#: reads the full model -- the kite's one structural file (awesIO).
STRUC_GEOMETRY_FILENAME = DEFAULT_STRUC_GEOMETRY_FILENAME

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


@dataclass
class BillowCase:
    """What does not change between coupled solves of one kite and config."""

    kite_name: str
    project_dir: Path
    config: dict
    system_config: dict
    system_config_path: Path
    struc_geometry: Structure
    aero_geometry: dict
    body_aero: Any
    vsm_solver: Any
    vel_app: Any
    initial_polar_data: Any
    tether: Any


@dataclass
class PreparedStructure:
    """One live Billow model, ready to be solved (and solved again).

    ``reader`` keeps ``Structure.resolved``'s arrays by name; ``struc_nodes`` and
    ``m_arr`` are the MODEL's (``instantiate`` relaxes the bridle and may add
    nodes), and ``canopy_sections`` the membrane's grid the load is mapped on.
    The VSM objects are this structure's own copies: the body is mutated onto
    the deformed shape every iteration, so reusing it across calls is simply
    more of the same iteration.
    """

    geometry: Structure
    structure: Any
    reader: dict
    struc_nodes: np.ndarray
    struc_nodes_initial: np.ndarray
    m_arr: np.ndarray
    canopy_sections: list
    strut_sections: list
    aero2struc_mapping: Any
    body_aero: Any
    vsm_solver: Any
    initial_polar_data: Any

    def rest_length(self, element_index: int) -> float:
        """Live rest length [m] of one element, in the reader's ordering."""
        return structural_billow.get_rest_length(self.structure, int(element_index))

    @property
    def power_tape_length(self) -> float:
        """Live rest length [m] of the depower tape."""
        return self.rest_length(self.reader["power_tape_index"])

    @property
    def steering_half_difference(self) -> float:
        """Live steering-tape half-difference [m]: (second - first) / 2, the
        sign convention of ``steering_tape_final_extension`` (+ shortens the
        first tape and lengthens the second)."""
        first, second = self.reader["steering_tape_indices"][:2]
        return 0.5 * (self.rest_length(second) - self.rest_length(first))


_READER_FIELDS = (
    "struc_nodes", "m_arr", "le_indices", "te_indices", "power_tape_index",
    "steering_tape_indices", "pulley_node_indices", "canopy_sections",
    "strut_sections", "simplified_bridle_points", "kite_connectivity",
    "bridle_connectivity", "bridle_diameter", "l0", "k", "c", "linktype",
    "pulley_line_indices", "pulley_line_to_other_node_pair_dict",
)


def build_case(
    config_overrides: Mapping[str, Any] | None = None,
    *,
    kite_name: str = DEFAULT_KITE_NAME,
    project_dir=None,
    system_config_path=None,
) -> BillowCase:
    """Load the kite's files and build the VSM body and the tether, once.

    ``config_overrides`` is merged (nested dicts key by key) over the kite's
    ``as_config.yaml`` and :data:`BILLOW_CONFIG_DEFAULTS` BEFORE the VSM solver
    is built: ``aerodynamic_vsm.initialize`` constructs the solver from the
    config it is handed, so an aerodynamic key set afterwards is silently
    ignored. ``system_config_path`` defaults to the kite's ``system.yaml``.
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
    resolve_config_paths(config, project_dir)

    struc_geometry = load_structure(struc_geometry_path)
    if struc_geometry.is_wireframe:
        raise ValueError(f"{struc_geometry_path} is a wireframe; Billow needs the full model")
    aero_geometry = load_yaml(aero_geometry_path)
    n_struc_ribs = len(struc_geometry.stations)
    n_panels_aero = (n_struc_ribs - 1) * config["aerodynamic"][
        "n_aero_panels_per_struc_section"
    ]
    bridle_path = (
        struc_geometry_path if config.get("is_with_aero_bridle", False) else None
    )
    body_aero, vsm_solver, vel_app, initial_polar_data = aerodynamic_vsm.initialize(
        aero_geometry_path, config, n_panels_aero, bridle_path=bridle_path
    )
    return BillowCase(
        kite_name=kite_name,
        project_dir=project_dir,
        config=config,
        system_config=system_config,
        system_config_path=system_config_path,
        struc_geometry=struc_geometry,
        aero_geometry=aero_geometry,
        body_aero=body_aero,
        vsm_solver=vsm_solver,
        vel_app=vel_app,
        initial_polar_data=initial_polar_data,
        # The class config["tether"] asks for: the Williams trim refuses any other.
        tether=build_tether(config, system_config),
    )


def prepare_structure(case: BillowCase, config: Mapping[str, Any] | None = None):
    """Resolve the geometry and build one Billow model, ready to be solved.

    ``config`` defaults to the case's. The KCU mass is the system file's
    (``utils.resolve_kcu_mass``); the ``Structure`` itself is read-only, so
    every structure shares the case's.
    """
    config = case.config if config is None else config
    geometry = case.struc_geometry
    reader = dict(
        zip(
            _READER_FIELDS,
            resolve_structure(geometry, config=config, system_config=case.system_config),
        )
    )
    # Rotate before the model is built, so the bridle relaxation and every
    # reference strain are taken in the attitude the run starts from.
    struc_nodes = rotate_geometry(
        reader["struc_nodes"], **resolve_initial_geometry_rotation_kwargs(config)
    )
    structure = structural_billow.instantiate(
        config=config,
        structure=geometry,
        struc_nodes=struc_nodes,
        kite_connectivity_arr=reader["kite_connectivity"],
        l0_arr=reader["l0"],
        k_arr=reader["k"],
        c_arr=reader["c"],
        m_arr=reader["m_arr"],
        linktype_arr=reader["linktype"],
        pulley_line_indices=reader["pulley_line_indices"],
        canopy_sections=reader["canopy_sections"],
        strut_sections=reader["strut_sections"],
    )
    # instantiate() relaxes the bridle and may ADD nodes (quad centres of the
    # cross canopy pattern), so everything downstream -- aero mesh, load
    # mapping, masses, the coupled loop -- starts from the model's nodes.
    struc_nodes = structure.model.nodes.copy()
    body_aero = copy.deepcopy(case.body_aero)
    return PreparedStructure(
        geometry=geometry,
        structure=structure,
        reader=reader,
        struc_nodes=struc_nodes,
        struc_nodes_initial=struc_nodes.copy(),
        m_arr=structure.masses,
        # Load the MEMBRANE's grid, not the reader's: with a refined canopy the
        # two differ and mapping onto the coarse one leaves the added nodes
        # unloaded.
        canopy_sections=[list(map(int, row)) for row in structure.fine_grid],
        strut_sections=[],
        aero2struc_mapping=(
            BilinearAeroToStructuralLoadMapper()
            .initialize(
                body_aero.panels,
                struc_nodes,
                reader["le_indices"],
                reader["te_indices"],
            )
            .panel_corner_map
        ),
        body_aero=body_aero,
        vsm_solver=copy.deepcopy(case.vsm_solver),
        initial_polar_data=copy.deepcopy(case.initial_polar_data),
    )


def solve_step(
    case: BillowCase,
    prepared: PreparedStructure,
    config: Mapping[str, Any] | None = None,
    *,
    final_state: dict | None = None,
):
    """One coupled solve on ``prepared``, from wherever it stands now.

    ``config`` (default the case's) sets the flight condition and the
    actuation of THIS solve: the depower tape walks
    ``power_tape_final_extension`` [m] from its current rest length in
    ``power_tape_extension_step`` increments, the steering half-difference
    ``steering_tape_final_extension`` [m] from its current one in
    ``steering_tape_extension_step`` increments; 0 leaves a tape where it is.
    A structure that has been solved starts from its own state (Billow's state
    IS the positions the next minimum-energy solve starts from), a fresh one
    from the model's nodes. ``final_state`` is passed on to
    ``coupled_solver.main`` (the final trimmed VSM body and trim result).

    Returns ``(tracking_data, meta)`` of ``coupled_solver.main``.
    """
    config = case.config if config is None else config
    structure = prepared.structure
    reach = config["power_tape_final_extension"]
    step = config["power_tape_extension_step"]
    n_power_tape_steps = int(abs(reach / step)) if step else 0
    m_arr = prepared.m_arr
    print(f"Total structural mass: {float(np.sum(m_arr)):.3f} kg")
    reader = prepared.reader
    return coupled_solver.main(
        m_arr=m_arr,
        struc_nodes=np.asarray(structure.state.positions, dtype=float).copy(),
        struc_nodes_initial=prepared.struc_nodes_initial,
        system_model=build_system_model(
            case.system_config_path, case.tether, m_arr, config
        ),
        config=config,
        initial_length_power_tape=prepared.power_tape_length,
        n_power_tape_steps=n_power_tape_steps,
        power_tape_final_extension=reach,
        power_tape_extension_step=step,
        kite_connectivity_arr=reader["kite_connectivity"],
        bridle_connectivity_arr=reader["bridle_connectivity"],
        pulley_line_indices=reader["pulley_line_indices"],
        pulley_line_to_other_node_pair_dict=reader[
            "pulley_line_to_other_node_pair_dict"
        ],
        struc_node_le_indices=reader["le_indices"],
        struc_node_te_indices=reader["te_indices"],
        body_aero=prepared.body_aero,
        vsm_solver=prepared.vsm_solver,
        vel_app=case.vel_app,
        initial_polar_data=prepared.initial_polar_data,
        bridle_diameter_arr=reader["bridle_diameter"],
        aero2struc_mapping=prepared.aero2struc_mapping,
        power_tape_index=reader["power_tape_index"],
        # Steering is actuated only when the tapes are passed; with a zero
        # target the solver leaves them alone.
        steering_tape_indices=reader["steering_tape_indices"],
        billow_structure=structure,
        canopy_sections=prepared.canopy_sections,
        strut_sections=prepared.strut_sections,
        final_state=final_state,
    )


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
    case = build_case(
        config_overrides,
        kite_name=kite_name,
        project_dir=project_dir,
        system_config_path=system_config_path,
    )
    config = case.config
    results_dir = Path(
        results_dir
        or aerostructural_results_root(case.project_dir, kite_name)
        / "billow"
        / build_actuation_case_folder(config)
    )
    results_dir = save_input_snapshot(config=config, results_dir=results_dir)

    prepared = prepare_structure(case)
    tracking_data, meta = solve_step(case, prepared)

    save_sim_output(tracking_data, meta, results_dir)
    reader = prepared.reader
    billow_structure = prepared.structure
    last = int(meta["n_iter"]) - 1
    final_nodes = np.asarray(tracking_data["positions"][last])
    save_geometry_snapshot(
        config,
        build_deformed_struc_geometry(
            case.struc_geometry,
            final_nodes,
            # the live tapes, not the input file's: the actuation ran after it
            bridle_rest_lengths=structural_billow.get_rest_lengths(
                billow_structure, reader["kite_connectivity"]
            )[-len(reader["bridle_connectivity"]):],
        ),
        build_deformed_aero_geometry(
            case.aero_geometry, final_nodes, reader["le_indices"], reader["te_indices"]
        ),
        results_dir,
        # No system.yaml sync: it derives wing statistics from the PSM
        # geometry's wing_elements, which the full model does not have.
    )

    solved = (
        np.asarray(tracking_data["solved_positions"][last])
        if "solved_positions" in tracking_data
        else billow_structure.struc_nodes
    )
    n_elements = len(reader["kite_connectivity"])
    is_line = (billow_structure.cable_row >= 0) | (billow_structure.pulley_row >= 0)
    tubes = billow_structure.model.element_set(structural_billow.TUBES)
    save_case_view(
        results_dir,
        build_case_view(
            backend="billow",
            kite_name=kite_name,
            config=config,
            system_config_path=case.system_config_path,
            struc_geometry=case.struc_geometry,
            nodes=final_nodes,
            masses=prepared.m_arr,
            connectivity=reader["kite_connectivity"],
            line_tension=line_tensions_by_element(billow_structure, solved, n_elements),
            le_indices=reader["le_indices"],
            te_indices=reader["te_indices"],
            power_tape_index=reader["power_tape_index"],
            steering_tape_indices=reader["steering_tape_indices"],
            pulley_line_indices=reader["pulley_line_indices"],
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
        "system_config_path": case.system_config_path,
    }

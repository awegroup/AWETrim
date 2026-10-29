"""Run one coupled VSM <-> wireframe (PSM) aerostructural case.

The wireframe is Billow's line-system fidelity: tension-only cables,
frictionless pulleys and the wing as a spring net (see ``structural_wireframe``).
:func:`solve_deformation` is the whole pipeline: resolve the kite's config and
geometry, build the VSM body and the structure, run the coupled fixed-point
loop (which converges the quasi-steady trim against the deforming structure)
and write ``sim_output.h5``, the deformed geometry snapshot and ``case.json``.

Moved here from ``examples/2_aerostructural/run_simulation_PSM.py`` on
2026-10-07; that script now only sets the inputs and calls this function.
"""

import copy
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml as _yaml

from awesio.validator import validate as awesio_validate

from awetrim.aerostructural import aerodynamic_vsm
from awetrim.aerostructural.case import (
    build_actuation_case_folder,
    build_system_model,
    build_tether,
    DEFAULT_KITE_NAME,
    DEFAULT_STRUC_GEOMETRY_FILENAME,
    WIREFRAME_STRUC_GEOMETRY_FILENAME,
    resolve_initial_geometry_rotation_kwargs,
    resolve_kite_paths,
)
from awetrim.aerostructural.case_view import build_case_view, save_case_view
from awetrim.aerostructural.coupled.awesio_structure import load_structure
from awetrim.aerostructural.billowing import panel_bays as panel_bays_by_construction
from awetrim.aerostructural.mapping import (
    BilinearAeroToStructuralLoadMapper,
    StrutLeverAeroToStructuralLoadMapper,
)
from awetrim.aerostructural.results import (
    aerostructural_results_root,
    build_deformed_aero_geometry,
    build_deformed_struc_geometry,
    save_geometry_snapshot,
    save_input_snapshot,
    save_sim_output,
)
from awetrim.aerostructural.utils import load_sim_output, load_yaml, rotate_geometry
from awetrim.aerostructural.wireframe import (
    coupled_solver_qsm,
    structural_geometry_io,
    structural_wireframe,
)
from awetrim.plotting.kite_structure import plot_3d_kite_structure
from awetrim.utils.config_paths import REPO_ROOT
from awetrim.utils.system_config import get_tether


def _resolve_starting_struc_nodes(
    config,
    project_dir,
    kite_name,
    struc_nodes_default,
):
    """
    Optionally override start nodes from a previous simulation result folder.

    Priority:
      1) config["starting_from_sim_subdir"] (new)
      2) config["starting_from_sim_of_date"] (legacy)

    The value is treated as a subdir under results/<kite_name>/, e.g.
    depower_p0100mm_steer_m0020mm/run_003.

    If both keys are empty, return struc_nodes_default.
    """
    sim_subdir = str(config.get("starting_from_sim_subdir", "")).strip()
    if sim_subdir == "":
        sim_subdir = str(config.get("starting_from_sim_of_date", "")).strip()

    if sim_subdir == "":
        return struc_nodes_default

    base_results_dir = Path(project_dir) / "results" / kite_name

    # Candidate 1: exact path from config
    candidates = [base_results_dir / sim_subdir]

    # Candidate 2/3: tolerate zero-sign naming mismatch, e.g. m0000mm vs p0000mm
    sim_subdir_m_to_p = sim_subdir.replace("m0000mm", "p0000mm")
    sim_subdir_p_to_m = sim_subdir.replace("p0000mm", "m0000mm")
    if sim_subdir_m_to_p != sim_subdir:
        candidates.append(base_results_dir / sim_subdir_m_to_p)
    if sim_subdir_p_to_m != sim_subdir:
        candidates.append(base_results_dir / sim_subdir_p_to_m)

    start_dir = None
    for cand in candidates:
        if cand.exists() and cand.is_dir():
            start_dir = cand
            break

    if start_dir is None:
        raise FileNotFoundError(
            "Configured starting simulation directory does not exist. "
            f"Tried: {', '.join(str(c) for c in candidates)}"
        )

    # Preferred: direct case-folder storage (sim_output.h5 inside start_dir).
    h5_path = start_dir / "sim_output.h5"
    # Backward compatibility: if not found, try legacy run_XXX layout.
    if not h5_path.exists():
        run_dirs = [
            d
            for d in start_dir.iterdir()
            if d.is_dir() and d.name.startswith("run_") and d.name[4:].isdigit()
        ]
        if len(run_dirs) > 0:
            start_dir = sorted(run_dirs, key=lambda p: int(p.name[4:]))[-1]
            logging.info(
                f"Using latest legacy run folder inside case folder: {start_dir.name}"
            )
            h5_path = start_dir / "sim_output.h5"

    if not h5_path.exists():
        raise FileNotFoundError(
            f"Configured starting simulation has no sim_output.h5: {h5_path}"
        )

    _, tracking_data = load_sim_output(h5_path)
    if "positions" not in tracking_data:
        raise KeyError(f"Expected 'positions' dataset in: {h5_path}")

    positions = np.asarray(tracking_data["positions"])
    if positions.ndim != 3 or positions.shape[2] != 3:
        raise ValueError(
            f"Invalid positions shape in {h5_path}: {positions.shape}. Expected (nt, n_nodes, 3)."
        )

    struc_nodes_loaded = np.array(positions[-1], dtype=float)
    if struc_nodes_loaded.shape != np.asarray(struc_nodes_default).shape:
        raise ValueError(
            "Loaded node shape does not match current geometry. "
            f"loaded={struc_nodes_loaded.shape}, current={np.asarray(struc_nodes_default).shape}"
        )

    logging.info(
        f"Starting from previous simulation final nodes: {start_dir} (n_nodes={len(struc_nodes_loaded)})"
    )
    return struc_nodes_loaded


def _resolve_starting_rest_lengths(
    config,
    project_dir,
    kite_name,
    l0_arr_default,
):
    """
    Optionally load final rest_lengths from a previous simulation result.

    If config["starting_from_sim_subdir"] is set, load the rest_lengths from the
    corresponding H5 file. Otherwise return l0_arr_default.

    Args:
        config: Configuration dictionary
        project_dir: Path to project root
        kite_name: Name of the kite
        l0_arr_default: Default rest length array from YAML geometry

    Returns:
        np.ndarray: Updated rest lengths (or defaults if not recovering)
    """
    sim_subdir = str(config.get("starting_from_sim_subdir", "")).strip()
    if sim_subdir == "":
        sim_subdir = str(config.get("starting_from_sim_of_date", "")).strip()

    if sim_subdir == "":
        return l0_arr_default

    base_results_dir = Path(project_dir) / "results" / kite_name

    # Candidate 1: exact path from config
    candidates = [base_results_dir / sim_subdir]

    # Candidate 2/3: tolerate zero-sign naming mismatch
    sim_subdir_m_to_p = sim_subdir.replace("m0000mm", "p0000mm")
    sim_subdir_p_to_m = sim_subdir.replace("p0000mm", "m0000mm")
    if sim_subdir_m_to_p != sim_subdir:
        candidates.append(base_results_dir / sim_subdir_m_to_p)
    if sim_subdir_p_to_m != sim_subdir:
        candidates.append(base_results_dir / sim_subdir_p_to_m)

    start_dir = None
    for cand in candidates:
        if cand.exists() and cand.is_dir():
            start_dir = cand
            break

    if start_dir is None:
        # No previous sim found, return defaults
        return l0_arr_default

    # Try to find sim_output.h5
    h5_path = start_dir / "sim_output.h5"
    if not h5_path.exists():
        run_dirs = [
            d
            for d in start_dir.iterdir()
            if d.is_dir() and d.name.startswith("run_") and d.name[4:].isdigit()
        ]
        if len(run_dirs) > 0:
            start_dir = sorted(run_dirs, key=lambda p: int(p.name[4:]))[-1]
            h5_path = start_dir / "sim_output.h5"

    if not h5_path.exists():
        logging.warning(
            f"No sim_output.h5 found in {start_dir}, using default rest lengths"
        )
        return l0_arr_default

    # Load rest_lengths from metadata
    try:
        metadata, _ = load_sim_output(h5_path)
        if "rest_lengths" in metadata:
            rest_lengths_loaded = np.asarray(metadata["rest_lengths"], dtype=float)
            if rest_lengths_loaded.shape == np.asarray(l0_arr_default).shape:
                logging.info(
                    f"Loaded final rest lengths from previous simulation: {start_dir.name}"
                )
                return rest_lengths_loaded
            else:
                logging.warning(
                    f"Loaded rest_lengths shape {rest_lengths_loaded.shape} "
                    f"does not match current geometry {np.asarray(l0_arr_default).shape}, "
                    f"using defaults"
                )
                return l0_arr_default
        else:
            logging.warning(
                f"No 'rest_lengths' in {h5_path} metadata, using default rest lengths"
            )
            return l0_arr_default
    except Exception as e:
        logging.warning(
            f"Error loading rest_lengths from {h5_path}: {e}, using defaults"
        )
        return l0_arr_default


def _deformed_aero_geometry(aero_geometry, nodes, le_indices, te_indices, mapper):
    """The deformed aero geometry, at the skin's sections when there is one."""
    if mapper is None:
        return build_deformed_aero_geometry(aero_geometry, nodes, le_indices, te_indices)
    edges = mapper.map(nodes)
    step = max(int(mapper.n_panels_per_section), 1)
    le, te = edges.leading_edge_points[::step], edges.trailing_edge_points[::step]
    stacked = np.vstack([nodes, le, te])
    n = len(nodes)
    return build_deformed_aero_geometry(
        aero_geometry, stacked, list(range(n, n + len(le))),
        list(range(n + len(le), n + len(le) + len(te))),
    )


_READER_FIELDS = (
    "struc_nodes", "m_arr", "le_indices", "te_indices", "power_tape_index",
    "steering_tape_indices", "pulley_node_indices", "kite_connectivity",
    "bridle_connectivity", "bridle_diameter", "l0", "k", "c", "linktype",
    "pulley_line_indices", "pulley_line_to_other_node_pair_dict",
)


@dataclass
class WireframeCase:
    """What does not change between coupled solves of one kite and config."""

    kite_name: str
    project_dir: Path
    config: dict
    system_config: dict
    system_config_path: Path
    struc_geometry_path: Path
    #: The wireframe as the PSM-format dict ``structural_geometry_io`` reads
    #: (``structure.wireframe_dict()``), an in-memory intermediate.
    struc_geometry: dict
    aero_geometry: dict
    body_aero: Any
    vsm_solver: Any
    vel_app: Any
    initial_polar_data: Any
    tether: Any
    structural_to_aero: Any = None
    #: Bay (rib pair) of every aerodynamic panel, by construction; None = the
    #: PSM convention, ``n_aero_panels_per_struc_section`` panels per rib bay.
    #: Used by the strut-lever load transfer.
    panel_bays: Any = None
    #: The wireframe as read: an awesIO ``Structure`` (bridle lines for VSM,
    #: snapshots, the case view).
    structure: Any = None


@dataclass
class PreparedWireframe:
    """One live wireframe structure, ready to be solved (and solved again).

    ``reader`` keeps the geometry reader's arrays by name, with ``l0`` the
    rest lengths the structure was built with. The VSM objects are this
    structure's own copies, carried across solves like the structure itself.
    """

    psystem: Any
    reader: dict
    struc_nodes_initial: np.ndarray
    m_arr: np.ndarray
    aero2struc_mapping: Any
    body_aero: Any
    vsm_solver: Any
    initial_polar_data: Any

    def rest_length(self, element_index: int) -> float:
        """Live rest length [m] of one element (pulley arms split)."""
        return float(structural_wireframe.get_rest_lengths(self.psystem)[int(element_index)])

    @property
    def power_tape_length(self) -> float:
        """Live rest length [m] of the depower tape."""
        return self.rest_length(self.reader["power_tape_index"])

    @property
    def steering_half_difference(self) -> float:
        """Live steering-tape half-difference [m] about the built lengths
        (+ shortens the first tape and lengthens the second)."""
        first, second = self.reader["steering_tape_indices"][:2]
        return 0.5 * (
            (self.reader["l0"][first] - self.rest_length(first))
            + (self.rest_length(second) - self.reader["l0"][second])
        )

    def pin_bridle_modulus(self, factor: float) -> None:
        """Every bridle line (tapes included) at ``factor`` x its own
        stiffness, the wing at its own: the fixed-stiffness model. Pair it
        with ``structural_pss.update_stiffness: false`` so the elongation
        bound does not ratchet the stiffness from solve to solve."""
        k = np.asarray(structural_wireframe.get_stiffnesses(self.psystem), dtype=float).copy()
        n_bridle = len(self.reader["bridle_connectivity"])
        k[len(k) - n_bridle:] *= float(factor)
        structural_wireframe.set_stiffnesses(self.psystem, k)

    def restore(self, sim_output_path) -> None:
        """Put the structure back where a saved solve ended: its final node
        positions, rest lengths and the stiffnesses it ran with.

        The structure IS the wireframe's whole state (a minimum-energy solve
        carries no velocity), so a continuation resumes from a saved step as
        it would have gone on. Unlike ``starting_from_sim_subdir``, the built
        rest lengths in ``reader["l0"]`` are kept: they are the reference the
        steering half-difference is measured from.
        """
        metadata, tracking = load_sim_output(sim_output_path)
        # Older files lack structure_positions; their positions[-1] carries the
        # final trim rotation (2 cm at the LEI-V3 tips), so a chain resumed
        # from one starts off the uninterrupted one. It converges to the same
        # state (1e-6 deg AoA), but not bit-identically as it does from
        # structure_positions.
        positions = np.asarray(
            metadata.get("structure_positions", np.asarray(tracking["positions"])[-1]),
            dtype=float,
        )
        rest_lengths = np.asarray(metadata["rest_lengths"], dtype=float)
        current = np.asarray(structural_wireframe.get_rest_lengths(self.psystem), dtype=float)
        if positions.shape != self.psystem.positions.shape or rest_lengths.shape != current.shape:
            raise ValueError(
                f"{sim_output_path} does not fit this structure: positions "
                f"{positions.shape} vs {self.psystem.positions.shape}, rest lengths "
                f"{rest_lengths.shape} vs {current.shape}"
            )
        for particle, position in zip(self.psystem.particles, positions):
            particle.update_pos(position)
        for index in np.flatnonzero(rest_lengths != current):
            self.psystem.update_rest_length(int(index), rest_lengths[index] - current[index])
        if "final_stiffnesses" in metadata:
            structural_wireframe.set_stiffnesses(
                self.psystem, np.asarray(metadata["final_stiffnesses"], dtype=float)
            )


def build_case(
    config: dict | None = None,
    *,
    config_overrides: dict | None = None,
    kite_name: str = DEFAULT_KITE_NAME,
    project_dir=None,
    tether_diameter: float | None = None,
    system_config_path=None,
    struc_geometry_path=None,
    structural_to_aero=None,
    n_panels_aero=None,
    panel_bays=None,
) -> WireframeCase:
    """Load the kite's files and build the VSM body and the tether, once.

    ``struc_geometry_path`` is a wireframe structure file: awesIO (the
    reduction :func:`build_case_reduced_fem` writes, or a deformed snapshot),
    or a retired PSM-format file, converted on the fly by ``load_structure``.
    Without it the wireframe is reduced from the kite's full structure
    (:func:`build_case_reduced_fem`, which takes ``config_overrides`` key by
    key and ``system.yaml`` as the default system file).

    ``config`` defaults to the kite's ``as_config.yaml``; ``config_overrides``
    is merged over it at the TOP level only (a nested block replaces the
    file's whole block -- pass a complete ``config`` for nested changes).
    ``system_config_path`` defaults to ``system_flown_2019.yaml``.
    ``tether_diameter`` overrides the system file's.
    """
    if struc_geometry_path is None:
        if structural_to_aero is not None or n_panels_aero is not None or panel_bays is not None:
            raise ValueError(
                "structural_to_aero / n_panels_aero / panel_bays need a struc_geometry_path; "
                "the FEM reduction sets its own"
            )
        return build_case_reduced_fem(
            config_overrides,
            config=config,
            kite_name=kite_name,
            project_dir=project_dir,
            system_config_path=system_config_path,
            tether_diameter=tether_diameter,
        )
    project_dir = Path(project_dir) if project_dir is not None else REPO_ROOT
    config_path, aero_geometry_path, _ = resolve_kite_paths(project_dir, kite_name)
    struc_geometry_path = Path(struc_geometry_path)

    # Load and validate the awesIO system config (single source of truth for physical params)
    # ``system_config_path`` lets a caller drive the deformation with a variant
    # of the kite's system.yaml -- e.g. system_flown_2019.yaml, whose KCU mass is the
    # as-flown 22.75 kg rather than the 8.4 kg optimisation value. The KCU mass
    # is read from here (see structural_geometry_io._resolve_kcu_mass) and
    # again by build_system_model, so overriding the path is the only way
    # to keep the structural cloud and the coupled QSM trim on the same mass.
    system_config_path = (
        Path(system_config_path)
        if system_config_path is not None
        else Path(project_dir) / "data" / kite_name / "system_flown_2019.yaml"
    )
    with system_config_path.open("r", encoding="utf-8") as _f:
        system_config = _yaml.safe_load(_f)
    awesio_validate(system_config, restrictive=False)

    if config is None:
        config = load_yaml(config_path)
    if config_overrides:
        config = {**config, **config_overrides}
    structure = load_structure(struc_geometry_path)
    if not structure.is_wireframe:
        raise ValueError(
            f"{struc_geometry_path} is a full structure (it has tubes); the wireframe "
            "driver reduces one with build_case_reduced_fem"
        )
    struc_geometry = structure.wireframe_dict()
    aero_geometry = load_yaml(aero_geometry_path)

    n_wing_struc_nodes = len(struc_geometry["wing_particles"]["data"])
    n_struc_ribs = n_wing_struc_nodes / 2
    if n_panels_aero is None:
        n_panels_aero = (n_struc_ribs - 1) * config["aerodynamic"][
            "n_aero_panels_per_struc_section"
        ]
    bridle_path = structure if config.get("is_with_aero_bridle", False) else None
    body_aero, vsm_solver, vel_app, initial_polar_data = aerodynamic_vsm.initialize(
        aero_geometry_path,
        config,
        n_panels_aero,
        bridle_path=bridle_path,
    )
    # ``tether_diameter`` lets a caller analyse the deformation with the same
    # tether it will use downstream; without it the deformation silently runs on
    # system.yaml's diameter while the caller's trim uses another one, and the
    # shape is then produced under loads that do not match the analysis.
    tether = build_tether(config, system_config, diameter=tether_diameter)
    return WireframeCase(
        kite_name=kite_name,
        project_dir=Path(project_dir),
        config=config,
        system_config=system_config,
        system_config_path=system_config_path,
        struc_geometry_path=struc_geometry_path,
        struc_geometry=struc_geometry,
        structure=structure,
        aero_geometry=aero_geometry,
        body_aero=body_aero,
        vsm_solver=vsm_solver,
        vel_app=vel_app,
        initial_polar_data=initial_polar_data,
        tether=tether,
        structural_to_aero=structural_to_aero,
        panel_bays=panel_bays,
    )


def prepare_structure(case: WireframeCase, config: dict | None = None) -> PreparedWireframe:
    """Read the geometry and build one wireframe structure, ready to be solved.

    ``config`` defaults to the case's. Honours the restart keys
    (``starting_from_sim_subdir``) for the start nodes and rest lengths.
    """
    config = case.config if config is None else config
    reader = dict(
        zip(
            _READER_FIELDS,
            structural_geometry_io.main(
                case.struc_geometry, config=config, system_config=case.system_config
            ),
        )
    )

    #####################################################
    ### rotating the initial geometry by some angle,
    ### to enable the wind to be horizontal
    #####################################################
    struc_nodes = rotate_geometry(
        reader["struc_nodes"],
        **resolve_initial_geometry_rotation_kwargs(config),
    )
    struc_nodes = _resolve_starting_struc_nodes(
        config=config,
        project_dir=case.project_dir,
        kite_name=case.kite_name,
        struc_nodes_default=struc_nodes,
    )
    # Also recover the final rest_lengths (element l0 values) from previous simulation if available
    reader["l0"] = _resolve_starting_rest_lengths(
        config=config,
        project_dir=case.project_dir,
        kite_name=case.kite_name,
        l0_arr_default=reader["l0"],
    )
    m_arr = reader["m_arr"]

    # logging initial conditions
    logging.info(f"\n\nINITIAL CONDITIONS, NODES \n")
    for idx, (node_i, m_i) in enumerate(zip(struc_nodes, m_arr)):
        logging.info(f"node_idx: {idx}: node: {node_i}, mass: {m_i}")

    logging.info(f"\n\nINITIAL CONDITIONS, ELEMENTS \n")
    for idx, conn in enumerate(reader["kite_connectivity"]):
        logging.info(
            f"conn_idx: {idx}: conn: {conn}, l0: {reader['l0'][idx]}, k: {reader['k'][idx]}, c: {reader['c'][idx]}, linktype: {reader['linktype'][idx]}"
        )

    psystem, _pss_initial_conditions, _pss_params, struc_nodes_initial = (
        structural_wireframe.instantiate(
            config,
            struc_nodes,
            m_arr,
            reader["kite_connectivity"],
            reader["l0"],
            reader["k"],
            reader["c"],
            reader["linktype"],
            reader["pulley_line_to_other_node_pair_dict"],
        )
    )
    if config["is_with_initial_structure_plot"]:
        plot_3d_kite_structure(
            struc_nodes,
            reader["kite_connectivity"],
            reader["power_tape_index"],
            k_arr=reader["k"],
            c_arr=reader["c"],
            linktype_arr=reader["linktype"],
            pulley_nodes=reader["pulley_node_indices"],
        )

    ##################
    ### AERO2STRUC ###
    ##################
    if coupled_solver_qsm.load_transfer_of(config) == "strut_lever":
        bays = case.panel_bays
        if bays is None:
            # the PSM convention: every rib is a strut, n panels per rib bay
            bays = panel_bays_by_construction(
                len(case.body_aero.panels),
                config["aerodynamic"]["n_aero_panels_per_struc_section"],
                range(len(reader["le_indices"])),
            )
        aero2struc_mapping = (
            StrutLeverAeroToStructuralLoadMapper()
            .initialize(bays, reader["le_indices"], reader["te_indices"])
            .panel_corner_map
        )
    else:
        aero2struc_mapping = (
            BilinearAeroToStructuralLoadMapper()
            .initialize(
                case.body_aero.panels,
                struc_nodes,
                reader["le_indices"],
                reader["te_indices"],
            )
            .panel_corner_map
        )
    return PreparedWireframe(
        psystem=psystem,
        reader=reader,
        struc_nodes_initial=struc_nodes_initial,
        m_arr=m_arr,
        aero2struc_mapping=aero2struc_mapping,
        body_aero=copy.deepcopy(case.body_aero),
        vsm_solver=copy.deepcopy(case.vsm_solver),
        initial_polar_data=copy.deepcopy(case.initial_polar_data),
    )


def solve_step(
    case: WireframeCase,
    prepared: PreparedWireframe,
    config: dict | None = None,
    *,
    final_state: dict | None = None,
):
    """One coupled solve on ``prepared``, from wherever it stands now.

    ``config`` (default the case's) sets the flight condition and the
    actuation of THIS solve: the depower tape walks
    ``power_tape_final_extension`` [m] from its current rest length in
    ``power_tape_extension_step`` increments, the steering half-difference
    ``steering_tape_final_extension`` [m] from the current tape lengths in
    ``steering_tape_extension_step`` increments (0 = no steering walk).
    ``final_state`` is passed on to ``coupled_solver_qsm.main`` and also
    receives the ``system_model`` the solve was trimmed with.

    Returns ``(tracking_data, meta)`` of ``coupled_solver_qsm.main``.
    """
    config = case.config if config is None else config
    reader = prepared.reader
    psystem = prepared.psystem
    power_tape_index = reader["power_tape_index"]
    steering_tape_indices = reader["steering_tape_indices"]

    #################
    ### ACTUATION ###
    #################
    initial_length_power_tape = prepared.power_tape_length
    power_tape_extension_step = config["power_tape_extension_step"]
    power_tape_final_extension = config["power_tape_final_extension"]
    if power_tape_extension_step != 0:
        n_power_tape_steps = int(power_tape_final_extension / power_tape_extension_step)
    else:
        n_power_tape_steps = 0
    logging.info(f"Initial depower tape length: {initial_length_power_tape:.3f}m")
    logging.info(
        f"Desired depower tape length: {initial_length_power_tape + power_tape_final_extension:.3f}m"
    )

    initial_length_steering_left = prepared.rest_length(steering_tape_indices[0])
    initial_length_steering_right = prepared.rest_length(steering_tape_indices[1])
    steering_tape_extension_step = config["steering_tape_extension_step"]
    steering_tape_final_extension = config["steering_tape_final_extension"]
    logging.info(
        f"Initial steering tape lengths: left={initial_length_steering_left:.3f}m, "
        f"right={initial_length_steering_right:.3f}m"
    )
    logging.info(
        f"Desired steering extension target: {steering_tape_final_extension:.3f}m "
        f"with internal step {steering_tape_extension_step:.3f}m"
    )

    ########################################
    # AWETRIM SYSTEM MODEL
    ########################################
    tether = case.tether
    _tether_cfg = config.get("tether", {}) or {}
    _tether_d = tether.diameter_tether
    _tether_rho = get_tether(case.system_config)["structure"].get("density", 970.0)
    logging.info(
        "Tether model for the deformation: %s (d=%.4f m, rho=%.1f kg/m3), "
        "include_in_trim=%s",
        type(tether).__name__,
        _tether_d,
        _tether_rho,
        bool(_tether_cfg.get("include_in_trim", False)),
    )
    m_arr = prepared.m_arr
    mass_total = float(np.sum(m_arr))
    print(f"Total structural mass (sum of particle masses): {mass_total:.3f} kg")
    system_model = build_system_model(case.system_config_path, tether, m_arr, config)
    # Report the tether the system model ACTUALLY ends up with. The factory
    # inside build_system_model prints the tether it built from system.yaml and
    # is then overwritten by ``tether`` -- so that earlier line is stale and
    # reads as if rigid-lumped were in use even when it is not.
    print(
        f"  -> system model tether in use: {type(system_model.tether).__name__} "
        f"(diameter={system_model.tether.diameter_tether:g}, "
        f"in trim={bool(_tether_cfg.get('include_in_trim', False))})"
    )

    ########################################
    ### AEROSTUCTURAL COUPLED SIMULATION ###
    ########################################
    tracking_data, meta = coupled_solver_qsm.main(
        m_arr=m_arr,
        struc_nodes=np.asarray(psystem.positions, dtype=float).copy(),
        struc_nodes_initial=prepared.struc_nodes_initial,
        system_model=system_model,
        config=config,
        ### ACTUATION
        initial_length_power_tape=initial_length_power_tape,
        n_power_tape_steps=n_power_tape_steps,
        power_tape_final_extension=power_tape_final_extension,
        power_tape_extension_step=power_tape_extension_step,
        initial_length_steering_left=initial_length_steering_left,
        initial_length_steering_right=initial_length_steering_right,
        steering_tape_indices=steering_tape_indices,
        steering_tape_final_extension=steering_tape_final_extension,
        steering_tape_extension_step=steering_tape_extension_step,
        ### CONNECTIVITY
        kite_connectivity_arr=reader["kite_connectivity"],
        bridle_connectivity_arr=reader["bridle_connectivity"],
        pulley_line_indices=reader["pulley_line_indices"],
        pulley_line_to_other_node_pair_dict=reader["pulley_line_to_other_node_pair_dict"],
        ### STRUC --> AERO
        struc_node_le_indices=reader["le_indices"],
        struc_node_te_indices=reader["te_indices"],
        ### AERO
        body_aero=prepared.body_aero,
        vsm_solver=prepared.vsm_solver,
        vel_app=case.vel_app,
        initial_polar_data=prepared.initial_polar_data,
        bridle_diameter_arr=reader["bridle_diameter"],
        # Keep the VSM bridle-line drag tracking the deforming/actuated bridle
        # (the initial static bridle mis-trims roll on steered shapes).
        bridle_line_specs=aerodynamic_vsm.parse_bridle_line_specs(
            case.structure, config
        ),
        ### AERO --> STRUC
        aero2struc_mapping=prepared.aero2struc_mapping,
        power_tape_index=power_tape_index,
        ### STRUC
        psystem=psystem,
        structural_to_aero=case.structural_to_aero,
        final_state=final_state,
    )
    if final_state is not None:
        final_state["system_model"] = system_model
    return tracking_data, meta


def solve_deformation(
    config: dict | None = None,
    *,
    config_overrides: dict | None = None,
    kite_name: str = DEFAULT_KITE_NAME,
    project_dir=None,
    results_dir=None,
    tether_diameter: float | None = None,
    system_config_path=None,
    struc_geometry_path=None,
    structural_to_aero=None,
    n_panels_aero=None,
    panel_bays=None,
) -> dict:
    """Run one PSS/QSM aerostructural deformation and snapshot the result.

    This is the whole coupled pipeline: resolve the kite's config/geometry,
    initialise the VSM body and the PSS structure, run the fixed-point
    aero-structural loop (which converges the VSM quasi-steady trim against the
    deforming structure), then write ``sim_output.h5`` and -- when the config
    sets ``is_save_geometry_snapshots`` -- the deformed ``structure.yml``
    / ``aero_geometry.yaml`` / ``system.yaml`` into the results directory.
    It is :func:`build_case`, :func:`prepare_structure` and one
    :func:`solve_step`; a continuation calls :func:`solve_step` again on the
    same prepared structure.

    ``config_overrides`` is merged over the kite's ``config.yaml`` -- use it to
    set the flight condition (``angle_elevation_deg``, ``angle_azimuth_deg``,
    ``angle_course_deg``, ``speed_radial``, ``distance_radial``,
    ``wind_speed_wind_ref``) and ``is_save_geometry_snapshots``.

    ``struc_geometry_path`` is a wireframe structure file (e.g. the reduction
    :func:`solve_deformation_reduced_fem` writes, or a deformed snapshot);
    without it this IS :func:`solve_deformation_reduced_fem` on the kite's
    full structure.
    ``structural_to_aero`` (a ``SkinStructuralToAeroMapper``) and
    ``n_panels_aero`` fly an aerodynamic mesh other than the ribs' own.

    ``system_config_path`` overrides the kite's ``system.yaml`` (default
    ``data/<kite_name>/system_flown_2019.yaml``), e.g. to deform with the
    as-flown KCU mass.

    Returns a dict with the results directory, the deformed nodes, the solver
    tracking data/meta, and the pieces the caller needs for reporting.
    """
    if struc_geometry_path is None:
        if structural_to_aero is not None or n_panels_aero is not None or panel_bays is not None:
            raise ValueError(
                "structural_to_aero / n_panels_aero / panel_bays need a struc_geometry_path; "
                "the FEM reduction sets its own"
            )
        return solve_deformation_reduced_fem(
            config_overrides,
            config=config,
            kite_name=kite_name,
            project_dir=project_dir,
            results_dir=results_dir,
            system_config_path=system_config_path,
            tether_diameter=tether_diameter,
        )
    case = build_case(
        config,
        config_overrides=config_overrides,
        kite_name=kite_name,
        project_dir=project_dir,
        tether_diameter=tether_diameter,
        system_config_path=system_config_path,
        struc_geometry_path=struc_geometry_path,
        structural_to_aero=structural_to_aero,
        n_panels_aero=n_panels_aero,
        panel_bays=panel_bays,
    )
    config = case.config
    case_folder = build_actuation_case_folder(config)
    results_root = aerostructural_results_root(case.project_dir, kite_name)
    results_dir = (
        Path(results_dir) if results_dir is not None else results_root / case_folder
    )
    results_dir = save_input_snapshot(
        config=config,
        results_dir=results_dir,
    )
    logging.info(f"config files saved in {results_dir}\n")

    prepared = prepare_structure(case)
    final_state = {}
    tracking_data, meta = solve_step(case, prepared, final_state=final_state)
    system_model = final_state["system_model"]
    reader = prepared.reader

    # Save results
    h5_path = save_sim_output(tracking_data, meta, results_dir)
    # The final coupled trim's converged circulation, as a plain .npy next to
    # the geometry snapshot: snapshot re-solves seed their gamma loop with it
    # (branch selection near stall) without having to open the h5.
    gamma_final = np.asarray(meta.get("gamma_distribution", []), dtype=float)
    if gamma_final.size:
        np.save(Path(results_dir) / "gamma_distribution.npy", gamma_final)
    final_nodes = np.asarray(tracking_data["positions"][meta["n_iter"] - 1])
    save_geometry_snapshot(
        config,
        build_deformed_struc_geometry(
            case.structure,
            final_nodes,
            # the live tapes, not the input file's: the actuation ran after it
            bridle_rest_lengths=np.asarray(
                structural_wireframe.get_rest_lengths(prepared.psystem)
            )[-len(reader["bridle_connectivity"]):],
        ),
        _deformed_aero_geometry(
            case.aero_geometry, final_nodes, reader["le_indices"],
            reader["te_indices"], structural_to_aero,
        ),
        results_dir,
        system_yaml_path=case.system_config_path,
    )

    save_case_view(
        results_dir,
        build_case_view(
            backend="wireframe",
            kite_name=kite_name,
            config=config,
            system_config_path=case.system_config_path,
            struc_geometry=case.structure,
            nodes=final_nodes,
            masses=prepared.m_arr,
            connectivity=reader["kite_connectivity"],
            line_tension=prepared.psystem.tensions(),
            le_indices=reader["le_indices"],
            te_indices=reader["te_indices"],
            power_tape_index=reader["power_tape_index"],
            steering_tape_indices=reader["steering_tape_indices"],
            pulley_line_indices=reader["pulley_line_indices"],
            rest_lengths=meta.get("rest_lengths"),
            fixed=config.get("structural_pss", {}).get("fixed_point_indices", [0]),
        ),
    )

    return {
        "results_dir": results_dir,
        "results_root": results_root,
        "case_folder": case_folder,
        "config": config,
        "h5_path": h5_path,
        "tracking_data": tracking_data,
        "meta": meta,
        "final_nodes": final_nodes,
        "m_arr": prepared.m_arr,
        "struc_geometry": case.struc_geometry,
        "structure": case.structure,
        "aero_geometry": case.aero_geometry,
        "kite_connectivity_arr": reader["kite_connectivity"],
        "l0_arr": reader["l0"],
        "k_arr": reader["k"],
        "struc_node_le_indices": reader["le_indices"],
        "struc_node_te_indices": reader["te_indices"],
        "power_tape_index": reader["power_tape_index"],
        "steering_tape_indices": reader["steering_tape_indices"],
        "system_model": system_model,
        "system_config_path": case.system_config_path,
    }


def billowing_contraction(config) -> float:
    """``aerodynamic.billowing.contraction``: uniform strut-to-strut TE
    contraction p (fraction), 0 = off."""
    block = (config.get("aerodynamic") or {}).get("billowing") or {}
    return float(block.get("contraction", 0.0))


def build_case_reduced_fem(
    config_overrides: dict | None = None,
    *,
    kite_name: str = DEFAULT_KITE_NAME,
    project_dir=None,
    work_dir=None,
    system_config_path=None,
    fem_geometry_filename: str = DEFAULT_STRUC_GEOMETRY_FILENAME,
    config: dict | None = None,
    tether_diameter: float | None = None,
) -> WireframeCase:
    """The wireframe case REDUCED from the full structure, one input file.

    The same input file as the Billow full model
    (``billow.driver.solve_deformation``: the kite's awesIO ``structure.yml``,
    ``fem_geometry_filename``), so the two differ only in the structural
    model. The full structure is read, its bridle relaxed onto the held wing
    exactly as Billow relaxes it (``structural_billow.relax_bridle_nodes``),
    and reduced to a wireframe (``reduce_fem.reduce_fem_geometry``: one rib
    per strut, two wing nodes per rib, the per-strut line fans collapsed to
    one equivalent line, tube and canopy stiffnesses from Billow's
    materials). The reduction is written as an awesIO wireframe document to
    ``<work_dir>/structure_wireframe.yml``
    (``case.WIREFRAME_STRUC_GEOMETRY_FILENAME``).

    The aerodynamic mesh is the FULL geometry's -- its true leading and
    trailing edges, every section, ``n_aero_panels_per_struc_section`` panels
    per FEM section -- carried on the reduced ribs by a
    ``SkinStructuralToAeroMapper``, so the wireframe and Billow fly the same
    wing (without it the wireframe would fly the PSM convention: a wing whose
    edges ARE the bridle attachment nodes, x/c 0.01 to 0.90, 11 % short of the
    chord).

    ``aerodynamic.billowing.contraction`` p > 0 billows that mesh
    (:mod:`awetrim.aerostructural.billowing`): the struts rotate about their
    leading edge until every inner bay's trailing edge is (1 - p) of the
    built one, and the canopy between them is the circular arc of its fabric
    length. The reduced ribs move with their struts and the wing members take
    the new distances as rest lengths, so the structure holds the contraction
    the aerodynamic mesh is drawn with; the bridle keeps its lengths. The
    refined arc points are carried by the skin directly (one per anchor), so
    the arc survives the deformation.

    ``config_overrides`` is merged key by key (nested dicts too) over the
    kite's ``as_config.yaml`` (or over ``config`` when given);
    ``system_config_path`` defaults to the kite's ``system.yaml``;
    ``tether_diameter`` overrides the system file's.
    """
    from awetrim.aerostructural.billow.driver import merge_config
    from awetrim.aerostructural.billow.structural_billow import relax_bridle_nodes
    from awetrim.aerostructural.billowing import billow_wing, refine_billowed_wing
    from awetrim.aerostructural.coupled.awesio_structure import (
        wireframe_to_awesio,
        write_awesio,
    )
    from awetrim.aerostructural.mapping import SkinStructuralToAeroMapper
    from awetrim.utils.system_config import get_kcu_mass
    from awetrim.aerostructural.wireframe.reduce_fem import reduce_fem_geometry

    project_dir = Path(project_dir) if project_dir is not None else REPO_ROOT
    config_path, _, _ = resolve_kite_paths(project_dir, kite_name)
    system_config_path = Path(
        system_config_path or project_dir / "data" / kite_name / "system.yaml"
    )
    with system_config_path.open("r", encoding="utf-8") as handle:
        system_config = _yaml.safe_load(handle)
    config = merge_config(
        config if config is not None else load_yaml(config_path), config_overrides
    )

    fem_path = project_dir / "data" / kite_name / fem_geometry_filename
    structure = load_structure(fem_path)
    # The KCU mass is not in the structure: system.yaml is its single source.
    reader = structure.resolved(kcu_mass=get_kcu_mass(system_config) or 0.0)
    nodes, masses = reader[0], reader[1]
    relaxed = relax_bridle_nodes(
        config,
        structure,
        nodes,
        reader[10],  # kite_connectivity_arr
        reader[13],  # l0_arr
        reader[14],  # k_arr
        reader[16],  # linktype_arr
        reader[17],  # pulley_line_indices
        reader[7],  # canopy_sections
        reader[8],  # strut_sections
    )
    reduced = reduce_fem_geometry(structure, nodes=relaxed, masses=masses, config=config)

    per_section = int(config["aerodynamic"]["n_aero_panels_per_struc_section"])
    n_ribs = len(reduced["wing_particles"]["data"]) // 2
    relaxed = np.asarray(relaxed, dtype=float)
    fem_le, fem_te = np.asarray(reader[2], dtype=int), np.asarray(reader[3], dtype=int)
    # the FEM section of every reduced rib (both run +y to -y)
    rib_le_node = [
        int(strut["nodes"][0])
        for strut in sorted(structure.struts, key=lambda strut: -relaxed[strut["nodes"][0]][1])
    ]
    rib_sections = [int(np.flatnonzero(fem_le == node)[0]) for node in rib_le_node]
    if rib_sections != sorted(rib_sections) or len(rib_sections) != n_ribs:
        raise ValueError(f"reduced ribs do not follow the FEM sections: {rib_sections}")

    le_points, te_points = relaxed[fem_le], relaxed[fem_te]
    contraction = billowing_contraction(config)
    if contraction > 0.0:
        wing = billow_wing(le_points, te_points, rib_sections, contraction)
        le_points, te_points = refine_billowed_wing(wing, per_section)
        skin_per_section = 1  # the arc points are anchored themselves
        # the ribs move with their struts; the wing members rest at the new shape
        positions = {
            int(row[0]): np.asarray(row[1:4], dtype=float)
            for row in reduced["wing_particles"]["data"]
        }
        for r, section in enumerate(rib_sections):
            for node in (2 * r + 1, 2 * r + 2):
                positions[node] = wing.move_with_section(section, positions[node])
        reduced["wing_particles"]["data"] = [
            [node, *map(float, positions[node])] for node in sorted(positions)
        ]
        ends = {
            row[0]: (int(row[1]), int(row[2])) for row in reduced["wing_connections"]["data"]
        }
        for row in reduced["wing_elements"]["data"]:
            ci, cj = ends[row[0]]
            row[1] = round(float(np.linalg.norm(positions[ci] - positions[cj])), 6)
        reduced["billowing"] = {
            "contraction": float(contraction),
            "bays": [
                {
                    "sections": [bay.first, bay.last],
                    "chord_built": round(float(bay.chord_built), 6),
                    "chord": round(float(bay.chord), 6),
                    "fabric_length": round(float(bay.fabric_length), 6),
                    "billowing": round(float(bay.billowing), 6),
                    "theta_deg": round(float(np.degrees(bay.theta)), 4),
                    "depth": round(float(bay.depth), 6),
                }
                for bay in wing.bays
            ],
        }
        logging.info(
            "billowing p = %.3f: inner bays 1 - c/L_f = %s",
            contraction,
            [round(b["billowing"], 4) for b in reduced["billowing"]["bays"][1:-1]],
        )
    else:
        skin_per_section = per_section

    reduced_nodes = np.zeros((1 + n_ribs * 2 + len(reduced["bridle_particles"]["data"]), 3))
    reduced_nodes[0] = reduced["bridle_point_node"]
    for key in ("wing_particles", "bridle_particles"):
        for row in reduced[key]["data"]:
            reduced_nodes[int(row[0])] = row[1:4]
    rotation = resolve_initial_geometry_rotation_kwargs(config)
    skin = SkinStructuralToAeroMapper().initialize(
        rotate_geometry(reduced_nodes, **rotation),
        front_indices=[2 * r + 1 for r in range(n_ribs)],
        rear_indices=[2 * r + 2 for r in range(n_ribs)],
        le_points=rotate_geometry(le_points, **rotation),
        te_points=rotate_geometry(te_points, **rotation),
        n_panels_per_section=skin_per_section,
    )
    n_panels_aero = (len(fem_le) - 1) * per_section

    work_dir = Path(
        work_dir
        or aerostructural_results_root(project_dir, kite_name)
        / "wireframe"
        / build_actuation_case_folder(config)
    )
    work_dir.mkdir(parents=True, exist_ok=True)
    reduced_path = work_dir / WIREFRAME_STRUC_GEOMETRY_FILENAME
    # written whole and then moved into place: parallel sweep processes build
    # the same case and must never read a half-written file
    partial = reduced_path.with_suffix(f".{os.getpid()}.tmp")
    write_awesio(
        wireframe_to_awesio(reduced, name=f"{kite_name} wireframe"), partial
    )
    os.replace(partial, reduced_path)
    logging.info(
        "wireframe reduced from %s: %d wing + %d bridle nodes, wing %.3f kg, "
        "bridle %.3f kg -> %s",
        fem_path.name,
        len(reduced["wing_particles"]["data"]),
        len(reduced["bridle_particles"]["data"]),
        reduced["reduction"]["wing_mass"],
        reduced["reduction"]["bridle_mass"],
        reduced_path,
    )
    return build_case(
        config,
        kite_name=kite_name,
        project_dir=project_dir,
        system_config_path=system_config_path,
        tether_diameter=tether_diameter,
        struc_geometry_path=reduced_path,
        structural_to_aero=skin,
        n_panels_aero=n_panels_aero,
        panel_bays=panel_bays_by_construction(n_panels_aero, per_section, rib_sections),
    )


def solve_deformation_reduced_fem(
    config_overrides: dict | None = None,
    *,
    kite_name: str = DEFAULT_KITE_NAME,
    project_dir=None,
    results_dir=None,
    system_config_path=None,
    fem_geometry_filename: str = DEFAULT_STRUC_GEOMETRY_FILENAME,
    config: dict | None = None,
    tether_diameter: float | None = None,
) -> dict:
    """One wireframe case on the wireframe REDUCED from the full FEM geometry
    (:func:`build_case_reduced_fem`, the reduction written to
    ``<results_dir>/structure_wireframe.yml``), solved with
    :func:`solve_deformation`.

    ``config_overrides`` is merged key by key (nested dicts too) over the
    kite's ``as_config.yaml`` (or over ``config`` when given);
    ``system_config_path`` defaults to the kite's ``system.yaml``;
    ``tether_diameter`` overrides the system file's.
    """
    from awetrim.aerostructural.billow.driver import merge_config

    project_dir = Path(project_dir) if project_dir is not None else REPO_ROOT
    config_path, _, _ = resolve_kite_paths(project_dir, kite_name)
    config = merge_config(
        config if config is not None else load_yaml(config_path), config_overrides
    )
    results_dir = Path(
        results_dir
        or aerostructural_results_root(project_dir, kite_name)
        / "wireframe"
        / build_actuation_case_folder(config)
    )
    case = build_case_reduced_fem(
        config=config,
        kite_name=kite_name,
        project_dir=project_dir,
        work_dir=results_dir,
        system_config_path=system_config_path,
        fem_geometry_filename=fem_geometry_filename,
        tether_diameter=tether_diameter,
    )
    return solve_deformation(
        config=case.config,
        kite_name=kite_name,
        project_dir=project_dir,
        results_dir=results_dir,
        system_config_path=case.system_config_path,
        tether_diameter=tether_diameter,
        struc_geometry_path=case.struc_geometry_path,
        structural_to_aero=case.structural_to_aero,
        n_panels_aero=len(case.body_aero.panels),
        panel_bays=case.panel_bays,
    )

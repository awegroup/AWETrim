"""Slow integration tests for the PSS/QSM coupled aerostructural solver.

Opt-in: every test here is marked ``slow`` and is deselected from the default
``pytest`` run (``addopts = -m 'not slow'`` in pyproject.toml). Run them with::

    pytest -m slow tests/aerostructural/test_pss_solver_slow.py

These reproduce the real ``run_simulation_PSM`` setup path (geometry -> VSM
init -> coupled solve) for the photogrammetry LEI-V3 geometry at 0.0 m and
0.2 m steering, with gravity and bridle/tether drag enabled, and assert the
solve CONVERGES to a physically sane trim.

By design they assert convergence flags + generous bounds + a monotonic turn
response (more steering -> more turn) rather than exact numeric values: that
matches the repo convention of not pinning solver values, while still catching
the regressions that bit us in practice (the trim-rotation throwaway-copy change
that broke convergence, and the KCU mass double-count that biased the trim).

Needs the VSM and PSS external solvers plus the data/LEI-V3-KITE files; skips
cleanly if any are unavailable.
"""

import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("VSM", reason="Vortex-Step-Method not installed")
pytest.importorskip("PSS", reason="Particle System Simulator not installed")

import yaml as _yaml

from awetrim.aerostructural import aerodynamic_vsm
from awetrim.aerostructural.mapping import BilinearAeroToStructuralLoadMapper
from awetrim.aerostructural.pss import (
    aerostructural_coupled_solver_qsm,
    structural_geometry_io,
    structural_pss,
)
from awetrim.aerostructural.pss.actuation import update_steering_tape_actuation
from awetrim.aerostructural.utils import load_yaml, rotate_geometry
from awetrim.system.tether import RigidLumpedTether

pytestmark = pytest.mark.slow

REPO_ROOT = Path(__file__).resolve().parents[2]
KITE_DIR = REPO_ROOT / "data" / "LEI-V3-KITE"
SYSTEM_CONFIG_PATH = KITE_DIR / "system_flown.yaml"

if not KITE_DIR.exists():
    pytest.skip(f"kite data dir missing: {KITE_DIR}", allow_module_level=True)

# Operating point: steering tape extensions [m] and the powered depower extension.
STEERING_VALUES_M = [0.0, 0.2]
DEPOWER_M = 0.2


def _load_scripts_common():
    """Import scripts/aerostructural/common.py under a unique name (no path clash).

    The integration test deliberately uses the real build_system_model so it
    covers the KCU mass split (mass_kcu = m_arr[0], mass_wing = sum(m_arr[1:])).
    """
    path = REPO_ROOT / "scripts" / "aerostructural" / "common.py"
    spec = importlib.util.spec_from_file_location("aerostructural_scripts_common", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_one(steering_m, setup):
    """Run the coupled solver for one steering extension; return solver meta."""
    common = setup["common"]
    base_config = setup["base_config"]

    cfg = copy.deepcopy(base_config)
    cfg["steering_tape_final_extension"] = float(steering_m)
    cfg["power_tape_final_extension"] = float(DEPOWER_M)
    cfg["is_save_geometry_snapshots"] = False

    struc_nodes = setup["struc_nodes_base"].copy()
    psystem, _, _, struc_nodes_initial = structural_pss.instantiate(
        cfg,
        struc_nodes,
        setup["m_arr"],
        setup["kite_connectivity_arr"],
        setup["l0_arr"],
        setup["k_arr"],
        setup["c_arr"],
        setup["linktype_arr"],
        setup["pulley_line_to_other_node_pair_dict"],
    )

    # Pre-actuate steering tapes to the full extension (matches run_sweep_*_PSM):
    # the solver call below passes no steering args, so no in-loop progressive
    # steering is applied.
    steering_tape_indices = setup["steering_tape_indices"]
    l0_arr = setup["l0_arr"]
    step = float(cfg.get("steering_tape_extension_step", 0.0))
    if abs(float(steering_m)) > 1e-9:
        effective_step = step if step != 0 else float(steering_m)
        update_steering_tape_actuation(
            psystem=psystem,
            steering_tape_indices=steering_tape_indices,
            steering_tape_extension_step=effective_step,
            initial_length_steering_left=l0_arr[steering_tape_indices[0]],
            initial_length_steering_right=l0_arr[steering_tape_indices[1]],
            steering_tape_final_extension=float(steering_m),
        )

    system_model = common.build_system_model(
        SYSTEM_CONFIG_PATH, setup["tether"], setup["m_arr"], cfg
    )

    aero2struc_mapping = (
        BilinearAeroToStructuralLoadMapper()
        .initialize(
            setup["body_aero_init"].panels,
            struc_nodes,
            setup["struc_node_le_indices"],
            setup["struc_node_te_indices"],
        )
        .panel_corner_map
    )

    _, meta = aerostructural_coupled_solver_qsm.main(
        m_arr=setup["m_arr"],
        struc_nodes=struc_nodes,
        struc_nodes_initial=struc_nodes_initial,
        system_model=system_model,
        config=cfg,
        initial_length_power_tape=setup["initial_length_power_tape"],
        n_power_tape_steps=setup["n_power_tape_steps"],
        power_tape_final_extension=DEPOWER_M,
        power_tape_extension_step=setup["depower_step"],
        kite_connectivity_arr=setup["kite_connectivity_arr"],
        bridle_connectivity_arr=setup["bridle_connectivity_arr"],
        pulley_line_indices=setup["pulley_line_indices"],
        pulley_line_to_other_node_pair_dict=setup["pulley_line_to_other_node_pair_dict"],
        struc_node_le_indices=setup["struc_node_le_indices"],
        struc_node_te_indices=setup["struc_node_te_indices"],
        body_aero=copy.deepcopy(setup["body_aero_init"]),
        vsm_solver=copy.deepcopy(setup["vsm_solver_init"]),
        vel_app=setup["vel_app"],
        initial_polar_data=copy.deepcopy(setup["initial_polar_data_init"]),
        bridle_diameter_arr=setup["bridle_diameter_arr"],
        aero2struc_mapping=aero2struc_mapping,
        power_tape_index=setup["power_tape_index"],
        psystem=psystem,
    )
    return meta


@pytest.fixture(scope="module")
def solver_results():
    """Run the coupled solver once per steering value; cache the meta dicts.

    Gravity and bridle/tether drag are enabled here regardless of the as_config
    defaults, per the integration-test intent.
    """
    common = _load_scripts_common()

    config_path, aero_geometry_path, struc_geometry_path = common.resolve_kite_paths(
        REPO_ROOT, "LEI-V3-KITE"
    )

    with SYSTEM_CONFIG_PATH.open("r", encoding="utf-8") as f:
        system_config = _yaml.safe_load(f)

    base_config = load_yaml(config_path)
    base_config["is_with_gravity"] = True
    base_config["is_with_aero_bridle"] = True
    base_config["is_with_aero_tether"] = True

    struc_geometry = load_yaml(struc_geometry_path)

    (
        struc_nodes_base,
        m_arr,
        struc_node_le_indices,
        struc_node_te_indices,
        power_tape_index,
        steering_tape_indices,
        _pulley_node_indices,
        kite_connectivity_arr,
        bridle_connectivity_arr,
        bridle_diameter_arr,
        l0_arr,
        k_arr,
        c_arr,
        linktype_arr,
        pulley_line_indices,
        pulley_line_to_other_node_pair_dict,
    ) = structural_geometry_io.main(
        struc_geometry, config=base_config, system_config=system_config
    )

    struc_nodes_base = rotate_geometry(
        struc_nodes_base, **common.resolve_initial_geometry_rotation_kwargs(base_config)
    )

    depower_step = float(base_config.get("power_tape_extension_step", 0.05))
    n_power_tape_steps = int(DEPOWER_M / depower_step) if depower_step != 0 else 0

    n_wing_struc_nodes = len(struc_geometry["wing_particles"]["data"])
    n_struc_ribs = n_wing_struc_nodes / 2
    n_panels_aero = (n_struc_ribs - 1) * base_config["aerodynamic"][
        "n_aero_panels_per_struc_section"
    ]
    bridle_path = (
        struc_geometry_path if base_config.get("is_with_aero_bridle", False) else None
    )
    body_aero_init, vsm_solver_init, vel_app, initial_polar_data_init = (
        aerodynamic_vsm.initialize(
            aero_geometry_path, base_config, n_panels_aero, bridle_path=bridle_path
        )
    )

    tether_struct = system_config["components"]["tether"]["structure"]
    tether = RigidLumpedTether(
        diameter=tether_struct["diameter"],
        density=tether_struct.get("density", 970.0),
    )

    setup = {
        "common": common,
        "base_config": base_config,
        "struc_nodes_base": struc_nodes_base,
        "m_arr": m_arr,
        "struc_node_le_indices": struc_node_le_indices,
        "struc_node_te_indices": struc_node_te_indices,
        "power_tape_index": power_tape_index,
        "steering_tape_indices": steering_tape_indices,
        "kite_connectivity_arr": kite_connectivity_arr,
        "bridle_connectivity_arr": bridle_connectivity_arr,
        "bridle_diameter_arr": bridle_diameter_arr,
        "l0_arr": l0_arr,
        "k_arr": k_arr,
        "c_arr": c_arr,
        "linktype_arr": linktype_arr,
        "pulley_line_indices": pulley_line_indices,
        "pulley_line_to_other_node_pair_dict": pulley_line_to_other_node_pair_dict,
        "initial_length_power_tape": float(l0_arr[power_tape_index]),
        "n_power_tape_steps": n_power_tape_steps,
        "depower_step": depower_step,
        "body_aero_init": body_aero_init,
        "vsm_solver_init": vsm_solver_init,
        "vel_app": vel_app,
        "initial_polar_data_init": initial_polar_data_init,
        "tether": tether,
    }

    max_iter = int(base_config["aero_structural_solver"]["max_iter"])
    return {
        "metas": {s: _run_one(s, setup) for s in STEERING_VALUES_M},
        "max_iter": max_iter,
    }


@pytest.mark.parametrize("steering", STEERING_VALUES_M)
def test_coupled_solver_converges(solver_results, steering):
    """The coupled solve converges to a sane trim for 0.0 and 0.2 m steering."""
    meta = solver_results["metas"][steering]

    assert meta["converged"] is True, f"steering={steering} m did not converge"
    assert meta["qs_success"] is True, f"steering={steering} m QSM trim failed"
    assert meta["n_iter"] <= solver_results["max_iter"]

    va = float(meta["va"])
    assert np.isfinite(va) and 0.0 < va < 80.0, f"implausible va={va}"

    opt_x = np.asarray(meta["opt_x"], dtype=float)
    assert opt_x.size >= 5 and np.all(np.isfinite(opt_x))
    course_rate = float(opt_x[4])
    assert abs(course_rate) < 5.0, f"implausible course rate {course_rate} rad/s"


def test_more_steering_gives_more_turn(solver_results):
    """0.2 m steering must produce a larger turn rate than 0.0 m (sign of life).

    Catches gross force/mass regressions: the KCU double-count, for instance,
    suppressed the course-rate response to steering.
    """
    cr0 = abs(float(np.asarray(solver_results["metas"][0.0]["opt_x"])[4]))
    cr2 = abs(float(np.asarray(solver_results["metas"][0.2]["opt_x"])[4]))
    assert cr2 > cr0 + 0.05, f"steering did not increase turn rate: {cr0} -> {cr2}"

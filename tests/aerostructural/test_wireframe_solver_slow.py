"""Slow integration tests for the wireframe coupled aerostructural solver.

Opt-in: every test here is marked ``slow`` and is deselected from the default
``pytest`` run (``addopts = -m 'not slow'`` in pyproject.toml). Run them with::

    pytest -m slow -o addopts="" tests/aerostructural/test_wireframe_solver_slow.py

These run the production path, ``wireframe.driver.solve_deformation`` without
a geometry path: the wireframe reduced from the kite's structure.yml with the
bridle relaxed onto the built wing (``build_case_reduced_fem``), as the public
``examples/2_aerostructural/run_simulation_PSM.py`` does. The 2019 as-flown
system, at the built tape and at 0.0 and 0.2 m steering, with gravity and
bridle drag on, must CONVERGE to a physically sane trim.

By design they assert convergence flags + generous bounds + a monotonic turn
response (more steering -> more turn) rather than exact numeric values: that
matches the repo convention of not pinning solver values, while still catching
the regressions that bit us in practice (the trim-rotation throwaway-copy change
that broke convergence, and the KCU mass double-count that biased the trim).
(Until 2026-10-29 the test hand-built the old PSM setup on the photogrammetry
geometry, which is no longer shipped.)

Needs the VSM and Billow external solvers plus the data/LEI-V3-KITE files;
skips cleanly if any are unavailable.
"""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("VSM", reason="Vortex-Step-Method not installed")
pytest.importorskip("billow", reason="Billow structural solver not installed")

pytestmark = pytest.mark.slow

REPO_ROOT = Path(__file__).resolve().parents[2]
KITE_DIR = REPO_ROOT / "data" / "LEI-V3-KITE"
SYSTEM_CONFIG_PATH = KITE_DIR / "system_flown_2019.yaml"

if not KITE_DIR.exists():
    pytest.skip(f"kite data dir missing: {KITE_DIR}", allow_module_level=True)

#: Steering-tape half-differences [m]; the power tape stays at the built length.
STEERING_VALUES_M = [0.0, 0.2]


@pytest.fixture(scope="module")
def solver_results(tmp_path_factory):
    """Run the coupled solver once per steering value; cache the meta dicts."""
    from awetrim.aerostructural.wireframe import driver

    metas = {}
    max_iter = None
    for steering in STEERING_VALUES_M:
        result = driver.solve_deformation(
            config_overrides={
                "is_with_gravity": True,
                "is_with_aero_bridle": True,
                "steering_tape_final_extension": float(steering),
                # the public example's mesh: 2 panels per FEM section
                "aerodynamic": {"n_aero_panels_per_struc_section": 2},
            },
            project_dir=REPO_ROOT,
            results_dir=tmp_path_factory.mktemp(f"steering_{steering:.2f}"),
            system_config_path=SYSTEM_CONFIG_PATH,
        )
        metas[steering] = result["meta"]
        max_iter = int(result["config"]["aero_structural_solver"]["max_iter"])
    return {"metas": metas, "max_iter": max_iter}


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

    # Frozen alpha-sweep anchoring relies on these: the freestream aoa_course
    # (sweep centre) and the apparent-wind magnitude Umag (sweep speed).
    aoa_course = float(meta["aoa_course_deg"])
    umag = float(meta["Umag"])
    assert np.isfinite(aoa_course) and abs(aoa_course) < 30.0
    assert np.isfinite(umag) and 0.0 < umag < 80.0


def test_more_steering_gives_more_turn(solver_results):
    """0.2 m steering must produce a larger turn rate than 0.0 m (sign of life).

    Catches gross force/mass regressions: the KCU double-count, for instance,
    suppressed the course-rate response to steering.
    """
    cr0 = abs(float(np.asarray(solver_results["metas"][0.0]["opt_x"])[4]))
    cr2 = abs(float(np.asarray(solver_results["metas"][0.2]["opt_x"])[4]))
    assert cr2 > cr0 + 0.05, f"steering did not increase turn rate: {cr0} -> {cr2}"

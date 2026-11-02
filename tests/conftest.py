import importlib.util
import sys
from pathlib import Path

import pytest

SRC_PATH = Path(__file__).resolve().parents[1] / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

STEP1_COMMON = Path(__file__).resolve().parents[1] / "examples" / "1_aerodynamics" / "common.py"


def load_step1_common():
    """``examples/1_aerodynamics/common.py`` under its own module name: the
    step-2 examples' helper module is called ``common`` too."""
    name = "aerodynamics_scripts_common"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, STEP1_COMMON)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def lei_v3_case(**changes):
    """The step-1 scripts' default operating point (the step-2 PSM example's), as a Case."""
    from awetrim.aerostructural.case import Case

    values = dict(
        elevation_deg=0.0, azimuth_deg=0.0, course_deg=90.0, distance_radial=300.0,
        speed_radial=1.5, timeder_speed_tangential=0.0, timeder_speed_radial=0.0,
        wind={"model": "uniform", "speed_wind_ref": 6.0},
        power_tape_extension=0.0, steering_tape_extension=0.0,
    )
    values.update(changes)
    return Case(**values)


#: Seed and search box the solver-comparison tests trim with.
STEP1_TRIM_INPUTS = {
    "x_guess": [40.0, 0.0, 0.0, 0.0, 0.0],
    "bounds_lower": [-2.0, -15.0, -15.0, -15.0, -5.0],
    "bounds_upper": [80.0, 15.0, 15.0, 15.0, 5.0],
}


def step1_trim_env(kite) -> dict:
    """The ``body / values / system_model / kcu`` dict the trim tests use."""
    import numpy as np

    values = {k: np.asarray(v, dtype=float) for k, v in STEP1_TRIM_INPUTS.items()}
    values["center_of_gravity"] = kite.center_of_gravity
    values["reference_point"] = kite.reference_point
    return dict(body=kite.body, values=values, system_model=kite.system_model, kcu=kite.kcu_drag)


@pytest.fixture(scope="module")
def lei_v3_step1_kite():
    """The step-1 LEI-V3 kite (``common.load_kite``, built shape) on a coarse
    mesh, 1 panel per FEM section (27), unbillowed: the trim and solver tests
    compare solvers on one body, they need neither the converged mesh nor the
    billowed wing (whose tips sit closer to stall, which the attached-first
    test is not about). One per test module: trims rotate the body."""
    common = load_step1_common()
    return common.load_kite(
        "LEI-V3-KITE",
        # the scripts' default case: at 8 m/s wind the tips sit on their
        # polars' C_l-max corner and the CasADi trim cycles across the kink
        lei_v3_case(),
        overrides={
            "aerodynamics": {
                "n_aero_panels_per_struc_section": 2,
                "billowing": {"contraction": 0.0},
            },
            "coupling": {"is_with_gravity": False},
        },
    )


def pytest_sessionfinish(session, exitstatus):
    """Exit with 0 if no tests were collected.
    
    This hook is called after the entire test run is finished, and is used to
    override the exit status of the test run. If no tests were collected, the
    exit status is set to 0 (indicating success) instead of 5 (indicating no
    tests were collected). Otherwise, the template repository test actionwould 
    be marked as failing even though no tests were run.
    """
    if exitstatus == 5:  # no tests collected
        session.exitstatus = 0

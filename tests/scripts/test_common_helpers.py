"""Unit tests for the cheap, pure helpers in examples/*/common.py.

Scripts are not an installed package, so each ``common.py`` is loaded directly
by file path under a unique module name (three different files share the name
``common``). Only solver-free helpers are exercised: CLI parsing, JSON
coercion, filesystem-tag formatting, config defaulting, and path resolution.
"""

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]


def _load(relpath, name):
    spec = importlib.util.spec_from_file_location(name, REPO / relpath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def aero_common():
    try:
        return _load("examples/1_aerodynamics/common.py", "scripts_aero_common")
    except Exception as exc:  # pragma: no cover - environment guard
        pytest.skip(f"cannot import examples/1_aerodynamics/common.py: {exc}")


@pytest.fixture(scope="module")
def as_common():
    try:
        return _load("examples/2_aerostructural/common.py", "scripts_as_common")
    except Exception as exc:  # pragma: no cover - environment guard
        pytest.skip(f"cannot import examples/2_aerostructural/common.py: {exc}")


# --- csv_vector -------------------------------------------------------------


def test_csv_vector_parses_and_strips(aero_common):
    out = aero_common.csv_vector("1, 2 ,3", length=3, name="x")
    assert isinstance(out, np.ndarray)
    assert out == pytest.approx([1.0, 2.0, 3.0])


def test_csv_vector_wrong_length_raises(aero_common):
    with pytest.raises(argparse.ArgumentTypeError):
        aero_common.csv_vector("1,2", length=3, name="ref")


# --- to_jsonable ------------------------------------------------------------


def test_to_jsonable_ndarray_to_list(aero_common):
    assert aero_common.to_jsonable(np.array([1.0, 2.0])) == [1.0, 2.0]


def test_to_jsonable_numpy_scalar(aero_common):
    out = aero_common.to_jsonable(np.float64(3.5))
    assert out == 3.5
    assert isinstance(out, float)


def test_to_jsonable_complex(aero_common):
    assert aero_common.to_jsonable(1 + 2j) == {"real": 1.0, "imag": 2.0}


def test_to_jsonable_drops_optimizer_key(aero_common):
    out = aero_common.to_jsonable({"a": 1, "optimizer": "ignored", "b": [1, 2]})
    assert out == {"a": 1, "b": [1, 2]}


# --- format_length_tag ------------------------------------------------------


@pytest.mark.parametrize(
    "value_m, expected",
    [
        (0.0, "p0000mm"),
        (0.15, "p0150mm"),
        (-0.15, "m0150mm"),
        (1.5, "p1500mm"),
    ],
)
def test_format_length_tag(as_common, value_m, expected):
    assert as_common.format_length_tag(value_m) == expected


def test_build_actuation_case_folder(as_common):
    config = {"power_tape_final_extension": 0.0, "steering_tape_final_extension": 0.15}
    assert as_common.build_actuation_case_folder(config) == "depower_p0000mm_steer_p0150mm"


# --- resolve_initial_geometry_rotation_kwargs -------------------------------


def test_rotation_kwargs_defaults(as_common):
    kwargs = as_common.resolve_initial_geometry_rotation_kwargs({})
    assert kwargs["angle_deg"] == [0.0, 0.0, 0.0]
    assert kwargs["angle_rad"] is None
    assert kwargs["point"] == [0.0, 0.0, 0.0]
    assert kwargs["axes"] == ["x", "y", "z"]


def test_rotation_kwargs_uses_legacy_scalar(as_common):
    kwargs = as_common.resolve_initial_geometry_rotation_kwargs(
        {"initial_geometry_rotation_deg": 10.0}
    )
    assert kwargs["angle_deg"] == [0.0, 10.0, 0.0]


def test_rotation_kwargs_rejects_both_deg_and_rad(as_common):
    with pytest.raises(ValueError):
        as_common.resolve_initial_geometry_rotation_kwargs(
            {
                "initial_geometry_rotation_angles_deg": [0, 10, 0],
                "initial_geometry_rotation_angles_rad": [0, 0.1, 0],
            }
        )


# --- resolve_kite_paths -----------------------------------------------------


def test_resolve_kite_paths(as_common):
    config_path, aero_path, struc_path = as_common.resolve_kite_paths("/proj", "V3")
    assert config_path.name == "as_config.yaml"
    assert aero_path.name == "aero_geometry.yaml"
    assert struc_path.name == as_common.DEFAULT_STRUC_GEOMETRY_FILENAME
    # All three resolve under data/<kite>/.
    assert config_path.parent.name == "V3"

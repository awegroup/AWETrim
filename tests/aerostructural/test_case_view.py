"""Tests for awetrim.aerostructural.case_view (case.json and the trim table)."""

import json

import numpy as np
import pytest

from awetrim.aerostructural import case_view
from awetrim.aerostructural.coupled.awesio_structure import SCHEMA, Structure


def test_wing_triangles_two_per_bay():
    triangles = case_view.wing_triangles([1, 3, 5], [2, 4, 6])
    assert triangles.shape == (4, 3)
    assert set(triangles.ravel()) == {1, 2, 3, 4, 5, 6}


def test_wing_triangles_rejects_unequal_edges():
    with pytest.raises(ValueError):
        case_view.wing_triangles([1, 3], [2])


def test_classify_lines_kinds_and_names():
    def table(headers, rows):
        return {"headers": headers, "units": ["-"] * len(headers), "data": rows}

    segments = [
        ("main_0", ["0", "7"], "main"), ("power_1", ["0", "8"], "power"),
        ("steer_2", ["8", "9"], "steer"),
        ("pul_3_a", ["7", "1"], "pul"), ("pul_3_b", ["1", "2"], "pul"),
    ]
    geometry = Structure({
        "metadata": {"schema": SCHEMA},
        "points": table(
            ["name", "fixed", "body", "pos_ENU", "extra_mass"],
            [[str(i), i == 0, None, [0.0, 0.0, float(i)], 0.0] for i in range(10)],
        ),
        "wings": table(["name", "pos_ENU", "Q_KA2ENU"], [["wing", [0.0] * 3, [1, 0, 0, 0]]]),
        "spanwise_stations": table(["name", "wing", "points"], [["rib", "wing", ["1", "2"]]]),
        "segments": table(
            ["name", "points", "rest_length", "diameter", "material", "line"],
            [[name, pts, 1.0, 0.002, "dyneema", line] for name, pts, line in segments],
        ),
        "pulleys": table(["name", "segments"], [["pul_3", ["pul_3_a", "pul_3_b"]]]),
    })
    connectivity = [[1, 2], [0, 7], [0, 8], [8, 9], [7, 1], [1, 2]]
    kinds, names = case_view.classify_lines(
        connectivity,
        wing_nodes=[1, 2],
        struc_geometry=geometry,
        power_tape_index=2,
        steering_tape_indices=[3, 99],
        pulley_line_indices=[4],
    )
    assert kinds == ["wing", "bridle", "depower_tape", "steering_tape", "pulley", "wing"]
    assert names[1] == "main" and names[4] == "pul" and names[0] == ""


def test_lift_and_drag_split_along_apparent_wind():
    # drag acts along the apparent wind (air relative to the kite)
    trim = {"total_aero_force_vec": [100.0, 0.0, 1000.0], "va_vel_world": [20.0, 0.0, 0.0]}
    lift, drag = case_view.lift_and_drag(trim)
    assert drag == pytest.approx(100.0)
    assert lift == pytest.approx(1000.0)


def test_lift_and_drag_nan_without_vectors():
    assert all(np.isnan(case_view.lift_and_drag({})))


def test_rigid_motion_removed_is_zero_for_a_rotation():
    rng = np.random.default_rng(0)
    nodes = rng.normal(size=(12, 3))
    angle = 0.4
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0.0],
                         [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]])
    moved = nodes @ rotation.T + np.array([1.0, -2.0, 3.0])
    assert case_view.rigid_motion_removed(nodes, moved).max() < 1e-10


def test_trim_summary_reads_both_meta_layouts():
    nodes = np.zeros((3, 3))
    nodes[1, 0] = 1.0
    base = {"initial": nodes, "final": nodes, "tracking": {}, "config": {"rho": 1.225},
            "view": {"line_kind": ["bridle"], "line_tension": [10.0], "masses": [2.0, 1.0]}}
    trim = {"Umag": 20.0, "aoa_deg": 5.0, "cl": 0.8, "cd": 0.1,
            "total_aero_force_vec": [50.0, 0.0, 500.0], "va_vel_world": [20.0, 0.0, 0.0]}
    flat = dict(base, meta={"trim_results": trim, "opt_x": [19.0, 0, 0, 0, 0],
                            "converged": True, "n_iter": 3})
    nested = dict(base, meta={"trim_results": json.dumps(trim), "speed_tangential": 19.0,
                              "converged": True, "n_iter": 3})
    for case in (flat, nested):
        rows = {label: value for _, label, value, _ in case_view.trim_summary(case)}
        assert rows["Apparent wind speed"] == 20.0
        assert rows["Angle of attack"] == 5.0
        assert rows["Lift-to-drag ratio (VSM force)"] == pytest.approx(10.0)
        assert rows["Total mass"] == 3.0


def test_trim_summary_rom_coefficients_use_the_case_density():
    from awetrim.environment.profile_laws import air_density

    nodes = np.zeros((3, 3))
    config = {"rho": 1.225, "air_density_law": "exponential", "height_gnd": 1500.0,
              "angle_elevation_deg": 30.0, "distance_radial": 200.0}
    trim = {"Umag": 20.0, "total_aero_force_vec": [50.0, 0.0, 500.0],
            "va_vel_world": [20.0, 0.0, 0.0]}
    case = {"initial": nodes, "final": nodes, "tracking": {}, "config": config,
            "view": {"reference_area_rom": 20.0},
            "meta": {"trim_results": trim, "converged": True, "n_iter": 1}}
    rows = {label: value for _, label, value, _ in case_view.trim_summary(case)}
    rho = air_density(200.0 * np.sin(np.radians(30.0)), 1.225, height_gnd=1500.0)
    assert rows["CL (ROM area 20.00 m2)"] == pytest.approx(500.0 / (0.5 * rho * 20.0**2 * 20.0))


def test_format_summary_groups_rows():
    text = case_view.format_summary([("A", "x", 1.0, "m"), ("A", "y", float("nan"), ""),
                                     ("B", "z", "yes", "")])
    assert text.splitlines()[0] == "A" and "n/a" in text and "\nB\n" in text

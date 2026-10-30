"""Unit tests for awetrim.aerostructural.utils

Covers the pure-numpy geometry/mass helpers shared by the coupled solvers:
- rotate_geometry (axis-angle rotations, pivot, legacy scalar angle, validation)
- calculate_cg, calculate_inertia, calculate_moments_of_inertia

Regression note: rotate_geometry must return a new rotated array instead of
mutating the caller-owned input in place. The coupled solver may assign the
returned value back to ``struc_nodes`` explicitly when it needs that behavior.

Per the aerostructural AGENTS.md: this module is numeric numpy only (no CasADi),
so deterministic numeric values are asserted directly.
"""

import numpy as np
import pytest

from awetrim.aerostructural.utils import (
    rotate_geometry,
    rotation_matrix_from_angles,
    calculate_cg,
    calculate_inertia,
    calculate_moments_of_inertia,
)


# ============================================================================
# rotate_geometry
# ============================================================================


class TestRotateGeometryBehaviour:
    def test_zero_rotation_is_identity(self):
        nodes = np.array([[1.0, 2.0, 3.0], [-4.0, 5.0, 6.0]])
        out = rotate_geometry(nodes, angle_deg=[0.0, 0.0, 0.0])
        assert np.allclose(out, nodes)

    def test_does_not_mutate_input(self):
        """Regression: rotation returns a new array and leaves input untouched."""
        nodes = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        original = nodes.copy()
        out = rotate_geometry(nodes, angle_deg=[10.0, 20.0, 30.0])
        assert out is not nodes
        assert np.array_equal(nodes, original)

    def test_rotation_about_z_90deg(self):
        # x,y,z sequential rotations; only the z-angle is non-zero.
        nodes = np.array([[1.0, 0.0, 0.0]])
        out = rotate_geometry(nodes, angle_deg=[0.0, 0.0, 90.0])
        assert np.allclose(out, [[0.0, 1.0, 0.0]], atol=1e-12)

    def test_rotation_preserves_norms(self):
        nodes = np.array([[1.0, 2.0, 3.0], [-2.0, 0.5, 4.0]])
        out = rotate_geometry(nodes, angle_deg=[15.0, -40.0, 70.0])
        assert np.allclose(
            np.linalg.norm(out, axis=1), np.linalg.norm(nodes, axis=1)
        )

    def test_rotation_preserves_pairwise_distance(self):
        nodes = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 2.0]])
        out = rotate_geometry(nodes, angle_deg=[30.0, 60.0, -25.0])
        d_in = np.linalg.norm(nodes[0] - nodes[1])
        d_out = np.linalg.norm(out[0] - out[1])
        assert d_out == pytest.approx(d_in)

    def test_pivot_point_is_fixed(self):
        pivot = (2.0, 3.0, 4.0)
        nodes = np.array([list(pivot), [5.0, 6.0, 7.0]])
        out = rotate_geometry(nodes, angle_deg=[45.0, 10.0, 90.0], point=pivot)
        assert np.allclose(out[0], pivot)

    def test_radians_matches_degrees(self):
        nodes = np.array([[1.0, 2.0, -3.0]])
        deg = rotate_geometry(nodes, angle_deg=[10.0, 20.0, 30.0])
        rad = rotate_geometry(nodes, angle_rad=np.radians([10.0, 20.0, 30.0]))
        assert np.allclose(deg, rad)


class TestRotateGeometryErrors:
    def test_requires_exactly_one_angle_spec_none(self):
        nodes = np.array([[1.0, 0.0, 0.0]])
        with pytest.raises(ValueError, match="exactly one"):
            rotate_geometry(nodes)

    def test_requires_exactly_one_angle_spec_both(self):
        nodes = np.array([[1.0, 0.0, 0.0]])
        with pytest.raises(ValueError, match="exactly one"):
            rotate_geometry(nodes, angle_deg=[0, 0, 0], angle_rad=[0, 0, 0])

    def test_rejects_a_scalar_angle(self):
        nodes = np.array([[1.0, 0.0, 0.0]])
        with pytest.raises(ValueError, match="exactly 3 values"):
            rotate_geometry(nodes, angle_deg=90.0)

    def test_rejects_wrong_node_shape(self):
        nodes = np.array([1.0, 0.0, 0.0])  # 1-D, not (n, 3)
        with pytest.raises(ValueError, match=r"shape \(n_nodes, 3\)"):
            rotate_geometry(nodes, angle_deg=[0, 0, 0])


# ============================================================================
# rotation_matrix_from_angles
# ============================================================================


class TestRotationMatrixFromAngles:
    def test_matches_rotate_geometry(self):
        """The exported matrix must reproduce rotate_geometry exactly: the
        coupled solver uses it to carry the Anderson outer history through
        the per-iteration attitude rotation, and any mismatch de-syncs the
        history from the geometry."""
        nodes = np.array([[1.0, 2.0, 3.0], [-4.0, 5.0, 6.0], [0.5, -0.5, 2.0]])
        angles = [3.1, -2.4, 1.7]
        rotation = rotation_matrix_from_angles(angle_deg=angles)
        assert np.allclose(nodes @ rotation.T, rotate_geometry(nodes, angle_deg=angles))

    def test_matches_rotate_geometry_about_pivot(self):
        nodes = np.array([[1.0, 2.0, 3.0], [-4.0, 5.0, 6.0]])
        angles = [10.0, 20.0, -30.0]
        pivot = np.array([0.3, -0.2, 1.5])
        rotation = rotation_matrix_from_angles(angle_deg=angles)
        assert np.allclose(
            (nodes - pivot) @ rotation.T + pivot,
            rotate_geometry(nodes, angle_deg=angles, point=pivot),
        )

    def test_is_orthonormal(self):
        rotation = rotation_matrix_from_angles(angle_deg=[5.0, -7.0, 12.0])
        assert np.allclose(rotation @ rotation.T, np.eye(3))
        assert np.isclose(np.linalg.det(rotation), 1.0)

    def test_zero_angles_identity(self):
        assert np.allclose(
            rotation_matrix_from_angles(angle_deg=[0.0, 0.0, 0.0]), np.eye(3)
        )


# ============================================================================
# calculate_cg
# ============================================================================


class TestCalculateCg:
    def test_equal_masses_midpoint(self):
        nodes = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        cg = calculate_cg(nodes, m_arr=[1.0, 1.0])
        assert np.allclose(cg, [1.0, 0.0, 0.0])

    def test_mass_weighting(self):
        nodes = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
        cg = calculate_cg(nodes, m_arr=[3.0, 1.0])
        assert cg[0] == pytest.approx(2.5)

    def test_zero_total_mass_raises(self):
        nodes = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        with pytest.raises(ValueError, match="Total mass must be non-zero"):
            calculate_cg(nodes, m_arr=[0.0, 0.0])

    def test_mass_length_mismatch_raises(self):
        nodes = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        with pytest.raises(ValueError, match="must match number of structural nodes"):
            calculate_cg(nodes, m_arr=[1.0, 1.0, 1.0])

    def test_bad_node_shape_raises(self):
        with pytest.raises(ValueError, match=r"shape \(n_nodes, 3\)"):
            calculate_cg(np.array([1.0, 2.0, 3.0]), m_arr=[1.0])


# ============================================================================
# calculate_inertia
# ============================================================================


class TestCalculateInertia:
    def test_single_point_mass_about_origin(self):
        # Mass 1 at (0, 2, 0): Ixx = Izz = m*r^2 = 4, Iyy = 0.
        tensor = calculate_inertia([(np.array([0.0, 2.0, 0.0]), 1.0)])
        assert tensor[0, 0] == pytest.approx(4.0)
        assert tensor[1, 1] == pytest.approx(0.0)
        assert tensor[2, 2] == pytest.approx(4.0)

    def test_tensor_is_symmetric(self):
        nodes = [
            (np.array([1.0, 2.0, 3.0]), 2.0),
            (np.array([-2.0, 1.0, 0.5]), 1.5),
        ]
        tensor = calculate_inertia(nodes)
        assert np.allclose(tensor, tensor.T)

    def test_symmetric_pair_has_zero_products(self):
        # Two equal masses mirrored across the x-axis -> Ixy, Iyz vanish.
        nodes = [
            (np.array([1.0, 2.0, 0.0]), 1.0),
            (np.array([1.0, -2.0, 0.0]), 1.0),
        ]
        tensor = calculate_inertia(nodes)
        assert tensor[0, 1] == pytest.approx(0.0)
        assert tensor[1, 2] == pytest.approx(0.0)

    def test_invalid_node_entry_raises(self):
        with pytest.raises(ValueError, match=r"\[position, mass\]"):
            calculate_inertia([(np.array([0.0, 0.0, 0.0]),)])


# ============================================================================
# calculate_moments_of_inertia
# ============================================================================


class TestCalculateMomentsOfInertia:
    def test_axis_moments_match_diagonal(self):
        nodes = np.array([[0.0, 2.0, 0.0]])
        moments = calculate_moments_of_inertia(nodes, m_arr=[1.0])
        # Ixx = 4, Iyy = 0, Izz = 4 for a point on the +y axis.
        assert np.allclose(moments, [4.0, 0.0, 4.0])

    def test_reference_point_shifts_moment(self):
        nodes = np.array([[0.0, 2.0, 0.0]])
        about_origin = calculate_moments_of_inertia(nodes, m_arr=[1.0])
        about_point = calculate_moments_of_inertia(
            nodes, m_arr=[1.0], point=(0.0, 2.0, 0.0)
        )
        # About the mass location the moments collapse to zero.
        assert np.allclose(about_point, [0.0, 0.0, 0.0])
        assert about_origin[0] > about_point[0]


# ============================================================================
# The awesIO structure: KCU mass, solver arrays, aggregate, line report
# ============================================================================

import logging  # noqa: E402
from pathlib import Path  # noqa: E402

import yaml  # noqa: E402

from awetrim.aerostructural.utils import (  # noqa: E402
    compute_kite_aggregate,
    printing_rest_lengths,
    resolve_kcu_mass,
    resolve_structure,
)

KITE_DIR = Path(__file__).resolve().parents[2] / "data" / "LEI-V3-KITE"


@pytest.fixture(scope="module")
def lei_v3_system():
    return yaml.safe_load((KITE_DIR / "system.yaml").read_text(encoding="utf-8"))


def test_kcu_mass_prefers_the_system_file(lei_v3_system):
    from awetrim.utils.system_config import get_kcu_mass

    expected = get_kcu_mass(lei_v3_system)
    assert resolve_kcu_mass({"kcu": {"mass": 1.0}}, lei_v3_system) == pytest.approx(expected)
    assert resolve_kcu_mass({"kcu": {"mass": 1.0}}) == 1.0
    assert resolve_kcu_mass({"kcu_mass": 2.0}) == 2.0


def test_kcu_mass_defaults_to_zero_with_a_warning(caplog):
    with caplog.at_level(logging.WARNING):
        assert resolve_kcu_mass() == 0.0
    assert "KCU mass" in caplog.text


def test_resolve_structure_puts_the_kcu_mass_on_the_kcu(lei_v3_system):
    light = resolve_structure(KITE_DIR / "structure.yml", config={"kcu": {"mass": 1.0}})
    heavy = resolve_structure(KITE_DIR / "structure.yml", config={"kcu": {"mass": 3.0}})
    assert heavy[1][0] - light[1][0] == pytest.approx(2.0)
    np.testing.assert_allclose(heavy[1][1:], light[1][1:])


def test_resolve_structure_ignores_strut_padding_with_a_warning(caplog):
    with caplog.at_level(logging.WARNING):
        resolve_structure(KITE_DIR / "structure.yml", config={"kcu_mass": 0.0, "strut_padding": "legacy"})
    assert "strut_padding" in caplog.text


def test_kite_aggregate_weighs_the_structure_and_the_kcu(lei_v3_system):
    from awetrim.aerostructural.coupled.awesio_structure import load_structure
    from awetrim.utils.system_config import get_kcu_mass

    structure = load_structure(KITE_DIR / "structure.yml")
    aggregate = compute_kite_aggregate(structure, lei_v3_system)
    expected = structure.masses().sum() + get_kcu_mass(lei_v3_system)
    assert aggregate["mass"] == pytest.approx(expected, abs=1e-4)
    assert np.asarray(aggregate["inertia_tensor"]).shape == (3, 3)


def test_printing_rest_lengths_reports_every_line(capsys):
    from awetrim.aerostructural.coupled.awesio_structure import load_structure

    structure = load_structure(KITE_DIR / "structure.yml")
    positions = np.stack([structure.positions, structure.positions])
    printing_rest_lengths({"positions": positions}, structure)
    out = capsys.readouterr().out
    for name in {e["name"] for e in structure.bridle}:
        assert name in out

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

    def test_legacy_scalar_angle_rotates_about_y(self):
        # Backward-compatible single scalar => rotation about +Y only.
        nodes = np.array([[1.0, 0.0, 0.0]])
        out = rotate_geometry(nodes, angle_deg=90.0)
        assert np.allclose(out, [[0.0, 0.0, -1.0]], atol=1e-12)

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

    def test_rejects_wrong_node_shape(self):
        nodes = np.array([1.0, 0.0, 0.0])  # 1-D, not (n, 3)
        with pytest.raises(ValueError, match=r"shape \(n_nodes, 3\)"):
            rotate_geometry(nodes, angle_deg=[0, 0, 0])


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

"""Unit tests for awetrim.aerostructural.tracking

The tracking helpers preallocate and fill the per-iteration history arrays
(positions, external/internal forces, residual norms, and optional per-panel
aero state) consumed by result storage and plotting.
"""

import numpy as np
import pytest

from awetrim.aerostructural.tracking import (
    setup_tracking_arrays,
    update_tracking_arrays,
    update_aero_tracking,
)


class TestSetupTrackingArrays:
    def test_core_array_shapes(self):
        t_vector = np.linspace(0, 3, 4)
        arrays = setup_tracking_arrays(n_pts=5, t_vector=t_vector)
        assert arrays["positions"].shape == (4, 5, 3)
        assert arrays["f_ext"].shape == (4, 5, 3)
        assert arrays["f_int"].shape == (4, 5, 3)
        assert arrays["residual_norm"].shape == (4,)
        assert arrays["max_residual"].shape == (4,)

    def test_no_aero_arrays_when_zero_panels(self):
        arrays = setup_tracking_arrays(n_pts=2, t_vector=np.arange(3), n_panels=0)
        assert "alpha_at_ac" not in arrays
        assert "stall_mask" not in arrays

    def test_aero_arrays_allocated_when_panels(self):
        arrays = setup_tracking_arrays(n_pts=2, t_vector=np.arange(3), n_panels=6)
        assert arrays["alpha_at_ac"].shape == (3, 6)
        assert np.all(np.isnan(arrays["alpha_at_ac"]))
        assert arrays["stall_mask"].shape == (3, 6)
        assert arrays["stall_mask"].dtype == bool
        assert not arrays["stall_mask"].any()


class TestUpdateTrackingArrays:
    def test_stores_positions_and_forces(self):
        arrays = setup_tracking_arrays(n_pts=2, t_vector=np.arange(3))
        nodes = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        f_ext = np.arange(6, dtype=float)
        f_int = np.array([3.0, 0.0, 4.0, 0.0, 0.0, 0.0])

        update_tracking_arrays(arrays, 1, nodes, f_ext, f_int)

        assert np.allclose(arrays["positions"][1], nodes)
        assert np.allclose(arrays["f_ext"][1], f_ext.reshape(2, 3))
        assert np.allclose(arrays["f_int"][1], f_int.reshape(2, 3))

    def test_residual_and_max_norms(self):
        arrays = setup_tracking_arrays(n_pts=2, t_vector=np.arange(3))
        nodes = np.zeros((2, 3))
        f_int = np.array([3.0, 0.0, 4.0, 0.0, 0.0, 0.0])  # norm = 5, max = 4

        update_tracking_arrays(arrays, 1, nodes, np.zeros(6), f_int)

        assert arrays["residual_norm"][1] == pytest.approx(5.0)
        assert arrays["max_residual"][1] == pytest.approx(4.0)

    def test_relative_residual_is_normalised_by_the_resultant_load(self):
        # The coupled gate's measure: ||f_res|| / |sum f_ext|.
        arrays = setup_tracking_arrays(n_pts=2, t_vector=np.arange(3))
        f_ext = np.array([0.0, 0.0, 6.0, 0.0, 0.0, 4.0])  # resultant 10 N
        f_int = np.array([3.0, 0.0, 4.0, 0.0, 0.0, 0.0])  # norm 5 N

        update_tracking_arrays(arrays, 1, np.zeros((2, 3)), f_ext, f_int)

        assert arrays["residual_relative"][1] == pytest.approx(0.5)
        assert np.isnan(arrays["residual_relative"][0])
        assert np.isnan(arrays["residual_relative"][2])

    def test_relative_residual_is_nan_without_a_load(self):
        arrays = setup_tracking_arrays(n_pts=1, t_vector=np.arange(2))
        update_tracking_arrays(arrays, 1, np.zeros((1, 3)), np.zeros(3), np.ones(3))
        assert np.isnan(arrays["residual_relative"][1])

    def test_other_indices_untouched(self):
        arrays = setup_tracking_arrays(n_pts=1, t_vector=np.arange(3))
        update_tracking_arrays(arrays, 1, np.ones((1, 3)), np.ones(3), np.ones(3))
        assert np.allclose(arrays["positions"][0], 0.0)
        assert np.allclose(arrays["positions"][2], 0.0)


class TestUpdateAeroTracking:
    def test_stores_alpha_and_stall(self):
        arrays = setup_tracking_arrays(n_pts=2, t_vector=np.arange(3), n_panels=4)
        alpha = np.array([0.1, 0.2, 0.3, 0.4])
        stall = np.array([False, True, False, True])

        update_aero_tracking(arrays, 0, alpha, stall)

        assert np.allclose(arrays["alpha_at_ac"][0], alpha)
        assert list(arrays["stall_mask"][0]) == [False, True, False, True]

    def test_none_alpha_is_noop(self):
        arrays = setup_tracking_arrays(n_pts=2, t_vector=np.arange(3), n_panels=4)
        update_aero_tracking(arrays, 0, None, None)
        # Untouched: still all-NaN.
        assert np.all(np.isnan(arrays["alpha_at_ac"][0]))

    def test_missing_aero_arrays_is_noop(self):
        arrays = setup_tracking_arrays(n_pts=2, t_vector=np.arange(3), n_panels=0)
        # Must not raise even though no aero arrays exist.
        update_aero_tracking(arrays, 0, np.array([0.1]), np.array([True]))

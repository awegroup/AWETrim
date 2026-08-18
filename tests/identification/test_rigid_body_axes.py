"""Tests for awetrim.identification.rigid_body_axes (FRD sense, handedness)."""

import numpy as np
import pytest

from awetrim.identification.rigid_body_axes import (
    FRD_IN_STRUC,
    RigidBodyAxes,
    compute_rigid_body_axes,
)


def _box_cloud(lx: float, ly: float, lz: float, offset=(0.0, 0.0, 0.0)):
    """8 equal masses at the corners of an axis-aligned box: principal axes are
    the box axes with I_x < I_y < I_z iff lx > ly > lz."""
    corners = np.array(
        [[sx * lx, sy * ly, sz * lz] for sx in (-1, 1) for sy in (-1, 1)
         for sz in (-1, 1)],
        dtype=float,
    ) + np.asarray(offset, dtype=float)
    return corners, np.ones(len(corners))


def test_axes_are_frd_and_right_handed():
    # Span-wise (structural y) longest -> smallest inertia about y; chord (x)
    # next; thickness (z) smallest extent -> largest inertia about z.
    nodes, masses = _box_cloud(1.0, 4.0, 0.2, offset=(0.3, -0.1, 2.0))
    rb = compute_rigid_body_axes(nodes, masses)
    assert isinstance(rb, RigidBodyAxes)
    axes = rb.body_axes
    # Orthonormal, rows = body axes.
    np.testing.assert_allclose(axes @ axes.T, np.eye(3), atol=1e-12)
    # Right-handed: z = x cross y.
    np.testing.assert_allclose(np.cross(axes[0], axes[1]), axes[2], atol=1e-12)
    # FRD sense in the structural frame: forward = -X_struc, right = +Y_struc,
    # down = -Z_struc.
    for j in range(3):
        assert np.dot(axes[j], FRD_IN_STRUC[j]) > 0.99
    np.testing.assert_allclose(np.abs(axes), np.eye(3), atol=1e-12)


def test_frd_canonical_directions_are_right_handed():
    np.testing.assert_allclose(
        np.cross(FRD_IN_STRUC[0], FRD_IN_STRUC[1]), FRD_IN_STRUC[2]
    )
    # Down is opposite the course-frame radial (structural +z), right is
    # opposite the course-frame left (structural -y).
    assert FRD_IN_STRUC[2][2] < 0
    assert FRD_IN_STRUC[1][1] > 0


def test_principal_moments_and_cg_body():
    nodes, masses = _box_cloud(1.0, 4.0, 0.2, offset=(0.3, -0.1, 2.0))
    rb = compute_rigid_body_axes(nodes, masses)
    # Diagonalisation: rows @ I @ rows.T is diag(principal moments).
    diag = rb.body_axes @ rb.inertia_cg @ rb.body_axes.T
    np.testing.assert_allclose(diag, np.diag(rb.principal_moments), atol=1e-9)
    # Longest extent (y_struc = right/left) has the smallest moment (I_y).
    assert rb.principal_moments[1] < rb.principal_moments[0] < rb.principal_moments[2]
    np.testing.assert_allclose(rb.cg, [0.3, -0.1, 2.0])
    np.testing.assert_allclose(rb.cg_body, rb.body_axes @ rb.cg)


def test_flip_invariance_of_moments():
    """Flipping the sign of a principal axis never changes the moment; the
    FRD sense is a pure convention (sanity check of the flip logic)."""
    nodes, masses = _box_cloud(2.0, 1.0, 0.5)
    rb = compute_rigid_body_axes(nodes, masses)
    for j in range(3):
        m = rb.body_axes[j] @ rb.inertia_cg @ rb.body_axes[j]
        assert m == pytest.approx(rb.principal_moments[j])

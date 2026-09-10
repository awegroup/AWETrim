# Copyright (c) 2023-2026 Oriol Cayon, Delft University of Technology
#
# SPDX-License-Identifier: Apache-2.0

"""The SO(3) kernels every element with a material frame is built on.

The Cayley parameterisation is chosen over the exponential map because every map
is rational, so there is no removable singularity to branch around at
``phi = 0``. These tests pin down both halves of that claim: the maps are
correct, and they are well behaved at exactly zero.
"""

from __future__ import annotations

import casadi as ca
import numpy as np
import pytest

from awetrim.structural.rotations import (
    axial,
    cayley,
    cayley_vector,
    eye3,
    flat_to_frames,
    frames_to_flat,
    half_vector,
    orthonormalize,
    rotation_vector,
    skew,
    unflatten_frame,
)


def random_rodrigues(rng, max_angle: float = 2.8):
    """Rodrigues vector for a random rotation of up to ``max_angle`` radians."""
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    return 2.0 * np.tan(0.5 * rng.uniform(0.0, max_angle)) * axis


# --------------------------------------------------------------------------
# The maps themselves
# --------------------------------------------------------------------------


def test_cayley_returns_a_rotation():
    rng = np.random.default_rng(0)
    for _ in range(100):
        rotation = cayley(random_rodrigues(rng), np)
        assert rotation @ rotation.T == pytest.approx(np.eye(3), abs=1e-12)
        assert np.linalg.det(rotation) == pytest.approx(1.0, abs=1e-12)


def test_cayley_vector_inverts_cayley():
    rng = np.random.default_rng(1)
    for _ in range(100):
        psi = random_rodrigues(rng)
        assert cayley_vector(cayley(psi, np), np) == pytest.approx(psi, abs=1e-10)


def test_cayley_matches_the_rodrigues_formula():
    """Independent check against ``I + sin(phi) N + (1-cos(phi)) N^2``."""
    rng = np.random.default_rng(2)
    for _ in range(50):
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        angle = rng.uniform(-2.8, 2.8)
        expected = (
            np.eye(3)
            + np.sin(angle) * skew(axis, np)
            + (1.0 - np.cos(angle)) * skew(axis, np) @ skew(axis, np)
        )
        psi = 2.0 * np.tan(0.5 * angle) * axis
        assert cayley(psi, np) == pytest.approx(expected, abs=1e-10)


def test_half_vector_squares_to_the_full_rotation():
    rng = np.random.default_rng(3)
    for _ in range(100):
        psi = random_rodrigues(rng)
        half = cayley(half_vector(psi, np), np)
        assert half @ half == pytest.approx(cayley(psi, np), abs=1e-10)


def test_rotation_vector_recovers_the_angle_and_axis():
    """``psi = 2 tan(phi/2) n`` must map back to ``phi n``."""
    rng = np.random.default_rng(4)
    for _ in range(50):
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        angle = rng.uniform(0.05, 2.8)
        psi = 2.0 * np.tan(0.5 * angle) * axis
        assert rotation_vector(psi, np) == pytest.approx(angle * axis, abs=1e-9)


def test_skew_and_axial_are_inverse():
    rng = np.random.default_rng(5)
    for _ in range(50):
        vector = rng.normal(size=3)
        # axial() returns twice the axial vector of the skew part.
        assert axial(skew(vector, np), np) == pytest.approx(2.0 * vector, abs=1e-12)


# --------------------------------------------------------------------------
# Behaviour at exactly zero -- the whole reason for the Cayley form
# --------------------------------------------------------------------------


def test_identity_at_zero_without_a_special_case():
    zero = np.zeros(3)
    assert cayley(zero, np) == pytest.approx(np.eye(3), abs=1e-15)
    assert half_vector(zero, np) == pytest.approx(zero, abs=1e-15)
    assert rotation_vector(zero, np) == pytest.approx(zero, abs=1e-12)
    assert cayley_vector(np.eye(3), np) == pytest.approx(zero, abs=1e-15)


def test_half_vector_is_exactly_half_in_the_small_angle_limit():
    tiny = np.array([1e-8, -2e-8, 3e-9])
    assert half_vector(tiny, np) == pytest.approx(0.5 * tiny, rel=1e-6)


def test_maps_are_differentiable_at_zero_in_casadi():
    """No branch means CasADi gets a finite, correct Jacobian at psi = 0.

    An exponential-map implementation would need an ``if_else`` here, which is
    exactly the kink this parameterisation exists to avoid.
    """
    psi = ca.SX.sym("psi", 3)
    for expression in (
        ca.reshape(cayley(psi, ca), 9, 1),
        half_vector(psi, ca),
        rotation_vector(psi, ca),
    ):
        jacobian = ca.Function("j", [psi], [ca.jacobian(expression, psi)])
        values = np.asarray(jacobian(np.zeros(3)))
        assert np.all(np.isfinite(values)), "Jacobian is not finite at psi = 0"

    # d(rotation_vector)/d(psi) is the identity at the origin, which is what
    # makes -m . theta reduce to -m . psi to first order.
    jacobian = ca.Function("j", [psi], [ca.jacobian(rotation_vector(psi, ca), psi)])
    assert np.asarray(jacobian(np.zeros(3))) == pytest.approx(np.eye(3), abs=1e-6)


# --------------------------------------------------------------------------
# NumPy and CasADi must agree -- the xp namespace is a single source
# --------------------------------------------------------------------------


def test_numpy_and_casadi_paths_agree():
    rng = np.random.default_rng(6)
    symbol = ca.SX.sym("psi", 3)
    functions = {
        "cayley": ca.Function("f", [symbol], [ca.reshape(cayley(symbol, ca), 9, 1)]),
        "half": ca.Function("f", [symbol], [half_vector(symbol, ca)]),
        "rotvec": ca.Function("f", [symbol], [rotation_vector(symbol, ca)]),
    }
    for _ in range(20):
        psi = random_rodrigues(rng)
        assert np.asarray(functions["cayley"](psi)).reshape(3, 3).T == pytest.approx(
            cayley(psi, np), abs=1e-12
        )
        assert np.asarray(functions["half"](psi)).ravel() == pytest.approx(
            half_vector(psi, np), abs=1e-12
        )
        assert np.asarray(functions["rotvec"](psi)).ravel() == pytest.approx(
            rotation_vector(psi, np), abs=1e-12
        )


def test_eye3_matches_numpy_in_both_namespaces():
    assert eye3(np) == pytest.approx(np.eye(3))
    assert np.asarray(ca.DM(eye3(ca))) == pytest.approx(np.eye(3))


# --------------------------------------------------------------------------
# Frame storage
# --------------------------------------------------------------------------


def test_flatten_round_trips_row_major():
    rng = np.random.default_rng(7)
    frames = np.stack([cayley(random_rodrigues(rng), np) for _ in range(4)])
    assert flat_to_frames(frames_to_flat(frames)) == pytest.approx(frames)
    # Row-major: the first nine entries are the first frame read across rows.
    assert frames_to_flat(frames)[:9] == pytest.approx(frames[0].reshape(-1))


def test_unflatten_frame_agrees_across_namespaces():
    rng = np.random.default_rng(8)
    frame = cayley(random_rodrigues(rng), np)
    flat = frame.reshape(-1)
    assert unflatten_frame(flat, np) == pytest.approx(frame)

    symbol = ca.SX.sym("f", 9)
    function = ca.Function("f", [symbol], [ca.reshape(unflatten_frame(symbol, ca), 9, 1)])
    rebuilt = np.asarray(function(flat)).reshape(3, 3).T
    assert rebuilt == pytest.approx(frame, abs=1e-12)


def test_orthonormalize_projects_back_onto_so3():
    rng = np.random.default_rng(9)
    frame = cayley(random_rodrigues(rng), np)
    drifted = frame + 1e-3 * rng.normal(size=(3, 3))
    projected = orthonormalize(drifted)

    assert projected @ projected.T == pytest.approx(np.eye(3), abs=1e-12)
    assert np.linalg.det(projected) == pytest.approx(1.0, abs=1e-12)
    assert projected == pytest.approx(frame, abs=5e-3)


def test_orthonormalize_never_returns_a_reflection():
    reflection = np.diag([1.0, 1.0, -1.0])
    assert np.linalg.det(orthonormalize(reflection)) == pytest.approx(1.0, abs=1e-12)

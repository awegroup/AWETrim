"""Tests for awetrim.aerodynamics.parametric_airfoil.

These assert geometric structure, output shapes, and the .dat round-trip; no VSM
installation is required (the module is pure numpy).
"""

import inspect

import numpy as np
import pytest

from awetrim.aerodynamics.parametric_airfoil import (
    LEI_airfoil,
    cubic_bezier,
    generate_profile,
    reading_profile_from_airfoil_dat_files,
    save_profile_as_dat_file,
)


# A representative, well-inside-the-trained-box parameter set.
BASELINE = dict(
    t_val=0.08,
    eta_val=0.25,
    kappa_val=0.08,
    delta_val=-2.0,
    lambda_val=0.2,
    phi_val=0.5,
)


# ---------------------------------------------------------------------------
# Bezier primitive
# ---------------------------------------------------------------------------


def test_cubic_bezier_endpoints_and_shapes():
    P0 = np.array([0.0, 0.0])
    P1 = np.array([0.0, 1.0])
    P2 = np.array([1.0, 1.0])
    P3 = np.array([1.0, 0.0])
    t = np.linspace(0, 1, 25)
    points, slope = cubic_bezier(P0, P1, P2, P3, t)
    assert points.shape == (25, 2)
    assert slope.shape == (25,)
    # Curve passes through its endpoints.
    np.testing.assert_allclose(points[0], P0, atol=1e-12)
    np.testing.assert_allclose(points[-1], P3, atol=1e-12)


# ---------------------------------------------------------------------------
# Profile generation
# ---------------------------------------------------------------------------


def test_generate_profile_returns_closed_contour():
    all_points, profile_name, seam_a = generate_profile(**BASELINE)
    assert all_points.ndim == 2 and all_points.shape[1] == 2
    assert all_points.shape[0] > 100  # a finely discretised contour
    assert isinstance(profile_name, str) and profile_name.startswith("LEI_")
    assert np.isfinite(seam_a)
    assert np.all(np.isfinite(all_points))


def test_profile_normalised_to_unit_chord():
    all_points, _, _ = generate_profile(**BASELINE)
    # x is non-dimensionalised by chord: leading edge near 0, trailing edge near 1.
    assert all_points[:, 0].min() == pytest.approx(0.0, abs=0.05)
    assert all_points[:, 0].max() == pytest.approx(1.0, abs=0.05)


def test_lei_airfoil_returns_29_element_bundle():
    bundle = LEI_airfoil(
        tube_size=BASELINE["t_val"],
        c_x=BASELINE["eta_val"],
        c_y=BASELINE["kappa_val"],
        TE_angle=BASELINE["delta_val"],
        TE_cam_tension=BASELINE["lambda_val"],
        LE_tension=BASELINE["phi_val"],
    )
    assert isinstance(bundle, tuple)
    assert len(bundle) == 29


def test_flat_mode_when_camber_below_tube_radius():
    # kappa below the tube radius (t/2) triggers the flat branch; still valid geometry.
    params = dict(BASELINE)
    params["kappa_val"] = params["t_val"] / 2 - 0.01
    all_points, _, seam_a = generate_profile(**params)
    assert np.all(np.isfinite(all_points))
    assert np.isfinite(seam_a)


# ---------------------------------------------------------------------------
# .dat round-trip
# ---------------------------------------------------------------------------


def test_dat_file_round_trip(tmp_path):
    all_points, profile_name, seam_a = generate_profile(**BASELINE)
    out = save_profile_as_dat_file(
        all_points, profile_name, tmp_path / "profile.dat", seam_a=seam_a
    )
    parsed = reading_profile_from_airfoil_dat_files(out)
    assert parsed["name"] == profile_name
    read_points = np.asarray(parsed["points"])
    assert read_points.shape == all_points.shape
    np.testing.assert_allclose(read_points, all_points, atol=1e-7)


# ---------------------------------------------------------------------------
# Public signatures
# ---------------------------------------------------------------------------


def test_generate_profile_signature():
    sig = inspect.signature(generate_profile)
    params = list(sig.parameters)
    assert params[:6] == [
        "t_val",
        "eta_val",
        "kappa_val",
        "delta_val",
        "lambda_val",
        "phi_val",
    ]

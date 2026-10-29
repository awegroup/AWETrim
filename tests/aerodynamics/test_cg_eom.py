"""Tests for awetrim.aerodynamics.cg_eom pure helpers (no VSM required)."""

from __future__ import annotations

import numpy as np
import pytest

from awetrim.aerodynamics.cg_eom import pitch_neutral_point

E_PITCH = np.array([0.0, 1.0, 0.0])
E_X = np.array([1.0, 0.0, 0.0])


def _point_force_slopes(x_p: float, z_p: float, lift_slope: float):
    """Force slope applied at a fixed point p: dM_B = p x dF."""
    p = np.array([x_p, 0.0, z_p])
    dF = np.array([0.0, 0.0, lift_slope])
    return np.cross(p, dF), dF


def test_neutral_point_recovers_point_of_application():
    # A lift-slope force acting at x_p: the neutral point IS x_p, exactly,
    # independent of the (moment-free) z offset of the application point.
    dM, dF = _point_force_slopes(x_p=1.7, z_p=-9.0, lift_slope=500.0)
    res = pitch_neutral_point(dM, dF, pitch_axis=E_PITCH, x_axis=E_X)
    assert res["x_np_from_B_m"] == pytest.approx(1.7)
    assert not res["coupling_degenerate"]
    # margin sign agrees with the restoring verdict of the slope about B.
    assert res["restoring_B"] == (res["slope_B_Nm_per_rad"] < 0.0)
    assert (res["margin_B_m"] > 0.0) == res["restoring_B"]


def test_margin_is_signed_distance_to_neutral_point():
    dM, dF = _point_force_slopes(x_p=2.0, z_p=0.0, lift_slope=300.0)
    res = pitch_neutral_point(dM, dF, pitch_axis=E_PITCH, x_axis=E_X)
    # slope about a pivot at the neutral point must vanish: S(d) = S_B - d D.
    s_at_np = (
        res["slope_B_Nm_per_rad"]
        - res["x_np_from_B_m"] * res["coupling_N_per_rad"]
    )
    assert s_at_np == pytest.approx(0.0, abs=1e-9)
    assert abs(res["margin_B_m"]) == pytest.approx(abs(res["x_np_from_B_m"]))


def test_cg_transfer_matches_direct_moment_transport():
    dM, dF = _point_force_slopes(x_p=1.0, z_p=-8.0, lift_slope=400.0)
    c = np.array([0.4, 0.1, -7.5])
    res = pitch_neutral_point(
        dM, dF, pitch_axis=E_PITCH, x_axis=E_X, cg_offset=c
    )
    expected = float(E_PITCH @ (dM - np.cross(c, dF)))
    assert res["slope_CG_Nm_per_rad"] == pytest.approx(expected)
    assert res["cg_offset_along_x_m"] == pytest.approx(0.4)
    # Pure along-x CG offset: the two neutral points differ by exactly c_x.
    res_x = pitch_neutral_point(
        dM, dF, pitch_axis=E_PITCH, x_axis=E_X,
        cg_offset=np.array([0.4, 0.0, 0.0]),
    )
    assert res_x["x_np_from_CG_m"] == pytest.approx(
        res_x["x_np_from_B_m"] - 0.4
    )


def test_chord_fractions_and_axis_normalisation():
    dM, dF = _point_force_slopes(x_p=0.5, z_p=0.0, lift_slope=100.0)
    res = pitch_neutral_point(
        dM, dF, pitch_axis=2.0 * E_PITCH, x_axis=-3.0 * E_X,
        cg_offset=np.zeros(3), chord=2.5,
    )
    # Non-unit axes are normalised; flipped x_axis flips the position sign
    # but not the margin (distance to the same physical zero point).
    assert res["x_np_from_B_m"] == pytest.approx(-0.5)
    assert res["margin_B_frac"] == pytest.approx(res["margin_B_m"] / 2.5)
    assert res["margin_CG_frac"] == pytest.approx(res["margin_CG_m"] / 2.5)
    res_plus = pitch_neutral_point(dM, dF, pitch_axis=E_PITCH, x_axis=E_X)
    assert res["margin_B_m"] == pytest.approx(res_plus["margin_B_m"])


def test_degenerate_coupling_returns_none():
    # Force slope along the pitch axis: x_hat x dF is orthogonal to e_p, so
    # shifting the pivot along x cannot change the pitch slope.
    dM = np.array([0.0, -50.0, 0.0])
    dF = np.array([0.0, 123.0, 0.0])
    res = pitch_neutral_point(dM, dF, pitch_axis=E_PITCH, x_axis=E_X)
    assert res["coupling_degenerate"]
    assert res["x_np_from_B_m"] is None
    assert res["margin_B_m"] is None
    assert res["restoring_B"]  # the slope verdict itself still stands

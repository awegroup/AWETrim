import numpy as np
import pytest

from awetrim.kinematics.parametrized_patterns import (
    create_pattern_from_dict,
    make_bspline_path_parameters_from_named_curve,
    named_curve_angles,
)


def test_create_pattern_from_dict_rejects_unsupported_type():
    """A type with no constructor (e.g. cst_helix, the cycle-config default)
    must raise a clear ValueError listing supported types, not a KeyError."""
    with pytest.raises(ValueError, match="Unknown or unsupported pattern type"):
        create_pattern_from_dict("cst_helix", {})


def test_create_pattern_from_dict_reports_missing_params():
    """A supported type with missing params reports them explicitly."""
    with pytest.raises(ValueError, match="Missing required parameters"):
        create_pattern_from_dict("spline_open", {"M": 6})


def test_named_curve_angles_support_lissajous_and_helix():
    s = np.linspace(0.0, 2.0 * np.pi, 5)

    phi_lissajous, beta_lissajous = named_curve_angles(
        s,
        curve_type="lissajous",
        az_amp0=0.32,
        beta0=0.3,
        beta_amp0=0.15,
    )
    phi_helix, beta_helix = named_curve_angles(
        s,
        curve_type="helix",
        az_amp0=0.32,
        beta0=0.3,
        beta_amp0=0.15,
    )

    assert phi_lissajous.shape == s.shape
    assert beta_lissajous.shape == s.shape
    assert phi_helix.shape == s.shape
    assert beta_helix.shape == s.shape
    assert not np.allclose(beta_lissajous, beta_helix)


def test_named_curve_angles_lemniscate_is_amplitude_matched_figure_eight():
    """The Bernoulli lemniscate is a drop-in for the lissajous eight: same
    azimuth/elevation amplitudes, starting at the centre crossing."""
    s = np.linspace(0.0, 2.0 * np.pi, 2001)
    az_amp0, beta0, beta_amp0 = 0.32, 0.3, 0.15

    phi, beta = named_curve_angles(
        s,
        curve_type="lemniscate",
        az_amp0=az_amp0,
        beta0=beta0,
        beta_amp0=beta_amp0,
    )

    assert np.isclose(np.max(np.abs(phi)), az_amp0, atol=1e-6)
    assert np.isclose(np.max(np.abs(beta - beta0)), beta_amp0, atol=1e-6)
    # starts at the centre crossing and closes periodically there
    assert np.isclose(phi[0], 0.0, atol=1e-9)
    assert np.isclose(beta[0], beta0, atol=1e-9)
    assert np.isclose(phi[-1], 0.0, atol=1e-6)

    phi_liss, _ = named_curve_angles(
        s, curve_type="lissajous", az_amp0=az_amp0, beta0=beta0, beta_amp0=beta_amp0
    )
    assert not np.allclose(phi, phi_liss)


def test_named_curve_angles_rejects_unknown_curve():
    with pytest.raises(ValueError, match="lissajous"):
        named_curve_angles(np.array([0.0, 1.0]), curve_type="spiral")


@pytest.mark.parametrize("curve_type", ["lissajous", "lemniscate"])
def test_periodic_uploop_fit_matches_runtime(curve_type):
    """A reversed (uploop) fit must reproduce the curve once rebuilt the way
    create_pattern_from_dict does at sim time. Regression for the singular
    reversed-grid (u in [-1, 0]) basis matrix and the missing ``downloops``
    passthrough that left uploops evaluated in the downloop sense."""
    az_amp0, beta0, beta_amp0 = 0.32, 0.3, 0.15
    path_parameters = make_bspline_path_parameters_from_named_curve(
        spline_type="periodic",
        M=10,
        r0=230.0,
        s_init=0.0,
        s_final=2.0 * np.pi,
        n_fit=200,
        curve_type=curve_type,
        az_amp0=az_amp0,
        beta0=beta0,
        beta_amp0=beta_amp0,
        downloops=False,
    )
    assert path_parameters["downloops"] is False

    pattern = create_pattern_from_dict("spline_periodic", path_parameters)

    s = np.linspace(0.0, 2.0 * np.pi, 200, endpoint=True)
    az_target, el_target = named_curve_angles(
        s,
        curve_type=curve_type,
        az_amp0=az_amp0,
        beta0=beta0,
        beta_amp0=beta_amp0,
        downloops=False,
    )
    az_fit = np.array([float(pattern.azimuth(230.0, sv)) for sv in s])
    el_fit = np.array([float(pattern.elevation(230.0, sv)) for sv in s])

    assert np.max(np.abs(az_fit - az_target)) < 5e-2
    assert np.max(np.abs(el_fit - el_target)) < 5e-2


def test_make_periodic_bspline_path_parameters_are_pattern_ready():
    path_parameters = make_bspline_path_parameters_from_named_curve(
        spline_type="periodic",
        M=10,
        r0=230.0,
        s_init=0.0,
        s_final=2.0 * np.pi,
        n_fit=80,
        curve_type="helix",
        az_amp0=0.32,
        beta0=0.3,
        beta_amp0=0.15,
    )

    assert path_parameters["M"] == 10
    assert len(path_parameters["C_phi"]) == 10
    assert len(path_parameters["C_beta"]) == 10

    pattern = create_pattern_from_dict("spline_periodic", path_parameters)

    assert pattern.M == 10


def test_make_open_bspline_path_parameters_are_pattern_ready():
    path_parameters = make_bspline_path_parameters_from_named_curve(
        spline_type="open",
        M=6,
        r0=230.0,
        s_init=0.0,
        s_final=1.0,
        n_fit=40,
        curve_type="lissajous",
        az_amp0=0.32,
        beta0=0.3,
        beta_amp0=0.15,
    )

    assert path_parameters["M"] == 6
    assert len(path_parameters["C_phi"]) == 6
    assert len(path_parameters["C_beta"]) == 6

    pattern = create_pattern_from_dict("spline_open", path_parameters)

    assert pattern.M == 6

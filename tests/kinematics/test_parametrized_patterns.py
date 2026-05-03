import numpy as np

from awetrim.kinematics.parametrized_patterns import (
    create_pattern_from_dict,
    make_bspline_path_parameters_from_named_curve,
    named_curve_angles,
)


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

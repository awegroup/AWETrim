"""Tests for awetrim.identification.aerostructural_dataset."""

import numpy as np
import pytest

from awetrim.identification import aerostructural_dataset as ds


def _trim(alpha_deg=8.0, va=20.0, force=(0.0, 0.0, 2000.0)):
    a = np.deg2rad(alpha_deg)
    return {
        "opt_x": [20.0, 0.0, 0.0, 0.0, 0.0],
        "va_vel_world": [va * np.cos(a), 0.0, va * np.sin(a)],
        "omega_c_vsm": [0.0, -0.07, 0.0],
        "total_aero_force_vec": list(force),
        "kcu_drag_force_vsm": [50.0, 0.0, 0.0],
        "inertial_force": [0.0, 0.0, 30.0],
        "tether_force": 2000.0,
        "aoa_deg": alpha_deg,
        "stalled_fraction": 0.0,
        "success_physical": True,
    }


def _row(anchor="c/00", steer=0.0, **trim_kwargs):
    return ds.trim_row(
        anchor=anchor, chain="c", step=0, wind_speed=6.0, power_tape_length=1.9,
        steering_half_difference=steer, rho=1.225, mass_kcu=8.4, mass_total=19.4,
        trim=_trim(**trim_kwargs), meta={"converged": True, "n_iter": 9},
        body_z=[0.0, 0.0, -1.0], polar_check=1e-15,
    )


def test_vector_columns_round_trip():
    import pandas as pd

    table = pd.DataFrame([ds.vector_columns("va", [1.0, 2.0, 3.0])])
    assert list(table.columns) == ["va_x", "va_y", "va_z"]
    np.testing.assert_array_equal(ds.vector(table, "va"), [[1.0, 2.0, 3.0]])


def test_sweep_files_round_trip(tmp_path):
    polar = [
        {"d_alpha_deg": d, "alpha_body_deg": 8.0 + d, "va_world": np.ones(3),
         "force_world": np.full(3, np.nan) if d > 1 else np.zeros(3),
         "gamma_converged": d <= 1, "av_stage": None if d < 0 else "attached"}
        for d in (-1.0, 0.0, 1.0, 2.0)
    ]
    ds.append_rows(tmp_path / ds.TRIMS_FILE, [_row()])
    ds.append_rows(tmp_path / ds.POLARS_FILE, ds.polar_rows("c/00", polar[:2]))
    ds.append_rows(tmp_path / ds.POLARS_FILE, ds.polar_rows("c/00", polar[2:]))
    trims, polars = ds.load_sweep(tmp_path)
    assert len(trims) == 1 and len(polars) == 4
    assert trims.loc[0, "omega_y"] == -0.07
    assert bool(trims.loc[0, "converged"]) is True
    # An empty AV stage stays an empty string; a failed solve's NaN stays NaN.
    assert polars["av_stage"].tolist() == ["", "attached", "attached", "attached"]
    assert np.isnan(polars.loc[3, "force_aero_x"])


def test_lift_drag_coefficients_sign_and_scale():
    va = np.array([[20.0, 0.0, 0.0]])
    q_s = 0.5 * 1.225 * 400.0 * 10.0
    cl, cd = ds.lift_drag_coefficients([[100.0, 0.0, 500.0]], va, [[0.0, 0.0, -1.0]],
                                       1.225, 10.0)
    assert cl[0] == pytest.approx(500.0 / q_s)
    assert cd[0] == pytest.approx(100.0 / q_s)
    # Lift below the body (body up axis flipped) is negative lift.
    cl_neg, _ = ds.lift_drag_coefficients([[100.0, 0.0, 500.0]], va,
                                          [[0.0, 0.0, 1.0]], 1.225, 10.0)
    assert cl_neg[0] == pytest.approx(-cl[0])


def test_bridle_angle_of_attack_definition():
    va_c = np.array([[18.0, 3.0, 5.0]])
    _, e_r = ds._va_frame(va_c)
    e_chi = -va_c / np.linalg.norm(va_c)
    for a in (0.0, 0.1, -0.2):
        f = np.sin(a) * e_chi - np.cos(a) * e_r
        assert ds.bridle_angle_of_attack(f, va_c)[0] == pytest.approx(a)


def test_roll_about_va_zero_along_radial():
    va_c = np.array([[20.0, 0.0, 2.0]])
    _, e_r = ds._va_frame(va_c)
    assert ds.roll_about_va(e_r, va_c)[0] == pytest.approx(0.0)


def test_coefficient_tables_shapes_and_controls(tmp_path):
    import pandas as pd

    trims = pd.DataFrame([_row("c/00"), _row("c/01", steer=0.07)])
    polars = pd.DataFrame(
        ds.polar_rows("c/00", [
            {"d_alpha_deg": d, "alpha_body_deg": 8.0 + d,
             "va_world": [20.0, 0.0, 2.0], "force_world": [100.0, 0.0, 2000.0],
             "gamma_converged": True, "av_stage": "attached"}
            for d in (-1.0, 0.0, 1.0)
        ])
    )
    samples, by_trim = ds.coefficient_tables(
        trims, polars, reference_area=19.75, steering_tape_per_unit_us=1.4
    )
    # Two coupled-trim rows plus the two non-zero polar steps of c/00.
    assert sorted(samples.source.tolist()) == ["coupled_trim"] * 2 + ["polar"] * 2
    assert set(samples.loc[samples.source == "polar", "d_alpha_deg"]) == {-1.0, 1.0}
    stepped = samples[(samples.anchor == "c/00") & (samples.d_alpha_deg == 1.0)]
    assert stepped.alpha.iloc[0] == pytest.approx(np.deg2rad(9.0))
    assert by_trim.set_index("anchor").loc["c/01", "u_s"] == pytest.approx(0.05)
    assert (by_trim.theta_b == by_trim.alpha_b - by_trim.alpha_w).all()
    assert {"phi_a_w", "tension", "v_tau", "chi_dot"} <= set(by_trim.columns)


def test_identification_settings_merge_the_fidelity_block(tmp_path):
    path = tmp_path / "identification.yaml"
    path.write_text(
        "fidelity: wireframe\n"
        "sweep:\n"
        "  wind_speeds: [4.0]\n"
        "  steering: {max: 0.1, step: 0.025}\n"
        "  fidelities:\n"
        "    wireframe: {steering: {power_tape_lengths: [1.7]}, bridle_modulus_factor: 9.0}\n"
        "    billow: {steering: {power_tape_lengths: [2.1]}}\n",
        encoding="utf-8",
    )
    fidelity, spec = ds.identification_settings(path, "sweep")
    assert fidelity == "wireframe"
    assert spec["steering"] == {"max": 0.1, "step": 0.025, "power_tape_lengths": [1.7]}
    assert spec["bridle_modulus_factor"] == 9.0 and "fidelities" not in spec
    _, billow = ds.identification_settings(path, "sweep", "billow")
    assert billow["steering"]["power_tape_lengths"] == [2.1]
    with pytest.raises(ValueError):
        ds.identification_settings(path, "sweep", "pss")

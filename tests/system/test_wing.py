import casadi as ca
import pytest
from picawe.system.system_model import SystemModel
from picawe.system.kite import Kite
import numpy as np


@pytest.mark.parametrize("model", ["inviscid", "coeffs"])
def test_aerodynamic_force_coefficients(model):
    if model == "inviscid":
        aero_input = {
            "model": "inviscid",
            "params": {
                "oswald_efficiency": 0.8,
                "aspect_ratio": 6.0,
                "CD0": 0.05,
                "angle_pitch_depower_0": 0.05,
                "delta_pitch_depower": 0.05,
            },
        }
    else:
        aero_input = {
            "model": "coeffs",
            "params": {
                "CL0": 0.1,
                "CD0": 0.05,
                "CS0": 0.0,
                "angle_pitch_depower_0": 0.05,
                "delta_pitch_depower": 0.05,
            },
            "coefficients": {
                "CL": [{"var": "alpha", "coef": 2.0, "power": 1}],
                "CD": [{"var": "alpha", "coef": 0.1, "power": 2}],
            },
        }

    kite = Kite(mass_wing=1.0, area_wing=10.0, aero_input=aero_input)
    system = SystemModel(kite=kite, dof=3, quasi_steady=True)
    system.angle_azimuth = 0
    system.angle_elevation = 0
    system.angle_course = 0
    system.speed_tangential = 50
    system.speed_radial = 0
    system.wind.speed_wind_ref = 10
    system.input_steering = 0.0
    ups = [0, 1]
    for up in ups:
        system.input_depower = up
        system.input_steering = 0.0

        aoa = float(system.angle_of_attack)
        aoa_expected = (
            np.arctan2(system.wind.speed_wind_ref, system.speed_tangential)
            + aero_input["params"].get("angle_pitch_depower_0", 0.0)
            + up * aero_input["params"].get("delta_pitch_depower", 0.0)
        )
        assert np.isclose(
            aoa, aoa_expected, rtol=1e-8
        ), f"AOA mismatch: got {aoa}, expected {aoa_expected}"

        if model == "inviscid":
            AR = aero_input["params"]["aspect_ratio"]
            e = aero_input["params"]["oswald_efficiency"]
            CD0 = aero_input["params"]["CD0"]
            CL = (2 * np.pi * aoa) / (1 + 2 / (AR * e))
            CD = CD0 + CL**2 / (np.pi * AR * e)

        else:
            CL0 = aero_input["params"]["CL0"]
            CD0 = aero_input["params"]["CD0"]
            CL = CL0 + 2.0 * aoa
            CD = CD0 + 0.1 * aoa**2

        CL_actual, CD_actual, CS_actual = system.aerodynamic_force_coefficients

        assert np.isclose(
            float(CL_actual), CL, rtol=1e-8
        ), f"CL mismatch: got {CL_actual}, expected {CL}"
        assert np.isclose(
            float(CD_actual), CD, rtol=1e-8
        ), f"CD mismatch: got {CD_actual}, expected {CD}"

        CL_actual = system.lift_coefficient
        CD_actual = system.drag_coefficient

        assert np.isclose(
            float(CL_actual), CL, rtol=1e-8
        ), f"Lift coefficient mismatch: got {CL_actual}, expected {CL}"
        assert np.isclose(
            float(CD_actual), CD, rtol=1e-8
        ), f"Drag coefficient mismatch: got {CD_actual}, expected {CD}"


@pytest.mark.parametrize(
    "azimuth, elevation, radial_distance, angle_course, speed_tangential, speed_wind, velocity_apparent_expected",
    [
        (0, 0, 100.0, 0.0, 50.0, 10.0, np.array([-50.0, 0.0, 10.0])),
        (np.pi / 2, 0, 100.0, np.pi / 2, 50.0, 10.0, np.array([-60, 0.0, 0.0])),
        (0, np.pi / 2, 100.0, np.pi, 50.0, 10.0, np.array([-40.0, 0.0, 0.0])),
        (0, 0, 100.0, -np.pi / 2, 50.0, 10.0, np.array([-50.0, 0.0, 10.0])),
        (np.pi / 2, 0, 100.0, 0.0, 50.0, 10.0, np.array([-50.0, 10.0, 0.0])),
    ],
)
def test_apparent_wind(
    azimuth,
    elevation,
    radial_distance,
    angle_course,
    speed_tangential,
    speed_wind,
    velocity_apparent_expected,
):
    aero_input = {
        "model": "inviscid",
        "params": {
            "oswald_efficiency": 0.8,
            "aspect_ratio": 6.0,
            "CD0": 0.05,
            "angle_pitch_depower_0": 0.05,
            "delta_pitch_depower": 0.05,
        },
    }
    kite = Kite(mass_wing=1.0, area_wing=10.0, aero_input=aero_input)
    system = SystemModel(kite=kite, dof=3, quasi_steady=True)

    # system.angle_azimuth = azimuth
    # system.angle_elevation = elevation
    # system.angle_course = angle_course
    # system.speed_tangential = speed_tangential
    # system.speed_radial = 0
    # system.wind.speed_wind_ref = speed_wind
    # system.input_steering = 0.0

    velocity_apparent_fun = ca.Function(
        "velocity_apparent",
        [
            system.angle_azimuth,
            system.angle_elevation,
            system.angle_course,
            system.speed_tangential,
            system.speed_radial,
            system.wind.speed_wind_ref,
        ],
        [system.velocity_apparent_wind_wing],
    )

    velocity_apparent = np.array(
        velocity_apparent_fun(
            azimuth,
            elevation,
            angle_course,
            speed_tangential,
            0,
            speed_wind,
        )
        .full()
        .flatten()
    )
    assert np.allclose(
        velocity_apparent, velocity_apparent_expected, rtol=1e-8
    ), f"Apparent wind mismatch: got {velocity_apparent}, expected {velocity_apparent_expected}"

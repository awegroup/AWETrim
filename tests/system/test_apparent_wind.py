import numpy as np
import pytest

from awetrim.environment.Wind import Wind
from awetrim.system.kite import Kite
from awetrim.system.system_model import SystemModel


_AERO_INPUT = {
    "model": "inviscid",
    "params": {
        "oswald_efficiency": 0.8,
        "aspect_ratio": 6.0,
        "CD0": 0.05,
        "angle_pitch_tether_0": -0.05,
        "slope_angle_pitch_tether_depower": -0.05,
    },
}


def _make_system(speed_wind: float) -> SystemModel:
    """Return a quasi-steady SystemModel with uniform, numeric wind from +x."""
    wind = Wind(wind_model="uniform", speed_wind_ref=speed_wind, direction_wind=0.0)
    kite = Kite(mass_wing=1.0, area_wing=10.0, aero_input=_AERO_INPUT)
    return SystemModel(kite=kite, dof=3, quasi_steady=True, wind_model=wind)


@pytest.mark.parametrize(
    "azimuth, elevation, radial_distance, angle_course, speed_tangential, speed_wind, expected",
    [
        # Case 1: wind from +x, kite at origin heading +x
        # Wind in C-frame → [0, 0, 10]; kite vel → [50, 0, 0]
        # va = [0,0,10] - [50,0,0] = [-50, 0, 10]
        (0, 0, 100.0, 0.0, 50.0, 10.0, [-50.0, 0.0, 10.0]),
        # Case 2: azimuth=90°, course=90° — wind projected onto C-frame gives [-10,0,0]
        # va = [-10,0,0] - [50,0,0] = [-60, 0, 0]
        (np.pi / 2, 0, 100.0, np.pi / 2, 50.0, 10.0, [-60.0, 0.0, 0.0]),
        # Case 3: elevation=90°, course=180° — T_CW = I, so wind in C = [10,0,0]
        # va = [10,0,0] - [50,0,0] = [-40, 0, 0]
        (0, np.pi / 2, 100.0, np.pi, 50.0, 10.0, [-40.0, 0.0, 0.0]),
        # Case 4: course=-90° — wind rotated to [0, 0, 10] same as case 1
        # va = [0,0,10] - [50,0,0] = [-50, 0, 10]
        (0, 0, 100.0, -np.pi / 2, 50.0, 10.0, [-50.0, 0.0, 10.0]),
        # Case 5: azimuth=90°, course=0° — wind in C gives [0,10,0]
        # va = [0,10,0] - [50,0,0] = [-50, 10, 0]
        (np.pi / 2, 0, 100.0, 0.0, 50.0, 10.0, [-50.0, 10.0, 0.0]),
    ],
)
def test_apparent_wind(
    azimuth, elevation, radial_distance, angle_course,
    speed_tangential, speed_wind, expected,
):
    """Apparent wind in the C-frame equals wind-in-C minus kite velocity.

    velocity_apparent_wind = T_C_from_W @ v_wind_W - [v_tau, 0, v_r]

    All properties are set to numeric floats so the CasADi expression evaluates
    to a DM (dense matrix) that can be compared directly against expected values.
    Wind blows from +x in the world frame (direction_wind=0), speed_radial=0.
    """
    system = _make_system(speed_wind)

    system.angle_azimuth = float(azimuth)
    system.angle_elevation = float(elevation)
    system.distance_radial = float(radial_distance)
    system.angle_course = float(angle_course)
    system.speed_tangential = float(speed_tangential)
    system.speed_radial = 0.0

    va = np.array(system.velocity_apparent_wind.full()).flatten()

    np.testing.assert_allclose(va, expected, atol=1e-10)


def test_wind_velocity_at_height_is_direction_aware():
    """velocity_wind_at_height_W must carry direction_wind, consistent with
    velocity_wind_W (regression: it previously put all speed on +x)."""
    import casadi as ca

    direction = 0.7
    speed = 10.0
    wind = Wind(wind_model="uniform", speed_wind_ref=speed, direction_wind=direction)

    v_at_height = np.array(ca.DM(wind.velocity_wind_at_height_W(100.0))).ravel()
    expected = speed * np.array([np.cos(direction), np.sin(direction), 0.0])
    np.testing.assert_allclose(v_at_height, expected, atol=1e-10)

    # Must match the kite-height world-frame wind helper.
    v_ref = np.array(ca.DM(wind.velocity_wind_W(100.0))).ravel()
    np.testing.assert_allclose(v_at_height, v_ref, atol=1e-10)


def test_wind_velocity_helpers_are_coherent():
    """velocity_wind_at_height(model, model.z) == velocity_wind(model) for a
    non-zero wind direction, so both course-frame wind paths agree."""
    wind = Wind(wind_model="uniform", speed_wind_ref=8.0, direction_wind=-0.4)
    kite = Kite(mass_wing=1.0, area_wing=10.0, aero_input=_AERO_INPUT)
    system = SystemModel(kite=kite, dof=3, quasi_steady=True, wind_model=wind)

    system.angle_azimuth = 0.3
    system.angle_elevation = 0.5
    system.distance_radial = 120.0
    system.angle_course = 1.1
    system.speed_tangential = 0.0
    system.speed_radial = 0.0

    v1 = np.array(system.wind.velocity_wind(system).full()).ravel()
    v2 = np.array(system.wind.velocity_wind_at_height(system, system.z).full()).ravel()
    np.testing.assert_allclose(v1, v2, atol=1e-10)

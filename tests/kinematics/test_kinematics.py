import pytest
import casadi as ca
import numpy as np


@pytest.mark.parametrize(
    "azimuth, elevation, radial_distance, expected_position, expected_position_W",
    [
        (0, 0, 100.0, np.array([0.0, 0.0, 100.0]), np.array([100.0, 0.0, 0.0])),
        (np.pi / 2, 0, 100.0, np.array([0.0, 0.0, 100.0]), np.array([0.0, 100.0, 0.0])),
        (np.pi, 0, 100.0, np.array([0.0, 0.0, 100.0]), np.array([-100.0, 0.0, 0.0])),
        (0, np.pi / 2, 100.0, np.array([0.0, 0.0, 100.0]), np.array([0.0, 0.0, 100.0])),
    ],
)
def test_position_properties(
    azimuth, elevation, radial_distance, expected_position, expected_position_W
):
    from picawe.kinematics.Kinematics import Position

    pos = Position()
    pos.angle_azimuth = azimuth
    pos.angle_elevation = elevation
    pos.distance_radial = radial_distance
    pos.angle_course = 0.0  # Not used in Position, but required for the class

    position = np.array(pos.position.full().flatten())  # CasADi DM → NumPy array
    position_W = np.array(pos.position_W.full().flatten())

    assert np.allclose(position, expected_position)
    assert np.allclose(position_W, expected_position_W)


@pytest.mark.parametrize(
    "azimuth, elevation, radial_distance, speed_tangential, speed_radial, angle_course, expected_velocity, expected_velocity_W",
    [
        (
            0,
            0,
            100.0,
            50.0,
            10.0,
            np.pi / 2,
            np.array([50.0, 0.0, 10.0]),
            np.array([10.0, 50.0, 0.0]),
        ),
        (
            np.pi / 2,
            0,
            100.0,
            50.0,
            10.0,
            np.pi / 2,
            np.array([50.0, 0.0, 10.0]),
            np.array([-50.0, 10.0, 0.0]),
        ),
        (
            0,
            0,
            100.0,
            50.0,
            10.0,
            0,
            np.array([50.0, 0.0, 10.0]),
            np.array([10.0, 0.0, 50.0]),
        ),
        (
            0,
            np.pi / 2,
            100.0,
            50.0,
            10.0,
            -np.pi / 2,
            np.array([50.0, 0.0, 10.0]),
            np.array([0.0, -50.0, 10.0]),
        ),
    ],
)
def test_velocity_properties(
    azimuth,
    elevation,
    radial_distance,
    speed_tangential,
    speed_radial,
    angle_course,
    expected_velocity,
    expected_velocity_W,
):
    from picawe.kinematics.Kinematics import KiteKinematics

    kite = KiteKinematics()
    kite.angle_azimuth = azimuth
    kite.angle_elevation = elevation
    kite.distance_radial = radial_distance
    kite.speed_tangential = speed_tangential
    kite.speed_radial = speed_radial
    kite.angle_course = angle_course

    velocity = np.array(kite.velocity_kite.full().flatten())
    velocity_W = np.array(kite.velocity_kite_W.full().flatten())

    assert np.allclose(velocity, expected_velocity)
    assert np.allclose(velocity_W, expected_velocity_W)


@pytest.mark.parametrize(
    "azimuth, elevation, radial_distance, speed_tangential, speed_radial, angle_course, course_rate, expected_omega_course",
    [
        (
            0,
            0,
            100.0,
            50.0,
            10.0,
            np.pi / 2,
            1,
            np.array([0.0, 0.5, -1]),
        ),
        (
            np.pi / 2,
            0,
            100.0,
            50.0,
            10.0,
            np.pi / 2,
            0,
            np.array([0.0, 0.5, 0]),
        ),
        (
            0,
            0,
            100.0,
            50.0,
            10.0,
            0,
            -1,
            np.array([0.0, 0.5, 1]),
        ),
    ],
)
def test_omega_course_properties(
    azimuth,
    elevation,
    radial_distance,
    speed_tangential,
    speed_radial,
    angle_course,
    course_rate,
    expected_omega_course,
):
    from picawe.kinematics.Kinematics import KiteKinematics

    kite = KiteKinematics()
    kite.angle_azimuth = azimuth
    kite.angle_elevation = elevation
    kite.distance_radial = radial_distance
    kite.speed_tangential = speed_tangential
    kite.speed_radial = speed_radial
    kite.angle_course = angle_course
    kite.timeder_angle_course = course_rate

    omega_course = np.array(kite.velocity_rotation_course_frame.full().flatten())
    assert np.allclose(
        omega_course, expected_omega_course
    ), f"Expected {expected_omega_course}, got {omega_course}"

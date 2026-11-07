"""TurnRadiusTable interpolation and the steering box the table is built at."""
from types import SimpleNamespace

import numpy as np
import pytest

from awetrim.system.turn_radius import TurnRadiusTable, steering_limit
from awetrim.utils.defaults import DEFAULT_OPTI_LIMITS


def table(positive, negative=None):
    positive = np.asarray(positive, dtype=float)
    courses = np.linspace(0.0, 2.0 * np.pi, positive.size, endpoint=False)
    return TurnRadiusTable(
        courses=courses,
        radius_positive=positive,
        radius_negative=positive if negative is None else np.asarray(negative, dtype=float),
        input_steering_max=0.14,
        distance_radial=200.0,
        angle_elevation=0.4,
        input_depower=1.6,
    )


def test_radius_interpolates_periodically_and_by_turn_sign():
    t = table([10.0, 20.0, 30.0, 40.0], [1.0, 2.0, 3.0, 4.0])
    assert t.radius(np.array([np.pi / 4.0]), +1.0)[0] == pytest.approx(15.0)
    assert t.radius(np.array([np.pi / 4.0]), -1.0)[0] == pytest.approx(1.5)
    # Past the last tabulated course: back towards the first.
    assert t.radius(np.array([7.0 * np.pi / 4.0]), +1.0)[0] == pytest.approx(25.0)
    assert t.radius(np.array([-np.pi / 4.0]), +1.0)[0] == pytest.approx(25.0)


def test_radius_is_inf_next_to_an_unflyable_course():
    t = table([10.0, np.nan, 30.0, 40.0])
    assert np.isinf(t.radius(np.array([np.pi / 4.0]), +1.0)[0])
    assert t.radius(np.array([5.0 * np.pi / 4.0]), +1.0)[0] == pytest.approx(35.0)
    assert t.summary()["unflyable"] == 2


def test_steering_limit_is_the_optimizer_box():
    clipped = SimpleNamespace(hardware_limits={"input_steering": (-0.1429, 0.1429)})
    assert steering_limit(clipped) == pytest.approx(0.1429)
    assert steering_limit(SimpleNamespace()) == pytest.approx(
        DEFAULT_OPTI_LIMITS["input_steering"][1]
    )

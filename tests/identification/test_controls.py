import numpy as np
import pandas as pd
import pytest

from awetrim.identification.controls import (
    AS_2019_DEPOWERED_EXTENSION_M,
    AS_2019_POWERED_EXTENSION_M,
    flight_dataframe_depower_to_power_tape_length,
    flight_dataframe_steering_to_us,
    flight_depower_to_power_tape_length,
    flight_steering_to_us,
    power_tape_extension_to_length,
    steering_extension_to_us,
)


def test_2019_flight_depower_maps_to_absolute_power_tape_length():
    kcu = np.array([22.0, 30.0])

    l_dp = flight_depower_to_power_tape_length(kcu)

    np.testing.assert_allclose(l_dp, [1.7, 2.1])


def test_2019_as_extensions_match_powered_and_depowered_lengths():
    extensions = np.array([AS_2019_POWERED_EXTENSION_M, AS_2019_DEPOWERED_EXTENSION_M])

    l_dp = power_tape_extension_to_length(extensions)

    np.testing.assert_allclose(l_dp, [1.7, 2.1])


def test_steering_conventions_for_flight_and_as_inputs():
    # Standardised u_s: 2019 kcu/200, so 1.4*u_s is the tape half-difference.
    np.testing.assert_allclose(flight_steering_to_us([10.0, -20.0]), [-0.05, 0.1])
    np.testing.assert_allclose(steering_extension_to_us([0.1, 0.2]), [0.1, 0.2])


def test_flight_dataframe_helpers_prefer_kcu_columns():
    df = pd.DataFrame(
        {
            "kcu_actual_depower": [22.0, 30.0],
            "kcu_actual_steering": [10.0, -20.0],
            "up": [0.0, 1.0],
            "us": [1.0, -1.0],
        }
    )

    np.testing.assert_allclose(
        flight_dataframe_depower_to_power_tape_length(df), [1.7, 2.1]
    )
    np.testing.assert_allclose(flight_dataframe_steering_to_us(df), [-0.05, 0.1])


def test_flight_dataframe_depower_rejects_legacy_up_only():
    with pytest.raises(KeyError, match="kcu_actual_depower"):
        flight_dataframe_depower_to_power_tape_length(pd.DataFrame({"up": [0.0, 1.0]}))

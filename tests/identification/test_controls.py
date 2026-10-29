import numpy as np
import pandas as pd
import pytest

from awetrim.identification.controls import (
    flight_dataframe_depower_to_power_tape_length,
    flight_dataframe_steering_to_us,
    flight_depower_to_power_tape_length,
    flight_steering_to_us,
    power_tape_extension_to_length,
    steering_extension_to_us,
)


MATCH = (1.94, 2.40)


def test_2019_flight_depower_maps_through_the_match():
    kcu = np.array([22.0, 30.0, 26.0])

    l_dp = flight_depower_to_power_tape_length(kcu, MATCH)

    np.testing.assert_allclose(l_dp, [1.94, 2.40, 2.17])


def test_flight_depower_needs_a_match():
    with pytest.raises(ValueError, match="flight_match"):
        flight_depower_to_power_tape_length([22.0], None)


def test_lei_v3_records_its_2019_match():
    from awetrim.identification.controls import flight_depower_match
    from awetrim.utils.config_paths import LEI_V3_DATA_DIR

    reel_out, reel_in = flight_depower_match(LEI_V3_DATA_DIR, 2019)
    assert reel_out < reel_in
    assert flight_depower_match(LEI_V3_DATA_DIR, 2025) is None


def test_as_extension_adds_to_the_built_tape():
    np.testing.assert_allclose(power_tape_extension_to_length([0.1, 0.5], 1.9), [2.0, 2.4])


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
        flight_dataframe_depower_to_power_tape_length(df, MATCH), [1.94, 2.40]
    )
    np.testing.assert_allclose(flight_dataframe_steering_to_us(df), [-0.05, 0.1])


def test_flight_dataframe_depower_rejects_legacy_up_only():
    with pytest.raises(KeyError, match="kcu_actual_depower"):
        flight_dataframe_depower_to_power_tape_length(pd.DataFrame({"up": [0.0, 1.0]}), MATCH)


def test_rom_depower_band_requires_the_controls_block():
    from awetrim.identification.controls import rom_depower_band

    with pytest.raises(ValueError):
        rom_depower_band(None)
    with pytest.raises(ValueError):
        rom_depower_band({"aerodynamics": {"model": "coeffs"}})


def test_rom_depower_band_reads_the_kite_controls_block(tmp_path):
    import yaml

    from awetrim.identification.controls import rom_depower_band

    cfg = {"controls": {"input_depower": {"powered": 0.1, "depowered": 0.9}}}
    assert rom_depower_band(cfg) == (0.1, 0.9)
    path = tmp_path / "rom_config.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    assert rom_depower_band(path) == (0.1, 0.9)

"""Unit tests for awetrim.flight_data.

The fixture writes a small file the way EKF-AWE's save_results does (byte
strings for text columns, the configuration as nested groups of attributes),
so the reader is pinned against the producer's format without real flight data.
"""

import h5py
import numpy as np
import pandas as pd
import pytest

from awetrim.flight_data import FlightData, cycles_from_phases, load_flight


@pytest.fixture
def ekf_file(tmp_path):
    path = tmp_path / "LEI-V3 Kite_2019-10-08.h5"
    n = 8
    with h5py.File(path, "w") as hf:
        ekf = hf.create_group("ekf_output")
        ekf.create_dataset("CL", data=np.linspace(0.8, 1.0, n))
        ekf.create_dataset("kite_roll", data=np.zeros(n))
        flight = hf.create_group("flight_data")
        flight.create_dataset("cycle", data=np.array([0, 0, 1, 1, 1, 2, 2, 3], dtype=float))
        flight.create_dataset("flight_phase", data=np.array([b"pp-ro"] * n))
        config = hf.create_group("config_data")
        config.attrs["system_yaml_used"] = b"system_flown_2019.yaml"
        kite = config.create_group("kite")
        kite.attrs["model_name"] = b"LEI-V3 Kite"
        kite.attrs["area"] = 19.75
        sim = config.create_group("simulation_parameters")
        sim.create_group("measurements").attrs["kite_apparent_windspeed"] = True
    return path


def test_reads_all_three_groups(ekf_file):
    flight = load_flight(ekf_file)
    assert isinstance(flight, FlightData)
    assert list(flight.ekf.columns) == ["CL", "kite_roll"]
    assert len(flight.ekf) == len(flight.measured) == 8
    assert flight.path == ekf_file


def test_byte_strings_are_decoded(ekf_file):
    flight = load_flight(ekf_file)
    assert flight.measured["flight_phase"].iloc[0] == "pp-ro"
    assert flight.config["system_yaml_used"] == "system_flown_2019.yaml"
    assert flight.config["kite"]["model_name"] == "LEI-V3 Kite"


def test_config_keeps_its_nesting(ekf_file):
    config = load_flight(ekf_file).config
    assert config["kite"]["area"] == pytest.approx(19.75)
    assert bool(config["simulation_parameters"]["measurements"]["kite_apparent_windspeed"])


def test_cycles_selects_an_inclusive_range(ekf_file):
    flight = load_flight(ekf_file)
    sub = flight.cycles(1, 2)
    assert sub.measured["cycle"].tolist() == [1, 1, 1, 2, 2]
    assert sub.ekf["CL"].tolist() == pytest.approx(flight.ekf["CL"].iloc[2:7].tolist())
    assert sub.ekf.index.tolist() == list(range(5))
    assert flight.cycles(3).measured["cycle"].tolist() == [3]


def test_misaligned_groups_are_refused(tmp_path):
    path = tmp_path / "bad.h5"
    with h5py.File(path, "w") as hf:
        hf.create_group("ekf_output").create_dataset("CL", data=np.ones(3))
        hf.create_group("flight_data").create_dataset("cycle", data=np.ones(4))
        hf.create_group("config_data")
    with pytest.raises(ValueError, match="3 rows"):
        load_flight(path)


def test_cycles_from_phases_numbers_each_full_run():
    phases = pd.Series([4, 1, 1, 2, 3, 4, 4, 1, 2, 3, 4, 1, 2])
    ids = cycles_from_phases(phases)
    assert ids.tolist() == [-1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, -1, -1]
    assert cycles_from_phases(pd.Series([], dtype=float)).size == 0

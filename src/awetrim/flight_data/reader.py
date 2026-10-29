# Copyright (c) 2023-2026 Oriol Cayon, Delft University of Technology
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""Read a flight reconstructed by EKF-AWE.

AWETrim does not process raw flight logs: EKF-AWE does, and writes one HDF5
file per flight with three groups, all read here.

- ``ekf_output``: the filter's state estimates and the quantities derived from
  them, one dataset per column.
- ``flight_data``: the processed flight log the filter ran on, sample by
  sample aligned with ``ekf_output``.
- ``config_data``: the configuration the file was made with, as nested
  groups of attributes (kite, KCU, tether, simulation parameters, ...).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


@dataclass
class FlightData:
    """One EKF-AWE flight: estimates, measurements and configuration.

    ``ekf`` and ``measured`` have one row per sample, in the same order.
    """

    ekf: pd.DataFrame
    measured: pd.DataFrame
    config: dict
    path: Path | None = field(default=None, compare=False)

    def cycles(self, first: int, last: int | None = None) -> "FlightData":
        """The samples of pumping cycles ``first`` to ``last`` (inclusive).

        Uses the ``cycle`` column of the measured data; ``last`` defaults to
        ``first``. The index is reset, so the subset starts at row 0.
        """
        last = first if last is None else last
        mask = self.measured["cycle"].between(first, last).to_numpy()
        return FlightData(
            ekf=self.ekf[mask].reset_index(drop=True),
            measured=self.measured[mask].reset_index(drop=True),
            config=self.config,
            path=self.path,
        )


def load_flight(path: str | Path) -> FlightData:
    """Read an EKF-AWE output file (``<kite model>_<YYYY-MM-DD>.h5``)."""
    path = Path(path)
    with h5py.File(path, "r") as hf:
        ekf = _frame(hf["ekf_output"])
        measured = _frame(hf["flight_data"])
        config = _attrs_to_dict(hf["config_data"])
    if len(ekf) != len(measured):
        raise ValueError(
            f"{path.name}: ekf_output has {len(ekf)} rows, flight_data {len(measured)}"
        )
    return FlightData(ekf=ekf, measured=measured, config=config, path=path)


def cycles_from_phases(phase_series: pd.Series) -> np.ndarray:
    """A cycle id per row, a cycle being one run of flight phases 1, 2, 3, 4.

    Consecutive identical phases are collapsed, each ordered run [1, 2, 3, 4]
    is a cycle (ids from 0), and rows outside every detected run get -1.
    """
    phases = phase_series.to_numpy()
    if phases.size == 0:
        return np.array([], dtype=int)

    change_idx = np.flatnonzero(np.diff(phases) != 0) + 1
    run_starts = np.concatenate(([0], change_idx))
    run_values = phases[run_starts]

    cycle_ids = np.full(phases.shape, -1, dtype=int)
    pattern = [1, 2, 3, 4]
    cycle = -1
    for i in range(len(run_values) - len(pattern) + 1):
        if list(run_values[i : i + 4]) != pattern:
            continue
        cycle += 1
        start_idx = run_starts[i]
        end_idx = run_starts[i + 4] if i + 4 < len(run_starts) else phases.size
        cycle_ids[start_idx:end_idx] = cycle
    return cycle_ids


def _frame(group: h5py.Group) -> pd.DataFrame:
    """One column per dataset; byte strings decoded to ``str``."""
    return pd.DataFrame(
        {
            name: (
                group[name][:].astype(str)
                if group[name].dtype.kind == "S"
                else group[name][:]
            )
            for name in group
            if isinstance(group[name], h5py.Dataset)
        }
    )


def _attrs_to_dict(group: h5py.Group) -> dict:
    """A group's attributes and, recursively, its subgroups."""
    out = {}
    for key, value in group.attrs.items():
        out[key] = value.decode("utf-8") if isinstance(value, bytes) else value
    for name in group:
        if isinstance(group[name], h5py.Group):
            out[name] = _attrs_to_dict(group[name])
    return out

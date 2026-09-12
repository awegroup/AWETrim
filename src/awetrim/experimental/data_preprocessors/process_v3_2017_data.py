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
"""Pre-processor for the 2017 Valkenburg V3 flight logs.

Maps the legacy KP log format (``DA_2017-03-30_FL0x_*.csv``, space-delimited,
20 Hz) onto the shared processed-flight-data schema consumed by
``create_input_from_csv``. Column meanings follow the dataset README in the
Flightdata30032017 repository, with two verified corrections:

- ``vx``/``vy``/``vz`` are NED (north/east/down) as documented, but
  ``acc_x/y/z`` are NOT the documented NED kinematic acceleration: they carry
  gravity (mean |acc| ~ 11 m/s^2) and do not correlate with dv/dt, i.e. they
  are body-frame specific force. Acceleration is therefore derived from the
  smoothed gradient of the ENU velocity instead.
- ``heading``/``course`` are radians in the CSV, not the documented degrees.

Not available in these logs: pitot airspeed and the flow vanes (``airspeed_*``
never populated) and the ground tether angle sensors (``elevation_sens``/
``azimuth_north`` all-NaN). ``kite_apparent_windspeed`` is filled from
``est_app_wind_kite`` — the ground station's own model ESTIMATE, not a
measurement — so for EKF runs on these flights consider
``measurements.kite_apparent_windspeed: false`` and
``calibrate_apparent_windspeed: false`` in ekf_config.yaml.
"""

import numpy as np
import pandas as pd
from pathlib import Path

from awetrim.experimental.data_preprocessors.process_v3_data import (
    detect_delimiter,
    find_log_path,
    save_flight_data,
)


def _smoothed_gradient(values: np.ndarray, t: np.ndarray, window: int) -> np.ndarray:
    grad = np.gradient(values, t)
    return np.convolve(grad, np.ones(window) / window, mode="same")


def process_data(config_data: dict, log_directory: Path) -> None:
    log_date = f'{config_data["year"]}-{config_data["month"]}-{config_data["day"]}'
    log_path = find_log_path(log_directory, log_date, config_data.get("log_filename"))
    delimiter = detect_delimiter(log_path)
    log = pd.read_csv(log_path, delimiter=delimiter, low_memory=False)

    log = log[log["height"] > 50].reset_index(drop=True)
    numeric_cols = log.select_dtypes(include=[float, int]).columns
    # limit_direction="both" also backfills leading NaNs (the xsens channels
    # drop out in ~4 % chunks) so the first EKF sample is always finite.
    log.loc[:, numeric_cols] = log[numeric_cols].interpolate(limit_direction="both")

    t = log["time"].values
    dt = float(np.median(np.diff(t)))
    window = max(1, round(1.0 / dt))  # ~1 s smoothing at the 20 Hz log rate

    flight_data = pd.DataFrame()

    # Kite position: already ENU relative to the ground station.
    flight_data["kite_position_x"] = log["pos_east"]
    flight_data["kite_position_y"] = log["pos_north"]
    flight_data["kite_position_z"] = log["height"]

    # Kite velocity: NED in the log -> ENU.
    flight_data["kite_velocity_x"] = log["vy"]
    flight_data["kite_velocity_y"] = log["vx"]
    flight_data["kite_velocity_z"] = -log["vz"]

    # Kite acceleration from the velocity gradient (see module docstring on
    # why the logged acc_* channels cannot be used directly).
    for axis in ("x", "y", "z"):
        flight_data[f"kite_acceleration_{axis}"] = _smoothed_gradient(
            flight_data[f"kite_velocity_{axis}"].values, t, window
        )

    # Attitude: sensor 0 = Xsens, sensor 1 = Pixhawk (degrees in the log; the
    # Pixhawk Euler angles look unreliable — keep them only so the sensor_ids
    # [0, 1] postprocessing loop finds its columns).
    for sensor, source in ((0, "xsens"), (1, "pixhawk")):
        flight_data[f"kite_roll_{sensor}"] = np.deg2rad(log[f"phi_{source}"])
        flight_data[f"kite_pitch_{sensor}"] = np.deg2rad(log[f"theta_{source}"])
        flight_data[f"kite_yaw_{sensor}"] = np.deg2rad(log[f"psi_{source}"])
        for angle in ("roll", "pitch", "yaw"):
            flight_data[f"kite_{angle}_rate_{sensor}"] = _smoothed_gradient(
                np.unwrap(flight_data[f"kite_{angle}_{sensor}"].values), t, window
            )

    # Ground station data. Tether force is already in newtons here (unlike the
    # kgf-logged 2019/2025 files).
    flight_data["ground_tether_force"] = log["force"]
    flight_data["ground_wind_speed"] = log["wind_vel"]
    flight_data["ground_wind_direction"] = 360 - 90 - log["wind_dir"]
    flight_data["tether_length"] = log["tether_length"]
    flight_data["tether_reelout_speed"] = log["v_reelout"]

    # KCU control data (percent; create_input_from_csv normalises).
    flight_data["kcu_set_depower"] = log["set_depower"]
    flight_data["kcu_set_steering"] = log["set_steering"]
    flight_data["kcu_actual_steering"] = log["actual_steering"]
    flight_data["kcu_actual_depower"] = log["actual_depower"]

    # Airflow: no pitot on these flights — est_app_wind_kite is the ground
    # station's estimate of the relative flow at the kite.
    flight_data["kite_apparent_windspeed"] = log["est_app_wind_kite"]
    flight_data["bridle_angle_of_attack"] = np.full(len(log), np.nan)
    flight_data["kite_airspeed_temperature"] = log["airspeed_temperature"]
    print(
        "Note: kite_apparent_windspeed filled from est_app_wind_kite as a "
        "DIAGNOSTIC only (a GS model estimate, not a pitot measurement); "
        "excluded from the EKF via simulation_parameters_by_date in "
        "ekf_config.yaml. No angle-of-attack data."
    )

    # Wind-window angles (radians in the CSV). Azimuth sign flipped to match
    # the processed-data convention used by the 2019/2025 pre-processor.
    flight_data["kite_heading"] = log["heading"]
    flight_data["kite_elevation"] = log["elevation"]
    flight_data["kite_course"] = log["course"]
    flight_data["kite_azimuth"] = -log["azimuth"]

    # No working ground tether angle sensors: fall back to the kite angles,
    # mirroring the 2019/2025 pre-processor (unused unless the tether angle
    # measurements are switched on in ekf_config.yaml).
    flight_data["tether_azimuth_ground"] = log["azimuth"]
    flight_data["tether_elevation_ground"] = log["elevation"]

    # Time and date data.
    flight_data["time"] = log["time"] - log["time"].iloc[0]
    flight_data["time_of_day"] = log["time_of_day"]
    flight_data["unix_time"] = log["time"]
    flight_data["date"] = log["date"]
    flight_data["flight_phase_index"] = np.zeros(len(log))

    save_flight_data(flight_data, config_data, log_date)

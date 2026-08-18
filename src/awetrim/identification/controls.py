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

"""Control-input conversions shared by the identification scripts."""

from __future__ import annotations

import numpy as np

# Flight KCU convention used by the 2019 LEI-V3 logs:
#   steering: positive KCU steering corresponds to negative u_s.
#   depower: kcu=22 is powered, kcu=30 is depowered.
#
# Steering norms are PER FLIGHT (standardised u_s, calibrated 2026-08-12):
# u_s is defined so that the nominal 1.4 m gain times u_s IS the
# steering-tape half-difference. The 2019 KCU's kcu/100 command moves only
# HALF the nominal tape (its 1.4 m corresponds to the TOTAL left-right
# difference), so the 2019 logs divide by 200. Evidence: with this scaling
# the aerostructural trim grids reproduce the flight turn-rate channel
# across the turn states and the full right-lobe cloud (gk 0.314 identified
# vs 0.285 model, 1/m); the raw kcu/100 reading overstated the physical
# steering 2x. The 2025 rig is hypothesised to already be on the
# standardised scale (unverified). There is deliberately NO ambient default:
# every caller states which flight's convention it is converting.
FLIGHT_STEERING_KCU_NORM_2019: float = 200.0
FLIGHT_STEERING_KCU_NORM_2025: float = 100.0
# Deprecated alias (pre-calibration name); kept for old callers, 2019 value.
FLIGHT_STEERING_KCU_NORM: float = FLIGHT_STEERING_KCU_NORM_2019

FLIGHT_DEPOWER_POWERED_KCU: float = 22.0
FLIGHT_DEPOWER_POWERED_LDP_M: float = 1.7
FLIGHT_DEPOWER_DEPOWERED_KCU: float = 30.0
FLIGHT_DEPOWER_DEPOWERED_LDP_M: float = 2.1

# AS/QSM structural geometry baseline has depower tape l_dp = 1.5 m, so the
# 2019 powered/depowered states correspond to +0.2 m / +0.6 m actuation.
AS_POWER_TAPE_BASELINE_LDP_M: float = 1.5
AS_2019_POWERED_EXTENSION_M: float = (
    FLIGHT_DEPOWER_POWERED_LDP_M - AS_POWER_TAPE_BASELINE_LDP_M
)
AS_2019_DEPOWERED_EXTENSION_M: float = (
    FLIGHT_DEPOWER_DEPOWERED_LDP_M - AS_POWER_TAPE_BASELINE_LDP_M
)
ROM_POWERED_INPUT_DEPOWER: float = FLIGHT_DEPOWER_POWERED_LDP_M
ROM_DEPOWERED_INPUT_DEPOWER: float = FLIGHT_DEPOWER_DEPOWERED_LDP_M
ROM_NEUTRAL_INPUT_STEERING: float = 0.0


def flight_steering_to_us(
    kcu_actual_steering, norm: float = FLIGHT_STEERING_KCU_NORM_2019
):
    """Map flight ``kcu_actual_steering`` to standardised steering ``u_s``.

    ``norm`` is PER FLIGHT: 2019 logs divide by 200
    (``FLIGHT_STEERING_KCU_NORM_2019``, see the calibration note above); pass
    ``FLIGHT_STEERING_KCU_NORM_2025`` (=100) for the 2025 logs. The default is
    the 2019 convention, matching this module's 2019-anchored depower maps.
    """
    return -np.asarray(kcu_actual_steering, dtype=float) / float(norm)


def flight_depower_to_power_tape_length(kcu_actual_depower):
    """Map flight ``kcu_actual_depower`` to absolute power-tape length [m]."""
    kcu = np.asarray(kcu_actual_depower, dtype=float)
    slope = (FLIGHT_DEPOWER_DEPOWERED_LDP_M - FLIGHT_DEPOWER_POWERED_LDP_M) / (
        FLIGHT_DEPOWER_DEPOWERED_KCU - FLIGHT_DEPOWER_POWERED_KCU
    )
    return FLIGHT_DEPOWER_POWERED_LDP_M + (kcu - FLIGHT_DEPOWER_POWERED_KCU) * slope


def flight_dataframe_depower_to_power_tape_length(flight_data):
    """Map a flight-data table to ROM depower input ``u_p = l_dp`` [m].

    Prefer measured KCU depower. Fall back to the setpoint only if the measured
    column is unavailable. The legacy ``up`` column is intentionally not used:
    in the 2019 files it is a normalized EKF/control signal, not the physical
    power-tape length used by the identified ROM.
    """
    if "kcu_actual_depower" in flight_data:
        return flight_depower_to_power_tape_length(flight_data["kcu_actual_depower"])
    if "kcu_set_depower" in flight_data:
        return flight_depower_to_power_tape_length(flight_data["kcu_set_depower"])
    raise KeyError(
        "Need kcu_actual_depower or kcu_set_depower to build physical ROM depower."
    )


def flight_dataframe_steering_to_us(
    flight_data, norm: float = FLIGHT_STEERING_KCU_NORM_2019
):
    """Map a flight-data table to ROM steering input ``u_s``.

    Defaults to the 2019 steering norm -- consistent with the 2019-anchored
    depower map of :func:`flight_dataframe_depower_to_power_tape_length`.
    Pass ``norm=FLIGHT_STEERING_KCU_NORM_2025`` for 2025 tables.
    """
    if "kcu_actual_steering" in flight_data:
        return flight_steering_to_us(flight_data["kcu_actual_steering"], norm=norm)
    if "kcu_set_steering" in flight_data:
        return flight_steering_to_us(flight_data["kcu_set_steering"], norm=norm)
    raise KeyError(
        "Need kcu_actual_steering or kcu_set_steering to build physical ROM steering."
    )


def steering_extension_to_us(steering_extension_m, ref_travel_m: float = 1.0):
    """Map AS steering-tape extension [m] to identification steering ``u_s``."""
    return np.asarray(steering_extension_m, dtype=float) / float(ref_travel_m)


def power_tape_extension_to_length(
    power_tape_extension_m,
    baseline_ldp_m: float = AS_POWER_TAPE_BASELINE_LDP_M,
):
    """Map AS power-tape extension [m] to absolute power-tape length [m]."""
    return float(baseline_ldp_m) + np.asarray(power_tape_extension_m, dtype=float)

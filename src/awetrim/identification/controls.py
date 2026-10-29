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

# Hardware steering zero offset of the 2019 rig (KCU/bridle asymmetry), in
# standardised u_s: add it to the logged u_s to get the physical steering.
# Fitted on 2019 REEL-OUT (the strong, clean steering response; the reel-in
# fit absorbs the gravity/tether baseline turn) as a steering-tape offset of
# -16.45 mm = -0.01645 m / 1.4 m (states_2019 meta
# steering_asymmetry_tape_offset_m; the flown-state campaigns apply the same
# tape offset since 2026-09-13).
FLIGHT_STEERING_ZERO_OFFSET_2019: float = -0.01645 / 1.4

# The 2019 KCU depower signal at the powered (reel-out) and depowered
# (reel-in) states. Which power-tape length these are is NOT known from the
# logs: it is the model tape that reproduces the flight, recorded per kite in
# identification.yaml (fit: flight_match, :func:`flight_depower_match`) and
# redone with every identification. (Until 2026-10-29 a fixed 1.7 / 2.1 m
# stood here -- the match of the previous model, not a flown value.)
FLIGHT_DEPOWER_POWERED_KCU: float = 22.0
FLIGHT_DEPOWER_DEPOWERED_KCU: float = 30.0
ROM_NEUTRAL_INPUT_STEERING: float = 0.0


def rom_depower_band(rom_config=None) -> tuple[float, float]:
    """The ``(powered, depowered)`` depower inputs a kite's ROM was identified on.

    The ROM's ``u_p`` is whatever the identification fed it, and that differs
    per kite: the LEI-V3 uses the absolute power-tape length in metres
    (``ROM_POWERED_INPUT_DEPOWER`` .. ``ROM_DEPOWERED_INPUT_DEPOWER``), a kite
    identified on a logged normalised depower signal uses that signal's range.
    A kite states its band in its ``rom_config.yaml``::

        controls:
          input_depower:
            powered: 0.1
            depowered: 0.9

    ``rom_config`` is that file's path or its loaded dict; a ROM without the
    section raises (there is no kite-independent default).
    """
    cfg = rom_config
    if cfg is not None and not isinstance(cfg, dict):
        import yaml

        with open(cfg, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    band = ((cfg or {}).get("controls") or {}).get("input_depower") or {}
    if "powered" not in band or "depowered" not in band:
        raise ValueError(
            "the ROM states no controls.input_depower {powered, depowered} band"
        )
    return float(band["powered"]), float(band["depowered"])


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


def flight_depower_match(kite_dir, flight=None) -> tuple[float, float] | None:
    """``(reel_out, reel_in)``: the model power-tape lengths [m] that
    reproduce a flight's powered / depowered states, as the kite's
    ``identification.yaml`` records them (``fit: flight_match``, the
    default fidelity's). ``None`` without such a record, or when ``flight``
    (a year) is given and the record is of another flight.
    """
    from pathlib import Path

    from awetrim.identification.aerostructural_dataset import identification_settings

    path = Path(kite_dir) / "identification.yaml"
    if not path.exists():
        return None
    match = identification_settings(path, "fit")[1].get("flight_match")
    if not match or (flight is not None and str(match.get("flight")) != str(flight)):
        return None
    return float(match["reel_out"]), float(match["reel_in"])


def flight_depower_to_power_tape_length(kcu_actual_depower, match):
    """Map flight ``kcu_actual_depower`` to the ROM's power-tape length [m].

    ``match = (reel_out, reel_in)`` (:func:`flight_depower_match`): the model
    tapes that reproduce the flight's powered and depowered states; the KCU
    signal maps linearly through them.
    """
    if match is None:
        raise ValueError(
            "no flight_match for this flight: identification.yaml must record the "
            "model tapes that reproduce it (fit: flight_match)"
        )
    powered, depowered = float(match[0]), float(match[1])
    kcu = np.asarray(kcu_actual_depower, dtype=float)
    slope = (depowered - powered) / (
        FLIGHT_DEPOWER_DEPOWERED_KCU - FLIGHT_DEPOWER_POWERED_KCU
    )
    return powered + (kcu - FLIGHT_DEPOWER_POWERED_KCU) * slope


def flight_dataframe_depower_to_power_tape_length(flight_data, match):
    """Map a flight-data table to ROM depower input ``u_p = l_dp`` [m]
    (``match``: see :func:`flight_depower_to_power_tape_length`).

    Prefer measured KCU depower. Fall back to the setpoint only if the measured
    column is unavailable. The legacy ``up`` column is intentionally not used:
    in the 2019 files it is a normalized EKF/control signal, not the physical
    power-tape length used by the identified ROM.
    """
    if "kcu_actual_depower" in flight_data:
        return flight_depower_to_power_tape_length(flight_data["kcu_actual_depower"], match)
    if "kcu_set_depower" in flight_data:
        return flight_depower_to_power_tape_length(flight_data["kcu_set_depower"], match)
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


def power_tape_extension_to_length(power_tape_extension_m, baseline_ldp_m: float):
    """Map AS power-tape extension [m] to absolute power-tape length [m], from
    the tape the structural geometry is built at (``baseline_ldp_m``)."""
    return float(baseline_ldp_m) + np.asarray(power_tape_extension_m, dtype=float)

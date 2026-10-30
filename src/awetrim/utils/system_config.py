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

"""Accessors for the awesIO ``system_schema`` component layout.

The current awesIO ``main`` schema nests components as arrays
(``components.kites[0]`` / ``components.tethers[0]``). Earlier AWETrim configs
used a singular ``components.kite`` / ``components.tether`` (and a few inline-flat
configs put ``wing`` / ``control_system`` directly under ``components``). These
helpers return the single primary kite / tether dict for any of those layouts so
callers don't have to branch on the schema version.
"""

from __future__ import annotations

import warnings


def get_kite(system_config: dict) -> dict:
    """Return the primary kite component dict.

    Handles the current awesIO layout (``components.kites`` array), the legacy
    singular ``components.kite``, and an inline-flat layout (wing / control_system
    directly under ``components``). Returns an empty dict if nothing is found.
    """
    components = system_config.get("components", {})
    kites = components.get("kites")
    if isinstance(kites, list) and kites:
        return kites[0]
    return components.get("kite", components)


def get_control_system(system_config: dict) -> dict:
    """Return the primary kite's ``control_system`` dict (empty dict if none)."""
    return get_kite(system_config).get("control_system", {}) or {}


def get_turbine(system_config: dict) -> dict:
    """Return the ``structure`` of the KCU's onboard turbine (empty dict if none).

    awesIO ``control_system.turbine``: ``rotor_diameter``, ``mass`` and
    ``position_KA`` (hub in the kite's KA frame, origin at the KCU).

    The pre-awesIO flat block ``turbine: {diameter, thrust_coefficient}``
    (before 2026-10-29) is still read, with a DeprecationWarning, as a rotor
    diameter alone: its mass was inside the KCU mass, and C_T is the model's.
    """
    turbine = get_control_system(system_config).get("turbine") or {}
    if "structure" not in turbine and "diameter" in turbine:
        warnings.warn(
            "control_system.turbine {diameter, thrust_coefficient} is the "
            "pre-awesIO layout; write turbine.structure {rotor_diameter, mass, "
            "position_KA} and take the turbine mass out of the KCU mass.",
            DeprecationWarning,
            stacklevel=2,
        )
        return {"rotor_diameter": turbine["diameter"]}
    return turbine.get("structure", {}) or {}


def get_kcu_mass(system_config: dict) -> float | None:
    """Return the mass lumped at the KCU [kg], or None if the file states none.

    Everything the control unit carries: its own ``structure.mass`` plus the
    onboard turbine's ``mass``, which awesIO keeps separate (each component's
    mass is its own). The turbine is lumped at the KCU wherever it is mounted.
    """
    cs_struct = get_control_system(system_config).get("structure", {}) or {}
    if "mass" not in cs_struct:
        return None
    return float(cs_struct["mass"]) + float(get_turbine(system_config).get("mass", 0.0) or 0.0)


def get_tether(system_config: dict) -> dict:
    """Return the primary tether component dict (empty dict if none).

    Handles the current awesIO layout (``components.tethers`` array) and the
    legacy singular ``components.tether``.
    """
    components = system_config.get("components", {})
    tethers = components.get("tethers")
    if isinstance(tethers, list) and tethers:
        return tethers[0]
    return components.get("tether", {})


def get_ground_station(system_config: dict) -> dict:
    """Return the ground-station component dict (empty dict if none)."""
    components = system_config.get("components", {})
    station = components.get("ground_station")
    return station if isinstance(station, dict) else {}


def get_drum(system_config: dict) -> dict:
    """Return the primary winch drum dict of the ground station.

    The drum carries the winch drive envelope (``min_tether_speed`` /
    ``max_tether_speed`` / ``max_winch_acceleration`` / ``max_tether_force``,
    awesIO ``drum_object``). Older system files have no ``drums`` entry, in
    which case an empty dict is returned and callers fall back to their own
    numerical defaults.
    """
    drums = get_ground_station(system_config).get("drums")
    if isinstance(drums, list) and drums:
        return drums[0]
    return {}

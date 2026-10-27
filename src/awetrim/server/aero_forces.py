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

"""``POST /aero_forces``: the ROM's lift and drag at one state.

Stateless and independent of the optimization session, so it answers while a
solve runs. The kite of each (system file, ROM file) pair is built once and
kept; the physics is :func:`awetrim.system.rom_forces.rom_aero_forces`.
"""

import math
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml

from awetrim.system.factory import create_system_model_from_yaml, resolve_rom_config_path
from awetrim.system.rom_forces import rom_aero_forces
from awetrim.utils.config_paths import LEI_V3_SYSTEM_CONFIG


@lru_cache(maxsize=8)
def _kite_and_validity(system_path: str, rom_path: Optional[str]) -> Tuple[Any, dict]:
    """The Kite of a system/ROM pair and the ROM file's ``validity`` block."""
    kwargs = {"aero_yaml_path": rom_path} if rom_path else {}
    kite = create_system_model_from_yaml(yaml_path=system_path, **kwargs).kite
    resolved = resolve_rom_config_path(system_path, rom_path)
    validity = {}
    if resolved is not None and resolved.exists():
        with resolved.open("r", encoding="utf-8") as f:
            validity = (yaml.safe_load(f) or {}).get("validity") or {}
    return kite, validity


def _band(validity: dict, key: str) -> Optional[list]:
    band = validity.get(key)
    return [float(min(band)), float(max(band))] if band else None


def evaluate_aero_forces(request: Dict[str, Any]) -> Dict[str, Any]:
    """AeroForcesReply-shaped dict for an AeroForcesRequest-shaped dict."""
    system_path = Path(request.get("system_config_path") or LEI_V3_SYSTEM_CONFIG)
    if not system_path.exists():
        raise FileNotFoundError(f"config file not found: {system_path}")
    rom_request = request.get("rom_config_path")
    rom_path = resolve_rom_config_path(system_path, rom_request)
    if rom_request and not (rom_path and rom_path.exists()):
        raise FileNotFoundError(f"config file not found: {rom_path}")

    kite, validity = _kite_and_validity(
        str(system_path), str(rom_path) if rom_request else None
    )
    alpha_deg = request.get("angle_of_attack")
    result = rom_aero_forces(
        kite,
        request["input_depower"],
        request["speed_apparent_wind"],
        angle_of_attack=None if alpha_deg is None else math.radians(alpha_deg),
        input_steering=request.get("input_steering", 0.0),
        rho=request.get("rho"),
    )
    for key in ("angle_of_attack", "angle_pitch_bridle", "angle_pitch_tether"):
        if result[key] is not None:
            result[key] = math.degrees(result[key])

    band_depower = _band(validity, "input_depower")
    band_alpha = _band(validity, "angle_of_attack_deg")
    result["rom_config"] = str(rom_path) if rom_path else None
    result["input_depower_validity"] = band_depower
    result["angle_of_attack_validity"] = band_alpha
    result["within_validity"] = all(
        band is None or band[0] <= value <= band[1]
        for band, value in (
            (band_depower, float(request["input_depower"])),
            (band_alpha, result["angle_of_attack"]),
        )
    )
    return result

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

from pathlib import Path
from typing import Union

import yaml

from awetrim.environment.Wind import Wind
from awetrim.environment.profile_laws import DEFAULT_POWER_LAW_ALPHA
from awetrim.system.kite import Kite
from awetrim.system.system_model import SystemModel
from awetrim.system.tether import RigidLumpedTether
from awetrim.system.williams_tether import WilliamsTether
from awetrim.utils.system_config import get_kite, get_tether


def create_tether_from_config(
    tether_cfg: dict | None,
    *,
    diameter: float,
    density: float,
):
    """Instantiate a tether model from an ``as_config.yaml``-style block.

    ``tether_cfg`` may be ``None`` (defaults to ``RigidLumpedTether``) or a
    dict with at least a ``model`` key selecting the class. ``diameter`` and
    ``density`` come from ``system.yaml`` (awesIO) and are passed to every
    backend.
    """
    if not tether_cfg:
        return RigidLumpedTether(diameter=diameter, density=density)

    model = str(tether_cfg.get("model", "rigid_lumped")).lower()
    if model in ("rigid_lumped", "rigid", "lumped"):
        return RigidLumpedTether(diameter=diameter, density=density)
    if model == "williams":
        return WilliamsTether(
            diameter=diameter,
            density=density,
            n_elements=int(tether_cfg.get("n_elements", 20)),
            elastic=bool(tether_cfg.get("is_elastic", False)),
            cf=float(tether_cfg.get("cf", 0.01)),
        )
    raise ValueError(
        f"Unknown tether model '{tether_cfg.get('model')}'. "
        "Expected one of: 'rigid_lumped', 'williams'."
    )


def create_wind_model_from_config(wind_cfg):
    """Create a :class:`Wind` from a YAML ``wind`` section.

    Keys: ``model`` (or ``model_type``; any value in
    :data:`awetrim.environment.profile_laws.ALL_MODELS`, default ``uniform``),
    ``z0``, ``alpha``, ``jet_amplitude``/``jet_height``/``jet_width``,
    ``tabulated_heights``/``tabulated_speeds``, ``direction_wind`` (default 0)
    and exactly one amplitude:

    - ``speed_wind_ref`` (+ optional ``height_ref``, default 6 m),
    - ``speed_wind_at_100`` (shorthand for ``height_ref: 100``),
    - ``speed_wind_at_ref`` (+ ``height_ref``),
    - ``speed_friction`` (log-based models).

    No profile formula is evaluated here: ``Wind`` converts between the
    reference speed and the friction velocity through ``profile_laws``.
    """
    if not wind_cfg:
        return None

    wind_model = Wind(
        wind_model=wind_cfg.get("model", wind_cfg.get("model_type", "uniform")),
        z0=wind_cfg.get("z0", 0.01),
        alpha=wind_cfg.get("alpha", DEFAULT_POWER_LAW_ALPHA),
        jet_amplitude=wind_cfg.get("jet_amplitude", 0.0),
        jet_height=wind_cfg.get("jet_height"),
        jet_width=wind_cfg.get("jet_width"),
        tabulated_heights=wind_cfg.get("tabulated_heights"),
        tabulated_speeds=wind_cfg.get("tabulated_speeds"),
        direction_wind=wind_cfg.get("direction_wind", 0),
        speed_wind_ref=wind_cfg.get("speed_wind_ref"),
    )
    if "height_ref" in wind_cfg:
        wind_model.height_ref = float(wind_cfg["height_ref"])

    if "speed_friction" in wind_cfg:
        wind_model.speed_friction = wind_cfg["speed_friction"]
    elif "speed_wind_at_100" in wind_cfg:
        wind_model.height_ref = 100.0
        wind_model.speed_wind_ref = wind_cfg["speed_wind_at_100"]
    elif "speed_wind_at_ref" in wind_cfg:
        wind_model.speed_wind_ref = wind_cfg["speed_wind_at_ref"]

    return wind_model


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_aero_config_path(
    cfg: dict,
    config_path: Path | None = None,
    aero_yaml_path: Union[str, Path, None] = None,
) -> Path | None:
    if aero_yaml_path is not None:
        path = Path(aero_yaml_path)
        return path if path.is_absolute() else (config_path.parent / path)

    model_ref = cfg.get("models", {}).get("reduced_order", {}).get("aerodynamics")
    if model_ref is not None and config_path is not None:
        path = Path(model_ref)
        return path if path.is_absolute() else (config_path.parent / path)

    if config_path is not None:
        sibling = config_path.parent / "rom_config.yaml"
        if sibling.exists():
            return sibling
        legacy = config_path.parent / "aero_coeffs_rom.yaml"
        if legacy.exists():
            return legacy

    return None


def load_aero_input(
    cfg: dict,
    config_path: Union[str, Path, None] = None,
    aero_yaml_path: Union[str, Path, None] = None,
) -> dict:
    """Return ROM aerodynamic input from a standalone file or legacy inline config."""
    base_path = Path(config_path) if config_path is not None else None
    aero_path = _resolve_aero_config_path(cfg, base_path, aero_yaml_path)
    if aero_path is not None:
        aero_cfg = _load_yaml(aero_path)
        return aero_cfg.get("aerodynamics", aero_cfg)

    if "components" in cfg:
        kite = get_kite(cfg)
        return kite["wing"]["structure"].get("aerodynamics", {})
    return cfg.get("wing", {}).get("aerodynamics", {})


def _extract_params_awesio(cfg: dict) -> tuple:
    """Extract model parameters from an awesIO-format config dict."""
    try:
        from awesio.validator import validate
    except ModuleNotFoundError:
        validate = None

    if validate is not None:
        validate(cfg, restrictive=False)

    kite = get_kite(cfg)
    wing_struct = kite["wing"]["structure"]
    cs_struct = kite.get("control_system", {}).get("structure", {})
    tether_struct = get_tether(cfg).get("structure", {})

    mass_wing = wing_struct.get("mass", 20.0)
    area_wing = wing_struct.get("projected_surface_area", 20.0)
    mass_kcu = cs_struct.get("mass", 0.0)
    # KCU envelope, used for its bluff-body drag (awetrim.aerodynamics.
    # kcu_drag). Absent in older/partial system files -> 0.0, which the
    # drag model reads as 'no KCU drag' rather than failing.
    length_kcu = float(cs_struct.get("length", 0.0) or 0.0)
    diameter_kcu = float(cs_struct.get("diameter", 0.0) or 0.0)
    tether_diameter = tether_struct.get("diameter", 0.006)
    tether_density = tether_struct.get("density", 970.0)
    wind_cfg = cfg.get("wind", {})

    hardware_limits = _extract_hardware_limits(cs_struct, tether_struct)

    return (
        mass_wing,
        area_wing,
        mass_kcu,
        length_kcu,
        diameter_kcu,
        tether_diameter,
        tether_density,
        wind_cfg,
        hardware_limits,
    )


def _extract_hardware_limits(cs_struct: dict, tether_struct: dict) -> dict:
    """Build the optimizer's hardware-limit overrides from system.yaml.

    Maps the KCU actuator limits (control_system.structure.steering /
    .depower) onto the ``DEFAULT_OPTI_LIMITS`` keys the trajectory optimizer
    reads, plus the max tether length used to cap the radial distance. Keys are
    only present when the corresponding field exists in the config; missing
    ones fall back to ``DEFAULT_OPTI_LIMITS``.
    """
    hw: dict = {}
    steering = cs_struct.get("steering", {})
    if "range" in steering:
        hw["input_steering"] = tuple(steering["range"])
    if "rate" in steering:
        hw["steering_rate"] = tuple(steering["rate"])
    depower = cs_struct.get("depower", {})
    if "range" in depower:
        hw["input_depower"] = tuple(depower["range"])
    if "rate" in depower:
        hw["depower_rate"] = tuple(depower["rate"])
    # The radial distance (kite-to-ground chord) cannot exceed the tether, so
    # its upper bound is the max tether length; the lower bound stays the
    # numerical default. Carried as a sentinel resolved in opti_phase.
    tether_length = tether_struct.get("length")
    if tether_length is not None:
        hw["_max_tether_length"] = float(tether_length)
    return hw


def _extract_params_legacy(cfg: dict) -> tuple:
    """Extract model parameters from the legacy lei_v3_system_config.yaml format."""
    wing = cfg.get("wing", {})
    kcu_cfg = cfg.get("kcu", {})
    tether_cfg = cfg.get("tether", {})
    wind_cfg = cfg.get("wind", {})

    mass_wing = wing.get("mass", 20.0)
    area_wing = wing.get("area", 20.0)
    aero_cfg = wing.get("aerodynamics", {})
    mass_kcu = kcu_cfg.get("mass", 0.0)
    tether_diameter = tether_cfg.get("diameter", 0.006)
    tether_density = tether_cfg.get("density", 970.0)

    # The legacy format carries no KCU envelope -> no KCU drag.
    return (
        mass_wing,
        area_wing,
        aero_cfg,
        mass_kcu,
        0.0,
        0.0,
        tether_diameter,
        tether_density,
        wind_cfg,
    )


def create_system_model_from_yaml(
    yaml_path: Union[str, Path],
    steering_control: str = "asymmetric",
    aero_yaml_path: Union[str, Path, None] = None,
    tether_config: dict | None = None,
):
    """Create a SystemModel from a YAML configuration.

    Accepts either the awesIO system.yml format (auto-detected by the presence of a
    ``components`` key) or the legacy ``lei_v3_system_config.yaml`` format.

    awesIO format (preferred):
        metadata / assembly / components.kites[0].wing.structure.{projected_surface_area,
        mass} / components.kites[0].control_system.structure.mass /
        components.tethers[0].structure.{diameter, density}. The kite and tether are
        resolved via ``awetrim.utils.system_config.get_kite`` / ``get_tether``, which
        also accept the legacy singular ``components.kite`` / ``components.tether``
        layout. ROM aerodynamics are loaded from ``aero_yaml_path`` or from a sibling
        ``rom_config.yaml`` file (legacy: ``aero_coeffs_rom.yaml``).

    Legacy format:
        wing.{mass, area, aerodynamics} / kcu.mass / tether.{diameter, density}
    """
    config_path = Path(yaml_path)
    cfg = _load_yaml(config_path)

    if "components" in cfg:
        (
            mass_wing,
            area_wing,
            mass_kcu,
            length_kcu,
            diameter_kcu,
            tether_diameter,
            tether_density,
            wind_cfg,
            hardware_limits,
        ) = _extract_params_awesio(cfg)
    else:
        (
            mass_wing,
            area_wing,
            aero_cfg,
            mass_kcu,
            length_kcu,
            diameter_kcu,
            tether_diameter,
            tether_density,
            wind_cfg,
        ) = _extract_params_legacy(cfg)
        # Legacy format carries no actuator-limit block -> all-default limits.
        hardware_limits = {}

    print(
        f"Creating SystemModel with mass_wing={mass_wing}, area_wing={area_wing}, mass_kcu={mass_kcu}, tether_diameter={tether_diameter}, tether_density={tether_density}"
    )
    # 0.0 here means the system file carries no KCU envelope, so the trims
    # will run without KCU drag -- worth seeing in a run log.
    print(f"  KCU envelope: length={length_kcu} m, diameter={diameter_kcu} m")
    if "components" in cfg:
        aero_cfg = load_aero_input(
            cfg, config_path=config_path, aero_yaml_path=aero_yaml_path
        )

    tether = create_tether_from_config(
        tether_config, diameter=tether_diameter, density=tether_density
    )
    print(
        f"  tether model: {type(tether).__name__} "
        f"(diameter={tether_diameter}, density={tether_density})"
    )
    kite = Kite(
        mass_wing=mass_wing,
        mass_kcu=mass_kcu,
        length_kcu=length_kcu,
        diameter_kcu=diameter_kcu,
        area_wing=area_wing,
        aero_input=aero_cfg,
        steering_control=steering_control,
    )

    return SystemModel(
        dof=3,
        kite=kite,
        tether=tether,
        wind_model=create_wind_model_from_config(wind_cfg),
        hardware_limits=hardware_limits,
    )


def load_aero_input_from_system_config(
    cfg: dict,
    config_path: Union[str, Path, None] = None,
    aero_yaml_path: Union[str, Path, None] = None,
) -> dict:
    """Return the ROM aerodynamic input from system metadata or legacy inline config."""
    return load_aero_input(cfg, config_path=config_path, aero_yaml_path=aero_yaml_path)

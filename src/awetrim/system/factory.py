# Copyright (c) 2023-2026 Oriol Cayon, Delft University of Technology
#
# Licensed under the EUPL, Version 1.2 or - as soon they will be approved by
# the European Commission - subsequent versions of the EUPL (the "Licence");
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at:
#
#     https://joinup.ec.europa.eu/software/page/eupl
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the Licence is distributed on an "AS IS" basis,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the Licence for the specific language governing permissions and
# limitations under the Licence.
#
# SPDX-License-Identifier: EUPL-1.2

from pathlib import Path
import math
from typing import Union

import yaml

from awetrim.environment.Wind import Wind
from awetrim.system.kite import Kite
from awetrim.system.system_model import SystemModel
from awetrim.system.tether import RigidLumpedTether
from awetrim.system.williams_tether import WilliamsTether


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
    """Create a Wind model from a YAML wind section."""
    if not wind_cfg:
        return None

    wind_model = Wind(
        wind_model=wind_cfg.get("model", wind_cfg.get("model_type", "uniform")),
        z0=wind_cfg.get("z0", 0.01),
        tabulated_heights=wind_cfg.get("tabulated_heights"),
        tabulated_speeds=wind_cfg.get("tabulated_speeds"),
        direction_wind=wind_cfg.get("direction_wind", 0),
        speed_wind_ref=wind_cfg.get("speed_wind_ref"),
    )

    if "speed_friction" in wind_cfg:
        wind_model.speed_friction = wind_cfg["speed_friction"]
    elif "speed_wind_at_100" in wind_cfg:
        if wind_model.wind_model == "uniform":
            wind_model.speed_wind_ref = wind_cfg["speed_wind_at_100"]
        else:
            wind_model.speed_friction = (
                wind_model.kappa
                * wind_cfg["speed_wind_at_100"]
                / math.log(100 / wind_model.z0)
            )
    elif "speed_wind_at_ref" in wind_cfg:
        height_ref = wind_cfg.get("height_ref", wind_model.height_ref)
        wind_model.height_ref = height_ref
        if wind_model.wind_model == "uniform":
            wind_model.speed_wind_ref = wind_cfg["speed_wind_at_ref"]
        else:
            wind_model.speed_friction = (
                wind_model.kappa
                * wind_cfg["speed_wind_at_ref"]
                / math.log(height_ref / wind_model.z0)
            )

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
        kite = cfg["components"].get("kite", cfg["components"])
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

    components = cfg["components"]
    kite = components.get("kite", components)  # support both nested and flat layouts
    wing_struct = kite["wing"]["structure"]
    cs_struct = kite.get("control_system", {}).get("structure", {})
    tether_struct = components.get("tether", {}).get("structure", {})

    mass_wing = wing_struct.get("mass", 20.0)
    area_wing = wing_struct.get("projected_surface_area", 20.0)
    mass_kcu = cs_struct.get("mass", 0.0)
    tether_diameter = tether_struct.get("diameter", 0.006)
    tether_density = tether_struct.get("density", 970.0)
    wind_cfg = cfg.get("wind", {})

    return mass_wing, area_wing, mass_kcu, tether_diameter, tether_density, wind_cfg


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

    return (
        mass_wing,
        area_wing,
        aero_cfg,
        mass_kcu,
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
        metadata / assembly / components.wing.structure.{projected_surface_area, mass} /
        components.control_system.structure.mass / components.tether.structure.{diameter,
        density}. ROM aerodynamics are loaded from ``aero_yaml_path`` or from a sibling
        ``rom_config.yaml`` file (legacy: ``aero_coeffs_rom.yaml``).

    Legacy format:
        wing.{mass, area, aerodynamics} / kcu.mass / tether.{diameter, density}
    """
    config_path = Path(yaml_path)
    cfg = _load_yaml(config_path)

    if "components" in cfg:
        mass_wing, area_wing, mass_kcu, tether_diameter, tether_density, wind_cfg = (
            _extract_params_awesio(cfg)
        )
    else:
        (
            mass_wing,
            area_wing,
            aero_cfg,
            mass_kcu,
            tether_diameter,
            tether_density,
            wind_cfg,
        ) = _extract_params_legacy(cfg)

    print(
        f"Creating SystemModel with mass_wing={mass_wing}, area_wing={area_wing}, mass_kcu={mass_kcu}, tether_diameter={tether_diameter}, tether_density={tether_density}"
    )
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
        area_wing=area_wing,
        aero_input=aero_cfg,
        steering_control=steering_control,
    )

    return SystemModel(
        dof=3,
        kite=kite,
        tether=tether,
        wind_model=create_wind_model_from_config(wind_cfg),
    )


def load_aero_input_from_system_config(
    cfg: dict,
    config_path: Union[str, Path, None] = None,
    aero_yaml_path: Union[str, Path, None] = None,
) -> dict:
    """Return the ROM aerodynamic input from system metadata or legacy inline config."""
    return load_aero_input(cfg, config_path=config_path, aero_yaml_path=aero_yaml_path)

"""Per-kite inputs of the full-cycle scripts, read from the kite's data folder.

``fit_periodic_cycle_config.py`` and ``run_full_cycle_opti.py`` take
``--kite K``, where K is a folder under ``data/`` (``LEI-V3-KITE``) or a path
to any folder laid out the same way. Nothing about a kite lives in code: a new
kite -- including one whose data must stay off the repository -- needs only
three files in its folder.

  system.yaml        hardware (awesIO format): mass, tether, the drum envelope
                     (reel speeds, acceleration, force rating) and the KCU
                     actuator ranges/rates -> the optimizer's hardware limits
  ROM config         the aerodynamic ROM the system file selects
                     (models.reduced_order.aerodynamics, else a sibling
                     rom_config.yaml; awetrim.system.factory.
                     resolve_rom_config_path), plus two blocks the cycle needs:
                       controls.input_depower: {powered, depowered}  REQUIRED
                         the depower band the ROM was identified on, in the
                         ROM's own u_p unit (V3: power-tape metres)
                       validity.angle_of_attack_deg: [lo, hi]        optional
                         the AoA range the ROM was identified on; replaces
                         the optimizer's default AoA bound
  cycle_profile.yaml what the cycle scripts need on top (see
                     data/LEI-V3-KITE/cycle_profile.yaml for an annotated one):
                       wind       REQUIRED  the constant profile the seed is
                                  checked at and the cycle optimized in
                                  (kwargs of run_full_cycle_opti.build_wind_model)
                       winch_law  REQUIRED  T = slope * (v_r - offset(u_p)),
                                  offset(u_p) = offset0 + gain * (u_p - ref);
                                  it marches the seed's forward simulation
                       seed       REQUIRED  r0, the cycle-duration prior and
                                  the figure-eight / reel-in size; any other
                                  fit_periodic_cycle_config.ARTIFICIAL knob
                                  may be added to override its default
                       m_per_second, min_turn_radius, flight   optional

The seed YAML is written to / read from ``<kite>/cycle_configs/`` and the
optimizer's outputs go to ``results/<kite folder name>/optimization/full_cycle``.
"""

from __future__ import annotations

import math
from pathlib import Path

import yaml

from awetrim.identification.controls import rom_depower_band
from awetrim.system.factory import resolve_rom_config_path
from awetrim.utils.config_paths import DATA_DIR

DEFAULT_KITE = "LEI-V3-KITE"
PROFILE_FILENAME = "cycle_profile.yaml"
SEED_FILENAME = "full_cycle_periodic_from_exp.yaml"

REQUIRED_WINCH_LAW = (
    "slope_winch_ro",
    "offset_winch_ro",
    "winch_offset_depower_gain",
    "winch_depower_ref",
    "min_tether_force",
)
# r0 / cycle_duration_s are consumed here; the rest are ARTIFICIAL knobs that
# carry no default in fit_periodic_cycle_config (the size of the pattern is a
# property of the kite, not of the method).
REQUIRED_SEED = (
    "r0",
    "cycle_duration_s",
    "reelout_fraction",
    "beta0",
    "beta_amp0",
    "az_amp0",
    "beta_reelin_peak",
)
_SECTIONS = {"wind", "winch_law", "seed", "m_per_second", "min_turn_radius", "flight"}


def kite_dir(kite: str | Path | None = None) -> Path:
    """The kite's data folder: a folder under ``data/`` or a path to one."""
    kite = kite or DEFAULT_KITE
    for candidate in (Path(kite), DATA_DIR / str(kite)):
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"Kite {kite!r} is neither a folder nor a folder under {DATA_DIR}"
    )


def _require(mapping: dict, keys, where: Path, section: str) -> None:
    missing = [k for k in keys if k not in mapping]
    if missing:
        raise ValueError(f"{where}: {section} is missing {missing}")


def _load_yaml(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found; a kite folder needs system.yaml, the ROM config "
            f"it selects and {PROFILE_FILENAME} (see cycle_kites.py)"
        )
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_kite(kite: str | Path | None = None) -> dict:
    """Everything the cycle scripts need about one kite, validated.

    Missing required entries and unknown top-level keys raise instead of
    falling back on another kite's numbers.
    """
    folder = kite_dir(kite)
    profile_path = folder / PROFILE_FILENAME
    profile = _load_yaml(profile_path)
    system_path = folder / "system.yaml"
    _load_yaml(system_path)  # existence check; parsed by the scripts
    # The SAME file the system model is built with (factory resolution).
    rom_path = resolve_rom_config_path(system_path) or folder / "rom_config.yaml"
    rom = _load_yaml(rom_path)

    unknown = sorted(set(profile) - _SECTIONS)
    if unknown:
        raise ValueError(f"{profile_path}: unknown keys {unknown}; allowed {sorted(_SECTIONS)}")
    _require(profile, ("wind", "winch_law", "seed"), profile_path, "the profile")
    _require(profile["wind"], ("model_type",), profile_path, "wind")
    _require(profile["winch_law"], REQUIRED_WINCH_LAW, profile_path, "winch_law")
    _require(profile["seed"], REQUIRED_SEED, profile_path, "seed")
    # The cycle never assumes a depower unit: without the band a V3 fallback
    # would silently put another kite's profile in power-tape metres.
    band = (rom.get("controls") or {}).get("input_depower") or {}
    _require(band, ("powered", "depowered"), rom_path, "controls.input_depower")

    seed = dict(profile["seed"])
    r0 = float(seed.pop("r0"))
    cycle_duration_s = float(seed.pop("cycle_duration_s"))

    opti_limits_override = {}
    aoa = (rom.get("validity") or {}).get("angle_of_attack_deg")
    if aoa is not None:
        opti_limits_override["angle_of_attack"] = [math.radians(float(a)) for a in aoa]

    flight = dict(profile.get("flight") or {})
    if "h5" in flight and not Path(flight["h5"]).is_absolute():
        flight["h5"] = folder / flight["h5"]
    if flight.get("loader") == "ekf_results":
        flight.setdefault("path_to_main", str(folder))

    return {
        "name": folder.name,
        "dir": folder,
        "system_config": system_path,
        "rom_config": rom_path,
        "cycle_config_dir": folder / "cycle_configs",
        "results_dir": Path("results") / folder.name / "optimization" / "full_cycle",
        "depower_band": rom_depower_band(rom),
        "wind": dict(profile["wind"]),
        "winch_law": dict(profile["winch_law"]),
        "r0": r0,
        "cycle_duration_s": cycle_duration_s,
        "artificial": seed,
        "opti_limits_override": opti_limits_override,
        "m_per_second": profile.get("m_per_second"),
        "min_turn_radius": profile.get("min_turn_radius"),
        "flight": flight,
    }


def wind_tag(wind: dict) -> str:
    """Filename tag identifying the wind a cycle was solved at."""
    if wind["model_type"] == "tabulated":
        heights, speeds = wind["heights"], wind["speeds"]
        i = min(range(len(heights)), key=lambda k: abs(heights[k] - 200.0))
        return f"wind_tab_{speeds[i]:g}at{heights[i]:g}m"
    tag = f"wind_{wind['speed_wind_ref']:g}at{wind['height_ref']:g}m_{wind['model_type']}"
    if "z0" in wind:
        tag += f"_z0_{wind['z0']:g}"
    return tag

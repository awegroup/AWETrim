"""Per-kite inputs of the reel-out examples, read from the kite's data folder.

``downloop_pattern.py`` and ``helix_pattern.py`` take ``--kite K``, where K is
a folder under ``data/`` (default ``LEI-V3-KITE``) or a path to any folder laid
out the same way. Each reads

  system.yaml                        hardware (``--system`` another file there)
  the ROM system.yaml selects        (``--rom`` another file there)
  cycle_configs/<pattern>.yaml       the phase config (``downloop_spline.yaml``,
                                     ``helix_spline.yaml``); an optional
                                     top-level ``example: seed_curve:`` block
                                     overrides keys of the script's
                                     INITIAL_GUESS (the size of the seed
                                     figure, e.g. ``r0``, ``az_amp0``), which
                                     ``size_reelout_seed`` then fits to the
                                     turns the kite holds
  identification.yaml (optional)     ``fit: flight_match``: the seed flies its
                                     reel-out tape; without a record, the
                                     config's own ``sim_parameters.input_depower``

Results go to ``results/<kite folder name>/optimization/<pattern>``.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from awetrim.identification.controls import flight_depower_match
from awetrim.utils.config_paths import DATA_DIR

DEFAULT_KITE = "LEI-V3-KITE"

#: The short names the LEI-V3 examples gave its files (--system / --rom).
FILE_ALIASES = {
    "optimisation": "system.yaml",
    "flown_2019": "system_flown_2019.yaml",
    "aerostructural": "rom_config_aerostructural.yaml",
    "aerostructural_flight": "rom_config_aerostructural_flight_corrected.yaml",
}

_SEED_KEYS = {
    "curve_type", "M", "n_fit", "s_init", "s_final", "r0",
    "az_amp0", "beta0", "beta_amp0", "downloops",
}


def kite_dir(kite: str | Path | None = None) -> Path:
    """The kite's data folder: a folder under ``data/`` or a path to one."""
    kite = kite or DEFAULT_KITE
    for candidate in (Path(kite), DATA_DIR / str(kite)):
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"Kite {kite!r} is neither a folder nor a folder under {DATA_DIR}")


def kite_file(folder: Path, name: str | Path) -> Path:
    """A file of the kite: an alias of FILE_ALIASES, a path, or a file in its folder."""
    name = FILE_ALIASES.get(str(name), name)
    for candidate in (Path(name), Path(folder) / name):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"{name} is neither a file nor a file in {folder}")


def pattern_config_path(folder: Path, filename: str) -> Path:
    path = Path(folder) / "cycle_configs" / filename
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found; the reel-out examples read their phase config from "
            "the kite's cycle_configs/ (see reelout_kites.py)"
        )
    return path


def seed_overrides(config_path: Path) -> dict:
    """The config's optional ``example: seed_curve:`` block, validated."""
    with Path(config_path).open("r", encoding="utf-8") as f:
        example = (yaml.safe_load(f) or {}).get("example") or {}
    unknown = sorted(set(example) - {"seed_curve"})
    if unknown:
        raise ValueError(f"{config_path}: example has unknown keys {unknown}; allowed ['seed_curve']")
    curve = dict(example.get("seed_curve") or {})
    unknown = sorted(set(curve) - _SEED_KEYS)
    if unknown:
        raise ValueError(f"{config_path}: example.seed_curve has unknown keys {unknown}")
    return curve


def apply_flight_depower(folder: Path, sim_parameters: dict) -> None:
    """The seed's depower: the kite's recorded flight-match reel-out tape, if any."""
    match = flight_depower_match(folder)
    if match is not None:
        sim_parameters["input_depower"] = match[0]

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

"""AWETrim-aware replacement for awes_ekf.setup.settings.

Provides the same interface as awes_ekf.setup.settings but resolves config
files from this repo's data/ layout (data/<kite>/ekf_config/*.yaml and
data/<kite>/*_config.yaml) instead of the hardcoded data/config/ path.

Re-exports SimulationConfig, TuningParameters, and validate_config from the
installed awes_ekf package so callers only need to change the import line.

Output convention: results/<kite_name>/ekf/<model>_<YYYY>-<MM>-<DD>.h5
matching the repo-wide results/<kite_name>/<analysis_type>/ pattern.
"""

from __future__ import annotations

import h5py
import pandas as pd
from pathlib import Path

import yaml

from awes_ekf.setup.settings import (  # re-export unchanged classes
    SimulationConfig,
    TuningParameters,
    validate_config,
)

from awetrim.utils.system_config import get_kite, get_tether

_EKF_REQUIRED_KEYS = {"simulation_parameters", "tuning_parameters"}
_KITE_NAME_KEY = "_awetrim_kite_name"

# Root of the AWETrim project (src/awetrim/experimental/ → three parents up).
_PROJECT_ROOT = Path(__file__).resolve().parents[3]


def load_config(project_dir: Path | None = None) -> dict:
    """Interactively load and merge kite configuration from a folder.

    Prompts the user for a folder path and loads both:
    - ekf_config.yaml (simulation_parameters, tuning_parameters)
    - a system YAML chosen from the folder's ``system*.yaml`` files (physical
      properties from the system structure). With several candidates (e.g.
      ``system.yaml`` with the 8.4 kg optimisation KCU vs ``system_flown_2019.yaml``
      with the 22.75 kg flown one) the user picks which system the EKF should
      assume; a single candidate is used directly.

    Merges them into a single config dict with kite, kcu, tether properties
    automatically extracted from the chosen system YAML.

    Injects _awetrim_kite_name into the returned dict so save_ekf_results can
    place output under results/<kite_name>/ekf/ without extra arguments.
    """
    from prompt_toolkit import prompt
    from prompt_toolkit.completion import PathCompleter

    # Prompt for config folder
    path_completer = PathCompleter(expanduser=True)
    config_folder_str = prompt(
        "Enter the path to the config folder (containing system.yaml and ekf_config.yaml): ",
        completer=path_completer,
    ).strip()

    config_folder = Path(config_folder_str).expanduser().resolve()
    if not config_folder.exists():
        raise FileNotFoundError(f"Folder does not exist: {config_folder}")

    # Look for both files
    ekf_config_path = config_folder / "ekf_config.yaml"
    if not ekf_config_path.exists():
        raise FileNotFoundError(f"ekf_config.yaml not found in {config_folder}")

    # Every system variant in the folder is a candidate (system.yaml,
    # system_flown_2019.yaml, ...); the physical properties the EKF assumes -- KCU
    # mass above all -- depend on which one is chosen.
    candidates = sorted(
        set(config_folder.glob("system*.yaml")) | set(config_folder.glob("system*.yml"))
    )
    if not candidates:
        raise FileNotFoundError(f"No system*.yaml / system*.yml in {config_folder}")
    if len(candidates) == 1:
        system_yaml_path = candidates[0]
    else:
        print("Available system configuration files:")
        for index, path in enumerate(candidates, start=1):
            print(f"{index}: {path.name}")
        selection = (
            input(f"Select a system file (1-{len(candidates)}) [default: 1]: ").strip()
            or "1"
        )
        system_yaml_path = candidates[int(selection) - 1]
    print(f"Using system config: {system_yaml_path.name}")

    # Load EKF config
    with ekf_config_path.open("r", encoding="utf-8") as fh:
        config_data = yaml.safe_load(fh)

    if not _EKF_REQUIRED_KEYS.issubset(config_data.keys()):
        raise ValueError(
            f"ekf_config.yaml missing required keys. Expected: {', '.join(sorted(_EKF_REQUIRED_KEYS))}"
        )

    # Load system.yaml and extract physical properties
    with system_yaml_path.open("r", encoding="utf-8") as fh:
        system_config = yaml.safe_load(fh)

    # Extract from the kite hierarchy (awesIO components.kites[0] / legacy kite)
    kite_node = get_kite(system_config)
    wing_struct = kite_node.get("wing", {}).get("structure", {})
    bridle_struct = kite_node.get("bridle", {}).get("structure", {})
    control_sys_struct = kite_node.get("control_system", {}).get("structure", {})

    tether_struct = get_tether(system_config).get("structure", {})

    # Build kite config
    kite_config = {
        "model_name": kite_node.get("name", "unknown"),
        "mass": wing_struct.get("mass", 0.0),
        "area": wing_struct.get(
            "projected_surface_area", wing_struct.get("planform_surface_area", 0.0)
        ),
        "span": wing_struct.get("span", 0.0),
        "sensor_ids": [0, 1],  # Default sensor IDs
    }

    # Build kcu config
    kcu_config = {
        "length": control_sys_struct.get("length", 1.0),
        "diameter": control_sys_struct.get("diameter", 0.48),
        "mass": control_sys_struct.get("mass", 0.0),
        # bridle_point_node is the body-frame ORIGIN in awesIO (always [0,0,0]),
        # so the KCU-to-wing distance is the wing CG height above it. Reading
        # the node itself gave 0 and collapsed the bridle segment in the tether
        # model (bridle drag lost, KCU drag split corrupted).
        "distance_kcu_kite": wing_struct.get("center_of_mass", [0.0, 0.0, 0.0])[2],
        "total_length_bridle_lines": bridle_struct.get(
            "total_nominal_line_length", 0.0
        ),
        "diameter_bridle_lines": bridle_struct.get("avg_line_diameter", 0.0),
    }

    # Build tether config
    tether_config = {
        "material_name": tether_struct.get("material", {}).get("type", "Dyneema-SK78"),
        "diameter": tether_struct.get("diameter", 0.01),
        "n_elements": 30,  # Default; adjust if needed
    }

    # Merge system properties into config
    config_data["kite"] = kite_config
    config_data["kcu"] = kcu_config
    config_data["tether"] = tether_config

    # Derive kite name from folder name (e.g., data/LEI-V3-KITE → LEI-V3-KITE)
    kite_name = config_folder.name
    config_data[_KITE_NAME_KEY] = kite_name
    # Record WHICH system variant the EKF assumed; it ends up in the results
    # h5 config_data group, so a flown-vs-optimisation run is identifiable.
    config_data["system_yaml_used"] = system_yaml_path.name

    print(f"EKF config loaded from: {ekf_config_path}")
    print(f"System config loaded from: {system_yaml_path}")
    print(f"Kite model: {kite_config['model_name']}")
    return config_data


def save_ekf_results(
    ekf_output_df: pd.DataFrame,
    flight_data: pd.DataFrame,
    kite_model: str,
    year: str,
    month: str,
    day: str,
    config_data: dict,
    addition: str = "",
    project_dir: Path | None = None,
) -> Path:
    """Save EKF results to results/<kite_name>/ekf/<model>_<date>.h5.

    Matches the repo convention results/<kite_name>/<analysis>/ and uses an
    absolute path from the project root rather than the CWD.

    Returns the path of the written file.
    """
    root = Path(project_dir) if project_dir is not None else _PROJECT_ROOT
    kite_name = config_data.get(_KITE_NAME_KEY, kite_model)

    out_dir = root / "results" / kite_name / "ekf"
    out_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{kite_model}_{year}-{month}-{day}{addition}.h5"
    h5_path = out_dir / filename

    def _encode_strings(df: pd.DataFrame) -> pd.DataFrame:
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].astype("S")
        return df

    def _sanitize_columns(df: pd.DataFrame) -> pd.DataFrame:
        df.columns = (
            df.columns.str.replace(" ", "_")
            .str.replace("(", "")
            .str.replace(")", "")
            .str.replace("/", "_")
            .str.replace("°", "deg")
        )
        return df

    ekf_output_df = _encode_strings(ekf_output_df.copy())
    flight_data = _sanitize_columns(_encode_strings(flight_data.copy()))

    def _save_dict(group: h5py.Group, d: dict) -> None:
        for key, value in d.items():
            if key.startswith("_"):
                continue  # skip internal AWETrim keys
            if isinstance(value, dict):
                _save_dict(group.create_group(key), value)
            else:
                group.attrs[key] = (
                    value.encode("utf-8") if isinstance(value, str) else value
                )

    with h5py.File(h5_path, "w") as hf:
        ekf_group = hf.create_group("ekf_output")
        ekf_group.attrs["description"] = (
            "Extended Kalman Filter output, including system parameters derived "
            "from postprocessing the EKF state vector with experimental data."
        )
        for col in ekf_output_df.columns:
            ekf_group.create_dataset(col, data=ekf_output_df[col].values)

        flight_group = hf.create_group("flight_data")
        flight_group.attrs["description"] = (
            "Experimental data collected during the flight test. "
            "Offsets are applied to orientation data and tether length."
        )
        for col in flight_data.columns:
            flight_group.create_dataset(col, data=flight_data[col].values)

        config_group = hf.create_group("config_data")
        config_group.attrs["description"] = (
            "Configuration data used for the simulation and postprocessing."
        )
        _save_dict(config_group, config_data)

    print(f"EKF results saved to: {h5_path.relative_to(root)}")
    return h5_path

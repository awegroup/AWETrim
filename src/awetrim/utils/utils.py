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

import casadi as ca
import h5py
import pandas as pd


def skew_symmetric(v):
    return ca.vertcat(
        ca.horzcat(0, -v[2], v[1]),
        ca.horzcat(v[2], 0, -v[0]),
        ca.horzcat(-v[1], v[0], 0),
    )


def calculate_angle_2vec(vector_a, vector_b, reference_vector=None):

    dot_product = ca.dot(vector_a, vector_b)
    magnitude_a = ca.norm_2(vector_a)
    magnitude_b = ca.norm_2(vector_b)

    cos_theta = dot_product / (magnitude_a * magnitude_b)
    angle_rad = ca.arccos(cos_theta)

    return angle_rad


def read_dict_from_group(group):
    config_dict = {}
    for key, value in group.attrs.items():
        if isinstance(value, bytes):
            value = value.decode("utf-8")  # Decode byte strings back to regular strings
        config_dict[key] = value

    for subgroup_name in group:
        subgroup = group[subgroup_name]
        config_dict[subgroup_name] = read_dict_from_group(subgroup)

    return config_dict


def read_ekf_results(year, month, day, kite_model, addition="", path_to_main=""):
    path = "data/LEI-V9-KITE/flight_logs/"
    date = str(year) + "-" + str(month) + "-" + str(day)
    file_name = str(kite_model) + "_" + date
    hdf5_path = path_to_main + path + file_name + addition + ".h5"
    ekf_output_df, flight_data_df, config_data = read_results_from_hdf5(hdf5_path)
    return ekf_output_df, flight_data_df, config_data


def read_results_from_hdf5(hdf5_path):
    with h5py.File(hdf5_path, "r") as hf:
        # Read the ekf_output_df DataFrame
        ekf_group = hf["ekf_output"]
        ekf_output_df = pd.DataFrame(
            {
                col: (
                    ekf_group[col][:].astype(str)
                    if ekf_group[col].dtype.kind == "S"
                    else ekf_group[col][:]
                )
                for col in ekf_group.keys()
            }
        )

        # Read the flight_data DataFrame
        flight_group = hf["flight_data"]
        flight_data_df = pd.DataFrame(
            {
                col: (
                    flight_group[col][:].astype(str)
                    if flight_group[col].dtype.kind == "S"
                    else flight_group[col][:]
                )
                for col in flight_group
                if isinstance(flight_group[col], h5py.Dataset)
            }
        )

        # Read config_data
        config_group = hf["config_data"]
        config_data = read_dict_from_group(config_group)

    return ekf_output_df, flight_data_df, config_data


def load_cycle_config_from_yaml(yaml_path):
    """Load pumping cycle configuration from YAML file.

    Reads a YAML configuration file containing wind, reelout, and reel-in parameters
    and returns them as dictionaries suitable for use in simulation scripts.

    Parameters
    ----------
    yaml_path : str or Path
        Path to the YAML configuration file.

    Returns
    -------
    tuple[dict, dict]
        A tuple of (REELOUT_CONFIG, REELIN_CONFIG) dictionaries.
        - REELOUT_CONFIG contains pattern_type, path_parameters, radial_parameters, sim_parameters
        - REELIN_CONFIG contains reel-in phase parameters (if available)

    Examples
    --------
    >>> from pathlib import Path
    >>> reelout_cfg, reelin_cfg = load_cycle_config_from_yaml("data/LEI-V3-KITE/cycle_configs/downloop_spline.yaml")
    """
    import yaml
    import numpy as np
    from pathlib import Path

    config_path = Path(yaml_path)
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Extract reelout configuration
    reelout_section = cfg.get("reelout", {})

    # Path parameters - convert lists to numpy arrays where needed
    path_params = reelout_section.get("path_parameters", {})
    path_parameters = path_params

    # Radial parameters
    radial_params = reelout_section.get("radial_parameters", {})
    radial_parameters = {
        "reeling_strategy": radial_params.get("reeling_strategy", "force"),
        "force_model": radial_params.get("force_model", "quadratic"),
        "reeling_speed": radial_params.get("reeling_speed", 0.0),
        "max_tether_force": radial_params.get("max_tether_force", 8400),
        "min_tether_force": radial_params.get("min_tether_force", 1500),
        "softplus": radial_params.get("softplus", True),
        "softplus_beta": radial_params.get("softplus_beta", 1e-4),
        "softminus": radial_params.get("softminus", True),
        "softminus_beta": radial_params.get("softminus_beta", 1e-3),
        "slope_winch_ro": radial_params.get("slope_winch_ro", 1500),
        "offset_winch_ro": radial_params.get("offset_winch_ro", 0),
    }

    # Simulation parameters
    sim_params = reelout_section.get("sim_parameters", {})
    sim_parameters = {
        "start_time": sim_params.get("start_time", 0),
        "end_time": sim_params.get("end_time", 35),
        "start_angle": sim_params.get("start_angle", np.pi / 2),
        "end_angle": sim_params.get("end_angle", np.pi / 2),
        "n_points": sim_params.get("n_points", 200),
        "input_depower": sim_params.get("input_depower", 0.0),
    }

    # Assemble REELOUT_CONFIG
    REELOUT_CONFIG = {
        "pattern_type": reelout_section.get("pattern_type", "cst_helix"),
        "path_parameters": path_parameters,
        "radial_parameters": radial_parameters,
        "sim_parameters": sim_parameters,
    }

    # Extract reel-in configuration
    reelin_section = cfg.get("reelin", {})

    # Reel-in path parameters
    reelin_path_params = reelin_section.get("path_parameters", {})
    reelin_path_parameters = {
        "elevation_start_ri": reelin_path_params.get(
            "elevation_start_ri", np.radians(30)
        ),
        "elevation_start_riro": reelin_path_params.get(
            "elevation_start_riro", np.radians(70)
        ),
        "elevation_start_ro": reelin_path_params.get(
            "elevation_start_ro", np.radians(30)
        ),
        "distance_radial_start": reelin_path_params.get("distance_radial_start", 360),
        "distance_radial_end": reelin_path_params.get("distance_radial_end", 230),
    }

    # Reel-in radial parameters
    reelin_radial_params = reelin_section.get("radial_parameters", {})
    reelin_radial_parameters = {
        "reeling_strategy": reelin_radial_params.get("reeling_strategy", "force"),
        "force_model": reelin_radial_params.get("force_model", "quadratic"),
        "reeling_speed": reelin_radial_params.get("reeling_speed", 1.0),
        "max_tether_force": reelin_radial_params.get("max_tether_force", 8400.0),
        "min_tether_force": reelin_radial_params.get("min_tether_force", 2000.0),
        "softplus": reelin_radial_params.get("softplus", True),
        "softplus_beta": reelin_radial_params.get("softplus_beta", 1e-4),
        "softminus": reelin_radial_params.get("softminus", True),
        "softminus_beta": reelin_radial_params.get("softminus_beta", 1e-3),
        "slope_winch_ri": reelin_radial_params.get("slope_winch_ri", 562),
        "offset_winch_ri": reelin_radial_params.get("offset_winch_ri", -5),
    }

    # Reel-in simulation parameters
    reelin_sim_params = reelin_section.get("sim_parameters", {})
    reelin_sim_parameters = {
        "start_time": reelin_sim_params.get("start_time", 0),
        "n_points": reelin_sim_params.get("n_points", 100),
        "n_points_ri": reelin_sim_params.get(
            "n_points_ri", reelin_sim_params.get("n_points", 100)
        ),
        "n_points_riro": reelin_sim_params.get(
            "n_points_riro", reelin_sim_params.get("n_points", 100)
        ),
    }

    # Assemble REELIN_CONFIG
    REELIN_CONFIG = {
        "path_parameters": reelin_path_parameters,
        "radial_parameters": reelin_radial_parameters,
        "sim_parameters": reelin_sim_parameters,
    }

    return REELOUT_CONFIG, REELIN_CONFIG

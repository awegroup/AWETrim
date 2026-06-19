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

"""Aerodynamic analysis interfaces for AWETrim."""

from awetrim.aerodynamics.parametric_airfoil import (
    LEI_airfoil,
    generate_profile,
    reading_profile_from_airfoil_dat_files,
    save_profile_as_dat_file,
)
from awetrim.aerodynamics.parametric_geometry import (
    WingSections,
    morph_wing,
    morph_wing_to,
)
from awetrim.aerodynamics.vsm_adapter import VSMAeroModelAdapter
from awetrim.aerodynamics.vsm_quasi_steady import (
    DEFAULT_AXES,
    DEFAULT_BOUNDS_LOWER,
    DEFAULT_BOUNDS_UPPER,
    DEFAULT_TRANSFORMATION_C_FROM_VSM,
    compute_vsm_trim_stability_derivatives,
    plot_vsm_quasi_steady_sweep,
    run_vsm_quasi_steady_sweep,
    solve_vsm_quasi_steady_trim,
    vsm_quasi_steady_sweep_to_dataframe,
)

__all__ = [
    "VSMAeroModelAdapter",
    "WingSections",
    "morph_wing",
    "morph_wing_to",
    "LEI_airfoil",
    "generate_profile",
    "reading_profile_from_airfoil_dat_files",
    "save_profile_as_dat_file",
    "DEFAULT_AXES",
    "DEFAULT_BOUNDS_LOWER",
    "DEFAULT_BOUNDS_UPPER",
    "DEFAULT_TRANSFORMATION_C_FROM_VSM",
    "compute_vsm_trim_stability_derivatives",
    "plot_vsm_quasi_steady_sweep",
    "run_vsm_quasi_steady_sweep",
    "solve_vsm_quasi_steady_trim",
    "vsm_quasi_steady_sweep_to_dataframe",
]

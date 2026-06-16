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

"""Aerodynamic analysis interfaces for AWETrim."""

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

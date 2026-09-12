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

"""The coupled solve itself: geometry, load transfer, and the fixed-point driver.

Backend-agnostic. ``coupled_solver`` dispatches over the structural backends in
``..wireframe`` and ``..billow``; ``read_struc_geometry_yaml`` is the one
geometry reader both of them consume, and ``aero2struc`` is the one load
transfer. A new backend is a branch in the driver, not a copy of it.
"""

from awetrim.aerostructural.coupled import (
    aero2struc,
    coupled_solver,
    read_struc_geometry_yaml,
)

__all__ = [
    "aero2struc",
    "coupled_solver",
    "read_struc_geometry_yaml",
]

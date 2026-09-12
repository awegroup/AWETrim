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

"""Element kernels: strain energy of one element, mapped over a set."""

from .base import ElementKernel, ElementSet
from .beam import (
    BeamSection,
    TimoshenkoBeamKernel,
    beam_strains,
    build_beam_elements,
    initial_frames_from_polyline,
)
from .inflatable import (
    InflatableBeamKernel,
    InflatableTubeLaw,
    build_inflatable_beam_elements,
    inflatable_beam_state,
)
from .cable import (
    CableKernel,
    PulleyKernel,
    build_cable_elements,
    build_pulley_elements,
    line_tensions,
)
from .membrane import (
    SLACK,
    TAUT,
    WRINKLED,
    MembraneKernel,
    build_membrane_elements,
    membrane_reference,
    membrane_regimes,
)

__all__ = [
    "ElementKernel",
    "ElementSet",
    "CableKernel",
    "PulleyKernel",
    "build_cable_elements",
    "build_pulley_elements",
    "line_tensions",
    "TimoshenkoBeamKernel",
    "BeamSection",
    "beam_strains",
    "build_beam_elements",
    "initial_frames_from_polyline",
    "InflatableTubeLaw",
    "InflatableBeamKernel",
    "build_inflatable_beam_elements",
    "inflatable_beam_state",
    "MembraneKernel",
    "build_membrane_elements",
    "membrane_reference",
    "membrane_regimes",
    "SLACK",
    "WRINKLED",
    "TAUT",
]

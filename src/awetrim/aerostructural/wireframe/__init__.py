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

"""Wireframe aerostructural backend — Billow's line system coupled to the VSM trim.

The bridle-and-lines fidelity: tension-only cables and frictionless pulleys, no
bending, solved for static equilibrium by energy minimisation. This was the
Particle System Simulator until 2026-09-12.
"""

from awetrim.aerostructural.wireframe.coupling import WireframeQsmCoupler

__all__ = ["WireframeQsmCoupler"]

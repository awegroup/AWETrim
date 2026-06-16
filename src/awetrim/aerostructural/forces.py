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

"""Force assembly helpers for aerostructural coupling."""

from __future__ import annotations

import numpy as np


def distribute_total_force_by_particle_mass(total_force, m_arr) -> np.ndarray:
    """Distribute a total 3D force over nodes proportional to particle masses."""
    total_force = np.asarray(total_force, dtype=float).reshape(3)
    masses = np.asarray(m_arr, dtype=float).reshape(-1)
    mass_sum = float(np.sum(masses))
    if mass_sum <= 1e-12:
        raise ValueError("Total particle mass must be positive to distribute force.")

    mass_fraction = masses / mass_sum
    return mass_fraction[:, None] * total_force[None, :]


__all__ = ["distribute_total_force_by_particle_mass"]

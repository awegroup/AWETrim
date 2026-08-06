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

"""Factory for :class:`awetrim.environment.Wind.Wind` models.

Centralises the construction of the three supported wind profiles so callers
(scripts, REST server) do not have to know the ``Wind`` class quirks:

- ``direction_wind`` must be numeric, otherwise ``Wind`` leaves it as a free
  CasADi symbol and any NLP built on top becomes unsolvable.
- ``height_ref`` defaults to 6 m and is read by the ``speed_wind_ref`` setter,
  so it must be assigned *before* the reference speed.
"""

from typing import Optional, Sequence

from awetrim.environment.Wind import Wind


def create_wind_model(
    model_type: str,
    *,
    U_ref: Optional[float] = None,
    z_ref: float = 100.0,
    z0: float = 0.03,
    heights: Optional[Sequence[float]] = None,
    speeds: Optional[Sequence[float]] = None,
    direction_wind: float = 0.0,
) -> Wind:
    """Build a fully numeric :class:`Wind` model.

    Args:
        model_type: ``"logarithmic"``, ``"uniform"`` or ``"tabulated"``.
        U_ref: wind speed [m/s] at ``z_ref`` (logarithmic) or everywhere
            (uniform). Required for those two model types.
        z_ref: reference height [m] at which ``U_ref`` is specified.
        z0: roughness length [m] (logarithmic only).
        heights: strictly increasing sample heights [m] (tabulated only).
        speeds: wind speeds [m/s] at ``heights`` (tabulated only).
        direction_wind: wind direction [rad]; 0 = wind blowing along +x.
    """
    if model_type in ("logarithmic", "uniform"):
        if U_ref is None:
            raise ValueError(f"'{model_type}' wind model requires U_ref")
        if U_ref <= 0:
            raise ValueError("U_ref must be positive")
        wind = Wind(wind_model=model_type, z0=z0, direction_wind=direction_wind)
        # height_ref must be set before speed_wind_ref: the setter derives
        # speed_friction from U_ref * kappa / ln(height_ref / z0).
        wind.height_ref = z_ref
        wind.speed_wind_ref = U_ref
        return wind

    if model_type == "tabulated":
        if heights is None or speeds is None:
            raise ValueError("'tabulated' wind model requires heights and speeds")
        heights = [float(h) for h in heights]
        speeds = [float(v) for v in speeds]
        if len(heights) != len(speeds):
            raise ValueError("heights and speeds must have the same length")
        if len(heights) < 2:
            raise ValueError("tabulated wind requires at least two samples")
        if any(h2 <= h1 for h1, h2 in zip(heights, heights[1:])):
            raise ValueError("heights must be strictly increasing")
        if heights[0] <= 0:
            raise ValueError("heights must be positive")
        return Wind(
            wind_model="tabulated",
            tabulated_heights=heights,
            tabulated_speeds=speeds,
            direction_wind=direction_wind,
        )

    raise ValueError(
        f"Unknown wind model type: {model_type!r}. "
        "Supported: 'logarithmic', 'uniform', 'tabulated'."
    )

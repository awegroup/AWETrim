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

Centralises the construction of the supported wind profiles so callers
(scripts, REST server) do not have to know the ``Wind`` class quirks:

- ``direction_wind`` must be numeric, otherwise ``Wind`` leaves it as a free
  CasADi symbol and any NLP built on top becomes unsolvable.
- the amplitude is the speed ``U_ref`` at ``z_ref``; ``Wind`` derives the
  friction velocity lazily, so no ordering constraints apply.

The formulas of every analytic law live in
:mod:`awetrim.environment.profile_laws`; the client-facing
``InflowConditions`` translation lives in
:mod:`awetrim.environment.wind_profiles`.
"""

from typing import Optional, Sequence

from awetrim.environment.Wind import Wind
from awetrim.environment.profile_laws import (
    ANALYTIC_MODELS as _ANALYTIC_MODELS,
    DEFAULT_POWER_LAW_ALPHA,
)


def create_wind_model(
    model_type: str,
    *,
    U_ref: Optional[float] = None,
    z_ref: float = 100.0,
    z0: float = 0.03,
    alpha: float = DEFAULT_POWER_LAW_ALPHA,
    heights: Optional[Sequence[float]] = None,
    speeds: Optional[Sequence[float]] = None,
    direction_wind: float = 0.0,
    jet_amplitude: float = 0.0,
    jet_height: Optional[float] = None,
    jet_width: Optional[float] = None,
) -> Wind:
    """Build a fully numeric :class:`Wind` model.

    Args:
        model_type: ``"logarithmic"``, ``"uniform"``, ``"power_law"``,
            ``"explog"``, ``"jet"`` or ``"tabulated"``.
        U_ref: wind speed [m/s] at ``z_ref`` (everywhere, for ``uniform``);
            for ``jet`` it is the reference speed of the log background.
            Required for every model except ``tabulated``.
        z_ref: reference height [m] at which ``U_ref`` is specified.
        z0: roughness length [m] (``logarithmic``, ``explog``, ``jet``).
        alpha: power-law exponent (``power_law``, ``explog``).
        heights: strictly increasing sample heights [m] (tabulated only).
        speeds: wind speeds [m/s] at ``heights`` (tabulated only).
        direction_wind: wind direction [rad]; 0 = wind blowing along +x.
        jet_amplitude: Gaussian jet peak speed [m/s] (``jet`` only).
        jet_height: height of the jet peak [m] (``jet`` only).
        jet_width: standard deviation of the jet [m] (``jet`` only).
    """
    if model_type in _ANALYTIC_MODELS:
        if U_ref is None:
            raise ValueError(f"'{model_type}' wind model requires U_ref")
        if U_ref <= 0:
            raise ValueError("U_ref must be positive")
        if model_type == "jet":
            if jet_height is None or jet_width is None:
                raise ValueError(
                    "'jet' wind model requires jet_height and jet_width"
                )
            if jet_width <= 0:
                raise ValueError("jet_width must be positive")
        wind = Wind(
            wind_model=model_type,
            z0=z0,
            direction_wind=direction_wind,
            alpha=alpha,
            jet_amplitude=jet_amplitude,
            jet_height=jet_height,
            jet_width=jet_width,
        )
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
        f"Unknown wind model type: {model_type!r}. Supported: "
        "'logarithmic', 'uniform', 'power_law', 'explog', 'jet', 'tabulated'."
    )

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

"""Analytic wind-profile laws -- the single place these formulas are written.

Every profile is a pure function of height ``z`` and a reference speed
``u_ref`` at ``z_ref``. The functions are written against a math namespace
``xp`` so the *same* code serves CasADi (``xp=casadi`` inside
:class:`awetrim.environment.Wind.Wind`, where ``z``/``u_ref`` may be ``MX``)
and NumPy (``xp=numpy`` for least-squares fits, plots and post-processing
scripts). Only ``log`` and ``exp`` are taken from ``xp``; ``+ - * / **`` are
polymorphic.

===========  ==============================================================
model        u(z)
===========  ==============================================================
uniform      u_ref
logarithmic  u_ref * ln(z/z0) / ln(z_ref/z0)
power_law    u_ref * (z/z_ref)^alpha
explog       2*logarithmic - power_law        (AtmosphericModels.jl, K = 1)
jet          logarithmic + A * exp(-(z - z_c)^2 / (2 sigma^2))
===========  ==============================================================

The log law is equivalently parametrised by the friction velocity
``u* = kappa * u_ref / ln(z_ref/z0)`` (so ``u(z) = u*/kappa * ln(z/z0)``);
:func:`friction_velocity` and :func:`speed_from_friction_velocity` convert
between the two. Code that needs either form must call these helpers rather
than restate them.
"""

from typing import Any

import numpy as np

#: Von Karman constant used by the logarithmic law.
KAPPA = 0.41

#: Default power-law exponent (AtmosphericModels.jl / KiteUtils default).
DEFAULT_POWER_LAW_ALPHA = 0.08163

#: Models parametrised by a reference speed at a reference height.
ANALYTIC_MODELS = ("uniform", "logarithmic", "power_law", "explog", "jet")

#: Models whose profile contains ln(z/z0) and is therefore singular at z <= z0.
LOG_BASED_MODELS = ("logarithmic", "explog", "jet")

#: Every ``Wind.wind_model`` value the library understands.
ALL_MODELS = ANALYTIC_MODELS + ("tabulated",)


def log_law(z: Any, u_ref: Any, z_ref: Any, z0: Any, xp: Any = np) -> Any:
    """Logarithmic profile ``u_ref * ln(z/z0) / ln(z_ref/z0)``."""
    return u_ref * xp.log(z / z0) / xp.log(z_ref / z0)


def power_law(z: Any, u_ref: Any, z_ref: Any, alpha: Any, xp: Any = np) -> Any:
    """Power-law profile ``u_ref * (z/z_ref)^alpha``."""
    return u_ref * (z / z_ref) ** alpha


def explog_law(
    z: Any, u_ref: Any, z_ref: Any, z0: Any, alpha: Any, xp: Any = np
) -> Any:
    """AtmosphericModels.jl EXPLOG with K = 1: ``log + K*(log - exp)``."""
    return 2.0 * log_law(z, u_ref, z_ref, z0, xp=xp) - power_law(
        z, u_ref, z_ref, alpha, xp=xp
    )


def jet_law(
    z: Any,
    u_ref: Any,
    z_ref: Any,
    z0: Any,
    jet_amplitude: Any,
    jet_height: Any,
    jet_width: Any,
    xp: Any = np,
) -> Any:
    """Log-law background plus a Gaussian jet of peak ``jet_amplitude``."""
    return log_law(z, u_ref, z_ref, z0, xp=xp) + jet_amplitude * xp.exp(
        -((z - jet_height) ** 2) / (2.0 * jet_width**2)
    )


def friction_velocity(
    u_ref: Any, z_ref: Any, z0: Any, kappa: float = KAPPA, xp: Any = np
) -> Any:
    """``u* = kappa * u_ref / ln(z_ref/z0)`` -- log-law friction velocity
    from the speed ``u_ref`` measured/defined at height ``z_ref``."""
    return kappa * u_ref / xp.log(z_ref / z0)


def speed_from_friction_velocity(
    u_star: Any, z: Any, z0: Any, kappa: float = KAPPA, xp: Any = np
) -> Any:
    """``u(z) = u*/kappa * ln(z/z0)`` -- log-law speed at ``z`` from ``u*``."""
    return u_star / kappa * xp.log(z / z0)


def speed_profile(
    model: str,
    z: Any,
    *,
    u_ref: Any,
    z_ref: Any,
    z0: Any = None,
    alpha: Any = None,
    jet_amplitude: Any = 0.0,
    jet_height: Any = None,
    jet_width: Any = None,
    xp: Any = np,
) -> Any:
    """Evaluate analytic profile ``model`` at ``z`` (dispatch by model name).

    ``uniform`` returns ``u_ref`` itself (no dependence on ``z``); callers
    that need an array must broadcast. ``tabulated`` is not analytic and is
    handled by :class:`Wind` directly.
    """
    if model == "uniform":
        return u_ref
    if model == "logarithmic":
        return log_law(z, u_ref, z_ref, z0, xp=xp)
    if model == "power_law":
        return power_law(z, u_ref, z_ref, alpha, xp=xp)
    if model == "explog":
        return explog_law(z, u_ref, z_ref, z0, alpha, xp=xp)
    if model == "jet":
        return jet_law(
            z, u_ref, z_ref, z0, jet_amplitude, jet_height, jet_width, xp=xp
        )
    raise ValueError(
        f"Unknown analytic wind model {model!r}; expected one of {ANALYTIC_MODELS}"
    )

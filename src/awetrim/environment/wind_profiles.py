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

"""The ``InflowConditions`` wind profile laws of the co-simulation contract.

Client-side (Julia/MATLAB/Python kite simulators) the inflow is described by
an ``InflowConditions`` struct: a reference wind speed at 6 m, a direction,
a profile-law selector and a few optional parameters. This module translates
that description into :func:`awetrim.environment.wind_factory.create_wind_model`
keyword arguments.

Laws 0-3 are the analytic profiles of AtmosphericModels.jl (same formulas, so
the optimizer sees exactly the wind the kite simulator flies in), evaluated
against the reference height :data:`REFERENCE_HEIGHT`:

===  ===========  ===================================================
id   name         u(z) / u_ref
===  ===========  ===================================================
0    CONST        1
1    EXP          (z/z_ref)^alpha
2    LOG          ln(z/z0) / ln(z_ref/z0)
3    EXPLOG       2*LOG(z) - EXP(z)   (blend with K = 1)
===  ===========  ===================================================

EXPLOG subtracts the two profiles, so it is only meaningful when ``alpha``
and ``z0`` describe the same site (as in the AtmosphericModels settings
files); mismatched values can make it drop towards zero at altitude.

Laws 4-6 are *fitted*: the shape comes from the ``heights``/``speeds`` samples
in the request, not from ``wind_speed``/``alpha``/``z0``:

- ``CUSTOM_LOG`` (4): least-squares log law, fits u_ref and z0.
- ``CUSTOM_EXP`` (5): least-squares power law, fits u_ref and alpha.
- ``CUSTOM_JET`` (6): log-law background plus a Gaussian jet,
  ``u(z) = u_bg(z) + U_J * exp(-(z - z_c)^2 / (2*sigma^2))``.

All resulting profiles are smooth in ``z``, so the NLP keeps its SX expansion
(unlike the piecewise-linear ``tabulated`` model).
"""

import math
from enum import IntEnum
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from awetrim.environment.profile_laws import (
    DEFAULT_POWER_LAW_ALPHA,
    KAPPA,
    jet_law,
    log_law,
    speed_from_friction_velocity,
)

# Height [m] at which InflowConditions.wind_speed is defined.
REFERENCE_HEIGHT = 6.0

DEFAULT_ALPHA = DEFAULT_POWER_LAW_ALPHA
DEFAULT_Z0 = 0.0002

# Plausibility clamps for a fitted roughness length [m].
MIN_FITTED_Z0 = 1e-6
MAX_FITTED_Z0 = 5.0


class ProfileLaw(IntEnum):
    """Wind profile selector of the ``InflowConditions`` struct."""

    CONST = 0
    EXP = 1
    LOG = 2
    EXPLOG = 3
    CUSTOM_LOG = 4
    CUSTOM_EXP = 5
    CUSTOM_JET = 6


#: Minimum number of (height, speed) samples each fitted law needs.
_MIN_SAMPLES = {
    ProfileLaw.CUSTOM_LOG: 2,
    ProfileLaw.CUSTOM_EXP: 2,
    ProfileLaw.CUSTOM_JET: 5,
}


def direction_wind_from_compass(wind_direction_deg: float) -> float:
    """Meteorological wind direction [deg] -> AWETrim ``direction_wind`` [rad].

    ``InflowConditions.wind_direction`` follows the convention of KiteUtils'
    ``upwind_dir``: degrees clockwise from North (0 = North, 90 = East) of the
    direction the wind is coming FROM. AWETrim's ``direction_wind`` is the
    direction the wind blows TOWARDS, in radians counter-clockwise from the
    world-frame +x axis (East) of an ENU frame.

    The trajectory itself is optimized in the wind-aligned frame (azimuth 0 =
    downwind), so this angle only orients the result in the world frame; it
    does not change the optimized shape.
    """
    angle = math.radians(90.0 - (float(wind_direction_deg) + 180.0))
    return math.atan2(math.sin(angle), math.cos(angle))  # wrap to (-pi, pi]


def _clean_samples(
    heights: Sequence[float], speeds: Sequence[float], law: ProfileLaw
) -> Tuple[np.ndarray, np.ndarray]:
    """Validate the (heights, speeds) table used by the fitted profile laws."""
    z = np.asarray(heights, dtype=float).ravel()
    u = np.asarray(speeds, dtype=float).ravel()
    if z.size != u.size:
        raise ValueError("heights and speeds must have the same length")
    if not np.all(np.isfinite(z)) or not np.all(np.isfinite(u)):
        raise ValueError("heights and speeds must be finite")
    if np.any(z <= 0.0):
        raise ValueError("heights must be positive")
    if np.any(u < 0.0):
        raise ValueError("speeds must be non-negative")

    needed = _MIN_SAMPLES[law]
    if z.size < needed:
        raise ValueError(
            f"profile_law {int(law)} ({law.name}) is fitted to the "
            f"heights/speeds table and needs at least {needed} samples, "
            f"got {z.size}"
        )
    if np.unique(z).size != z.size:
        raise ValueError("heights must be distinct")
    return z, u


def fit_log_law(
    heights: Sequence[float], speeds: Sequence[float]
) -> Tuple[float, float]:
    """Fit u(z) = a*ln(z) + b -> ``(u_ref at REFERENCE_HEIGHT, z0)``.

    The log law is linear in ``ln(z)``, so this is a plain least-squares fit.
    The fitted roughness ``z0 = exp(-b/a)`` is clamped to a plausible range.
    """
    z, u = _clean_samples(heights, speeds, ProfileLaw.CUSTOM_LOG)
    slope, intercept = np.polyfit(np.log(z), u, 1)
    if slope <= 0.0:
        raise ValueError(
            "CUSTOM_LOG needs a wind profile that increases with height; "
            "the given samples fit a non-increasing log law"
        )
    z0 = float(np.clip(math.exp(-intercept / slope), MIN_FITTED_Z0, MAX_FITTED_Z0))
    # The fitted slope is u*/kappa; evaluate the law at the reference height.
    u_ref = float(
        speed_from_friction_velocity(slope * KAPPA, REFERENCE_HEIGHT, z0, xp=math)
    )
    if u_ref <= 0.0:
        raise ValueError(
            "CUSTOM_LOG fit gives a non-positive wind speed at "
            f"{REFERENCE_HEIGHT} m; check the heights/speeds samples"
        )
    return u_ref, z0


def fit_power_law(
    heights: Sequence[float], speeds: Sequence[float]
) -> Tuple[float, float]:
    """Fit u(z) = u_ref*(z/z_ref)^alpha -> ``(u_ref, alpha)``.

    Linear least squares in ``(ln z, ln u)``; all sample speeds must be > 0.
    """
    z, u = _clean_samples(heights, speeds, ProfileLaw.CUSTOM_EXP)
    if np.any(u <= 0.0):
        raise ValueError("CUSTOM_EXP needs strictly positive speeds")
    alpha, log_u_ref = np.polyfit(np.log(z / REFERENCE_HEIGHT), np.log(u), 1)
    return float(math.exp(log_u_ref)), float(alpha)


def fit_jet_profile(
    heights: Sequence[float], speeds: Sequence[float]
) -> Dict[str, float]:
    """Fit ``u(z) = u_bg(z) + U_J*exp(-(z - z_c)^2/(2*sigma^2))``.

    The background ``u_bg`` is a log law (parameters ``u_ref``/``z0``); the
    jet adds a Gaussian bump of amplitude ``U_J`` centred at ``z_c`` with
    width ``sigma``. Returns the ``create_wind_model`` parameters of the
    ``"jet"`` model.
    """
    from scipy.optimize import least_squares  # local: keeps import cost off /init

    z, u = _clean_samples(heights, speeds, ProfileLaw.CUSTOM_JET)

    # Start from the pure log law; the residual peak seeds the jet.
    try:
        u_ref0, z0_0 = fit_log_law(z, u)
    except ValueError:
        u_ref0, z0_0 = float(np.max(u)), DEFAULT_Z0
    residual = u - log_law(z, u_ref0, REFERENCE_HEIGHT, z0_0, xp=np)
    peak = int(np.argmax(residual))
    span = float(np.max(z) - np.min(z))
    x0 = [
        u_ref0,
        z0_0,
        max(float(residual[peak]), 1e-3),
        float(z[peak]),
        max(span / 4.0, 1.0),
    ]
    lower = [1e-3, MIN_FITTED_Z0, 0.0, float(np.min(z)), 1.0]
    upper = [np.inf, MAX_FITTED_Z0, np.inf, float(np.max(z)), max(10.0 * span, 10.0)]
    x0 = list(np.clip(x0, lower, upper))

    def _residuals(p):
        u_ref, z0, u_jet, z_jet, sigma = p
        return jet_law(z, u_ref, REFERENCE_HEIGHT, z0, u_jet, z_jet, sigma, xp=np) - u

    solution = least_squares(_residuals, x0, bounds=(lower, upper))
    u_ref, z0, u_jet, z_jet, sigma = (float(v) for v in solution.x)
    return {
        "U_ref": u_ref,
        "z0": z0,
        "jet_amplitude": u_jet,
        "jet_height": z_jet,
        "jet_width": sigma,
    }


def wind_kwargs_from_inflow(inflow: Mapping[str, Any]) -> Dict[str, Any]:
    """``InflowConditions`` mapping -> ``create_wind_model`` keyword arguments.

    Missing optional keys fall back to the contract defaults (``alpha``
    0.08163, ``z0`` 0.0002, ``heights`` [6.0], ``speeds`` [wind_speed]).
    ``turbulence`` is not part of the result: the optimizer is deterministic
    and works on the mean profile.
    """
    law = ProfileLaw(int(inflow["profile_law"]))
    wind_speed = float(inflow["wind_speed"])
    alpha = float(_or_default(inflow.get("alpha"), DEFAULT_ALPHA))
    z0 = float(_or_default(inflow.get("z0"), DEFAULT_Z0))
    if z0 <= 0.0:
        raise ValueError("z0 must be positive")
    heights: List[float] = list(
        _or_default(inflow.get("heights"), [REFERENCE_HEIGHT])
    )
    speeds: List[float] = list(_or_default(inflow.get("speeds"), [wind_speed]))
    direction = direction_wind_from_compass(
        _or_default(inflow.get("wind_direction"), 0.0)
    )

    if law is ProfileLaw.CONST:
        kwargs: Dict[str, Any] = {"model_type": "uniform", "U_ref": wind_speed}
    elif law is ProfileLaw.EXP:
        kwargs = {"model_type": "power_law", "U_ref": wind_speed, "alpha": alpha}
    elif law is ProfileLaw.LOG:
        kwargs = {"model_type": "logarithmic", "U_ref": wind_speed, "z0": z0}
    elif law is ProfileLaw.EXPLOG:
        kwargs = {
            "model_type": "explog",
            "U_ref": wind_speed,
            "z0": z0,
            "alpha": alpha,
        }
    elif law is ProfileLaw.CUSTOM_LOG:
        u_ref, z0_fit = fit_log_law(heights, speeds)
        kwargs = {"model_type": "logarithmic", "U_ref": u_ref, "z0": z0_fit}
    elif law is ProfileLaw.CUSTOM_EXP:
        u_ref, alpha_fit = fit_power_law(heights, speeds)
        kwargs = {"model_type": "power_law", "U_ref": u_ref, "alpha": alpha_fit}
    else:  # ProfileLaw.CUSTOM_JET
        kwargs = {"model_type": "jet", **fit_jet_profile(heights, speeds)}

    if kwargs["model_type"] != "uniform":
        kwargs["z_ref"] = REFERENCE_HEIGHT
    kwargs["direction_wind"] = direction
    return kwargs


def _or_default(value: Any, default: Any) -> Any:
    """Treat an explicit ``None`` like an omitted optional field."""
    return default if value is None else value

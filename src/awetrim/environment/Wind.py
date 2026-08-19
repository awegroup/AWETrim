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

import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from awetrim.utils.reference_frames import transformation_C_from_W
from awetrim.environment.profile_laws import (
    ALL_MODELS,
    DEFAULT_POWER_LAW_ALPHA,
    KAPPA,
    LOG_BASED_MODELS,
    friction_velocity,
    speed_from_friction_velocity,
    speed_profile,
)


class Wind:
    """Wind profile u(z) and the world-frame velocity derived from it.

    Supported ``wind_model`` values: ``uniform``, ``logarithmic``,
    ``power_law``, ``explog``, ``jet`` (the analytic laws of
    :mod:`awetrim.environment.profile_laws`, which is the single place their
    formulas live) and ``tabulated`` (piecewise-linear CasADi interpolant).
    All but ``tabulated`` are smooth in ``z``, so the NLP built on top of
    them can still be expanded to SX.

    **Amplitude.** Every analytic profile is scaled by one amplitude, which
    can be given either as the reference speed ``speed_wind_ref`` at
    ``height_ref`` or (log-based models) as the friction velocity
    ``speed_friction``. Whichever was set last is stored; the other is
    derived on access through :mod:`profile_laws`, so ``z0`` / ``height_ref``
    may be changed at any time without the two going stale. When neither is
    set the amplitude is a free CasADi symbol: ``speed_friction`` for the
    log-based models (the identification/validation scripts feed the
    friction velocity as an NLP parameter or unknown), ``speed_wind_ref``
    otherwise.
    """

    def __init__(
        self,
        wind_model="logarithmic",
        z0=0.01,
        tabulated_heights=None,
        tabulated_speeds=None,
        direction_wind=None,
        speed_wind_ref=None,
        alpha=DEFAULT_POWER_LAW_ALPHA,
        jet_amplitude=0.0,
        jet_height=None,
        jet_width=None,
    ):
        if wind_model not in ALL_MODELS:
            raise ValueError(
                f"Unknown wind model: {wind_model!r}. Supported: {ALL_MODELS}"
            )
        self._height_ref = 6
        self.wind_model = wind_model
        self.kappa = KAPPA
        self.z0 = z0
        # Power-law exponent (power_law/explog) and Gaussian jet parameters
        # (jet); see profile_laws.jet_law.
        self.alpha = alpha
        self.jet_amplitude = jet_amplitude
        self.jet_height = jet_height
        self.jet_width = jet_width
        if direction_wind is None:
            self._direction_wind = ca.MX.sym("direction_wind")
        else:
            self._direction_wind = direction_wind

        # Single stored amplitude: (kind, value). ``_amplitude_set`` is False
        # while the amplitude is still the default free symbol.
        self._amplitude_kind = None
        self._amplitude = None
        self._amplitude_set = False
        if speed_wind_ref is not None:
            self.speed_wind_ref = speed_wind_ref
        else:
            kind = (
                "speed_friction"
                if wind_model in LOG_BASED_MODELS
                else "speed_wind_ref"
            )
            self._amplitude_kind = kind
            self._amplitude = ca.MX.sym(kind)

        # Store tabulated data if applicable
        self.tabulated_heights = tabulated_heights
        self.tabulated_speeds = tabulated_speeds

        if self.wind_model == "tabulated":
            if tabulated_heights is None or tabulated_speeds is None:
                raise ValueError("Tabulated wind model requires heights and speeds.")

            # Create linear interpolant (1D)
            self.wind_interp = ca.interpolant(
                "wind_interp",
                "linear",
                [tabulated_heights],
                tabulated_speeds,
            )

    # ------------------------------------------------------------------
    # Amplitude: reference speed <-> friction velocity
    # ------------------------------------------------------------------
    @property
    def speed_wind_ref(self):
        """Wind speed at ``height_ref`` (value, expression or free symbol)."""
        if self._amplitude_kind == "speed_wind_ref":
            return self._amplitude
        return speed_from_friction_velocity(
            self._amplitude, self.height_ref, self.z0, self.kappa, xp=ca
        )

    @speed_wind_ref.setter
    def speed_wind_ref(self, value):
        self._amplitude_kind = "speed_wind_ref"
        self._amplitude = value
        self._amplitude_set = True

    @property
    def speed_wind_ref_value(self):
        """``speed_wind_ref`` if an amplitude was set, else ``None``.

        The returned object may itself be symbolic when the caller set a
        symbol (e.g. the NLP parametrisation in ``residual_solver``).
        """
        return self.speed_wind_ref if self._amplitude_set else None

    @property
    def speed_friction(self):
        """Log-law friction velocity ``u*`` (value, expression or free symbol)."""
        if self._amplitude_kind == "speed_friction":
            return self._amplitude
        return friction_velocity(
            self._amplitude, self.height_ref, self.z0, self.kappa, xp=ca
        )

    @speed_friction.setter
    def speed_friction(self, value):
        self._amplitude_kind = "speed_friction"
        self._amplitude = value
        self._amplitude_set = True

    @property
    def direction_wind(self):
        return self._direction_wind

    @direction_wind.setter
    def direction_wind(self, value):
        self._direction_wind = value

    @property
    def height_ref(self):
        return self._height_ref

    @height_ref.setter
    def height_ref(self, value):
        self._height_ref = value

    # ------------------------------------------------------------------
    # Profile evaluation
    # ------------------------------------------------------------------
    def _profile_kwargs(self):
        return dict(
            u_ref=self.speed_wind_ref,
            z_ref=self.height_ref,
            z0=self.z0,
            alpha=self.alpha,
            jet_amplitude=self.jet_amplitude,
            jet_height=self.jet_height,
            jet_width=self.jet_width,
        )

    # Should be renamed to speed_wind_kite
    def speed_wind(self, height):
        if self.wind_model == "tabulated":
            return self.wind_interp(height)
        return speed_profile(self.wind_model, height, xp=ca, **self._profile_kwargs())

    def velocity_wind_W(self, height):
        return ca.vertcat(
            self.speed_wind(height) * ca.cos(self.direction_wind),
            self.speed_wind(height) * ca.sin(self.direction_wind),
            0,
        )

    def velocity_wind(self, model):
        """
        Compute the wind velocity in the body frame.
        """
        T_C_from_W = transformation_C_from_W(
            model.angle_azimuth, model.angle_elevation, model.angle_course
        )
        return T_C_from_W @ self.velocity_wind_W(model.z)

    def speed_wind_at_height(self, height):
        return self.speed_wind(height)

    def velocity_wind_at_height_W(self, height):
        # World-frame wind at an explicit height. Direction-aware and routed
        # through velocity_wind_W so it stays consistent with velocity_wind
        # (previously this dropped direction_wind and put all speed on +x).
        return self.velocity_wind_W(height)

    def velocity_wind_at_height(self, model, height):
        """
        Compute the wind velocity in the body frame.
        """
        T_C_from_W = transformation_C_from_W(
            model.angle_azimuth, model.angle_elevation, model.angle_course
        )
        return T_C_from_W @ self.velocity_wind_at_height_W(height)

    # ------------------------------------------------------------------
    # Visualization helper
    # ------------------------------------------------------------------
    def profile_numeric(self, heights):
        """Evaluate u(z) with NumPy on an array of heights (numeric amplitude)."""
        z = np.asarray(heights, dtype=float)
        if self.wind_model == "tabulated":
            z_tab = np.asarray(self.tabulated_heights, dtype=float)
            u_tab = np.asarray(self.tabulated_speeds, dtype=float)
            order = np.argsort(z_tab)
            return np.interp(z, z_tab[order], u_tab[order])
        kwargs = self._profile_kwargs()
        kwargs["u_ref"] = float(kwargs["u_ref"])
        speeds = np.asarray(speed_profile(self.wind_model, z, xp=np, **kwargs), dtype=float)
        if speeds.ndim == 0:  # uniform: no z-dependence
            speeds = np.full_like(z, float(speeds))
        return speeds

    def plot_profile(self, z_min=5.0, z_max=200.0, num=100, show=True):
        """Plot wind speed vs height for the configured model."""
        if z_max <= z_min:
            raise ValueError("z_max must be greater than z_min")

        z_samples = np.linspace(z_min, z_max, num)
        speeds = self.profile_numeric(z_samples)

        plt.figure(figsize=(4, 4))
        plt.plot(speeds, z_samples, label=self.wind_model)
        if self.wind_model == "tabulated":
            plt.plot(self.tabulated_speeds, self.tabulated_heights, "o", label="data")
        plt.xlabel("Wind speed (m/s)")
        plt.ylabel("Height (m)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        if show:
            plt.show()
        return z_samples, speeds

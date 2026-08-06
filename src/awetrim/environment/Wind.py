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


class Wind:
    """Wind profile u(z) and the world-frame velocity derived from it.

    Supported ``wind_model`` values: ``uniform``, ``logarithmic``,
    ``tabulated`` and the profile laws of the ``InflowConditions`` contract
    (see :mod:`awetrim.environment.wind_profiles`): ``power_law``, ``explog``
    and ``jet``. All but ``tabulated`` are smooth in ``z``, so the NLP built
    on top of them can still be expanded to SX.
    """

    def __init__(
        self,
        wind_model="logarithmic",
        z0=0.01,
        tabulated_heights=None,
        tabulated_speeds=None,
        direction_wind=None,
        speed_wind_ref=None,
        alpha=0.08163,
        jet_amplitude=0.0,
        jet_height=None,
        jet_width=None,
    ):
        self._height_ref = 6
        self.wind_model = wind_model
        self.kappa = 0.41
        self.z0 = z0
        # Power-law exponent (power_law/explog) and Gaussian jet parameters
        # (jet): u(z) = u_bg(z) + jet_amplitude*exp(-(z - jet_height)^2 /
        # (2*jet_width^2)).
        self.alpha = alpha
        self.jet_amplitude = jet_amplitude
        self.jet_height = jet_height
        self.jet_width = jet_width
        self._speed_friction = ca.MX.sym("speed_friction")
        if direction_wind is None:
            self._direction_wind = ca.MX.sym("direction_wind")
        else:
            self._direction_wind = direction_wind
        self._speed_wind_ref = ca.MX.sym("speed_wind_ref")
        self._speed_wind_ref_value = None
        if speed_wind_ref is not None:
            self.speed_wind_ref = speed_wind_ref

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

    @property
    def speed_wind_ref(self):
        if self._speed_wind_ref_value is not None:
            return self._speed_wind_ref_value
        return self._speed_wind_ref

    @property
    def speed_wind_ref_value(self):
        return self._speed_wind_ref_value

    @speed_wind_ref.setter
    def speed_wind_ref(self, value):
        self._speed_wind_ref_value = value
        if isinstance(value, (ca.MX, ca.SX)) and value.is_symbolic():
            self._speed_wind_ref = value
        self._speed_friction = value * self.kappa / ca.log(self.height_ref / self.z0)

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

    @property
    def speed_friction(self):
        return self._speed_friction

    @speed_friction.setter
    def speed_friction(self, value):
        self._speed_friction = value
        self._speed_wind_ref_value = (
            value / self.kappa * ca.log(self.height_ref / self.z0)
        )

    def _speed_logarithmic(self, height):
        # = speed_wind_ref * ln(z/z0) / ln(height_ref/z0)
        return self._speed_friction / self.kappa * ca.log(height / self.z0)

    def _speed_power_law(self, height):
        return self.speed_wind_ref * (height / self.height_ref) ** self.alpha

    # Should be renamed to speed_wind_kite
    def speed_wind(self, height):
        if self.wind_model == "uniform":
            return self.speed_wind_ref
        elif self.wind_model == "logarithmic":
            return self._speed_logarithmic(height)
        elif self.wind_model == "power_law":
            return self._speed_power_law(height)
        elif self.wind_model == "explog":
            # AtmosphericModels.jl EXPLOG with K = 1: log + K*(log - exp).
            return 2.0 * self._speed_logarithmic(height) - self._speed_power_law(
                height
            )
        elif self.wind_model == "jet":
            return self._speed_logarithmic(height) + self.jet_amplitude * ca.exp(
                -((height - self.jet_height) ** 2) / (2.0 * self.jet_width**2)
            )
        elif self.wind_model == "tabulated":
            return self.wind_interp(height)
        raise ValueError(f"Unknown wind model: {self.wind_model}")

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
    def plot_profile(self, z_min=5.0, z_max=200.0, num=100, show=True):
        """Plot wind speed vs height for the configured model."""
        if z_max <= z_min:
            raise ValueError("z_max must be greater than z_min")

        z_samples = np.linspace(z_min, z_max, num)
        speed_wind_ref_value = self.speed_wind_ref_value

        def log_speeds():
            return (float(self.speed_friction) / self.kappa) * np.log(
                z_samples / self.z0
            )

        def power_law_speeds():
            return float(speed_wind_ref_value) * (
                z_samples / self.height_ref
            ) ** self.alpha

        if self.wind_model == "uniform":
            speeds = np.full_like(z_samples, float(speed_wind_ref_value or 0.0))
        elif self.wind_model == "logarithmic":
            speeds = log_speeds()
        elif self.wind_model == "power_law":
            speeds = power_law_speeds()
        elif self.wind_model == "explog":
            speeds = 2.0 * log_speeds() - power_law_speeds()
        elif self.wind_model == "jet":
            speeds = log_speeds() + self.jet_amplitude * np.exp(
                -((z_samples - self.jet_height) ** 2) / (2.0 * self.jet_width**2)
            )
        elif self.wind_model == "tabulated":
            if self.tabulated_heights is None or self.tabulated_speeds is None:
                raise ValueError("Tabulated wind model requires heights and speeds.")
            z_sorted = np.array(self.tabulated_heights, dtype=float)
            w_sorted = np.array(self.tabulated_speeds, dtype=float)
            order = np.argsort(z_sorted)
            z_sorted = z_sorted[order]
            w_sorted = w_sorted[order]
            speeds = np.interp(z_samples, z_sorted, w_sorted)
        else:
            raise ValueError(f"Unknown wind model: {self.wind_model}")

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

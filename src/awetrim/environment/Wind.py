import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from awetrim.utils.reference_frames import transformation_C_from_W


class Wind:
    def __init__(
        self,
        wind_model="logarithmic",
        z0=0.01,
        tabulated_heights=None,
        tabulated_speeds=None,
        direction_wind=None,
        speed_wind_ref=None,
    ):
        if speed_wind_ref is None:
            self._speed_wind_ref = ca.MX.sym("speed_wind_ref")
        else:
            self._speed_wind_ref = speed_wind_ref
        self._speed_friction = ca.MX.sym("speed_friction")
        if direction_wind is None:
            self._direction_wind = ca.MX.sym(
                "direction_wind"
            )  # Wind coming from x-direction
        else:
            self._direction_wind = direction_wind
        self._height_ref = 6  # 10
        self.wind_model = wind_model
        self.kappa = 0.41
        self.z0 = z0

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
        return self._speed_wind_ref

    @speed_wind_ref.setter
    def speed_wind_ref(self, value):
        self._speed_friction = value * self.kappa / ca.log(self.height_ref / self.z0)
        self._speed_wind_ref = value

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
        self._speed_wind_ref = value / self.kappa * ca.log(self.height_ref / self.z0)

    # Should be renamed to speed_wind_kite
    def speed_wind(self, state):
        if self.wind_model == "uniform":
            return self.speed_wind_ref
        elif self.wind_model == "logarithmic":
            return self._speed_friction / self.kappa * ca.log(state.z / self.z0)
        elif self.wind_model == "tabulated":
            return self.wind_interp(state.z)

    def velocity_wind_W(self, state):
        return ca.vertcat(
            self.speed_wind(state) * ca.cos(self.direction_wind),
            self.speed_wind(state) * ca.sin(self.direction_wind),
            0,
        )

    def velocity_wind(self, state):
        """
        Compute the wind velocity in the body frame.
        """
        T_C_from_W = transformation_C_from_W(
            state.angle_azimuth, state.angle_elevation, state.angle_course
        )
        return T_C_from_W @ self.velocity_wind_W(state)

    def speed_wind_at_height(self, height):
        if self.wind_model == "uniform":
            return self.speed_wind_ref
        elif self.wind_model == "logarithmic":
            return self._speed_friction / self.kappa * ca.log(height / self.z0)
        elif self.wind_model == "tabulated":
            return self.wind_interp(height)

    def velocity_wind_at_height_W(self, height):
        return ca.vertcat(self.speed_wind_at_height(height), 0, 0)

    def velocity_wind_at_height(self, state, height):
        """
        Compute the wind velocity in the body frame.
        """
        T_C_from_W = transformation_C_from_W(
            state.angle_azimuth, state.angle_elevation, state.angle_course
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

        if self.wind_model == "uniform":
            speeds = np.full_like(z_samples, float(self.speed_wind_ref))
        elif self.wind_model == "logarithmic":
            speeds = (float(self.speed_friction) / self.kappa) * np.log(
                z_samples / self.z0
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

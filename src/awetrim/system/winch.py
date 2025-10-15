import casadi as ca
from picawe.utils.defaults import DEFAULT_WINCH_CONFIG


class Winch:
    def __init__(self, pattern_config, config=DEFAULT_WINCH_CONFIG):
        self.max_tether_length = config["max_tether_length"]
        self.min_tether_length = config["min_tether_length"]
        self.max_speed = config["max_speed"]
        self.min_speed = config["min_speed"]
        self.max_acceleration = config["max_acceleration"]
        self.min_acceleration = config["min_acceleration"]

        self.pattern_config = pattern_config

    def tension_curve(self, speed_radial):
        """Nominal tether force model as a CasADi function f(v_r).

        Uses `pattern_config` to choose the shape and optional smoothing:
        - force_model: "linear" or "quadratic" (default: "quadratic")
        - max_tether_force: required (N)
        - min_tether_force: optional (N, default 0)
        - softplus / softminus: optional boolean flags
        - softplus_beta / softminus_beta: optional sharpness parameters
        """

        model = self.pattern_config.get("force_model", "quadratic")
        max_tf = self.pattern_config.get("max_tether_force", None)
        min_tf = self.pattern_config.get("min_tether_force", 0)

        if max_tf is None:
            raise ValueError(
                "pattern_config must define 'max_tether_force' for tension_curve"
            )

        if model == "linear":
            T = self.pattern_config["slope"] * (
                speed_radial - self.pattern_config.get("offset", 0)
            )
        elif model == "quadratic":
            T = (
                self.pattern_config["slope"]
                * (speed_radial - self.pattern_config.get("offset", 0))
                * ca.fabs(speed_radial - self.pattern_config.get("offset", 0))
            )
        else:
            raise ValueError(f"Unknown force_model '{model}' in pattern_config")

        # Optional smoothing limits
        if self.pattern_config.get("softplus", False):
            beta = self.pattern_config.get(
                "softplus_beta", DEFAULT_WINCH_CONFIG.get("sharpness_beta", 1e-3)
            )
            T = T - (1 / beta) * ca.log(1 + ca.exp(beta * (T - max_tf)))
        if self.pattern_config.get("softminus", False):
            beta = self.pattern_config.get(
                "softminus_beta", DEFAULT_WINCH_CONFIG.get("sharpness_beta", 1e-3)
            )
            T = T + (1 / beta) * ca.log(1 + ca.exp(beta * (min_tf - T)))

        return T

    def radial_equation(self, speed_radial=None, tension_tether_ground=None):
        """Algebraic equation for radial dynamics when using force control.

        Args:
            self: Winch object
            kite_model: Kite object
        Returns:
            radial_equation: Algebraic equation for radial dynamics
        """

        if self.pattern_config["reeling_strategy"] == "force":
            if speed_radial is None or tension_tether_ground is None:
                raise ValueError(
                    "speed_radial and tension_tether_ground must be provided for force control"
                )
            # Use the unified tension_curve property
            tension_curve_val = self.tension_curve(speed_radial)
            radial_force_law = tension_tether_ground - tension_curve_val
        elif self.pattern_config["reeling_strategy"] == "constant":
            radial_force_law = speed_radial - self.pattern_config["reeling_speed"]
        else:
            raise ValueError("Unknown reeling_strategy in pattern_config")

        return radial_force_law

    def plot_tension_curve(
        self,
        reeling_speeds=None,
        vr_min=None,
        vr_max=None,
        n_points=200,
        show=True,
        ax=None,
    ):
        """Plot tension_curve over provided or default reeling speeds.

        Args:
            reeling_speeds: Optional array-like of v_r to evaluate. If None, a
                            linspace from vr_min to vr_max is used.
            vr_min: Minimum v_r for default range (defaults to self.min_speed).
            vr_max: Maximum v_r for default range (defaults to self.max_speed).
            n_points: Number of points for default range.
            show: Whether to call plt.show() when creating a figure.
            ax: Optional matplotlib axis to draw on.

        Returns:
            (v_r, T_vals): Arrays of speeds and corresponding model forces.
        """
        import numpy as np
        import matplotlib.pyplot as plt

        if reeling_speeds is None:
            vr_min = self.min_speed if vr_min is None else vr_min
            vr_max = self.max_speed if vr_max is None else vr_max
            v_r = np.linspace(vr_min, vr_max, n_points)
        else:
            v_r = np.asarray(reeling_speeds, dtype=float)

        T_fun = self.tension_curve
        T_vals = np.array([float(T_fun(v)) for v in v_r])

        created_fig = False
        if ax is None:
            fig, ax = plt.subplots()
            created_fig = True

        ax.plot(v_r, T_vals, label="tension_curve(v_r)")
        ax.set_xlabel("Reeling speed v_r (m/s)")
        ax.set_ylabel("tension_curve (N)")
        ax.grid(True)
        ax.legend()

        if created_fig and show:
            plt.show()

        return v_r, T_vals

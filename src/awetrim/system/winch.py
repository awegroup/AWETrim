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

from dataclasses import dataclass

import casadi as ca
import numpy as np
from awetrim.utils.defaults import DEFAULT_WINCH_CONFIG

# Floor on v_sat - speed_radial in the inverse soft v_sat clamp [m/s], see
# Winch._undo_v_sat_clamp.
V_SAT_GAP_MIN = 1e-6
# Floor on the argument of the inverse softplus, and on the unsaturated tension
# under the square root [N], in Winch.speed_law: both diverge (log(0), infinite
# slope of sqrt at 0) at the edges of the law's range.
SP_INV_Y_MIN = 1e-12
SPEED_LAW_T_MIN = 1.0
# Where Winch.radial_equation hands over from the force form to the speed form,
# as fractions of [min_tether_force, max_tether_force]: centre and width.
# Measured 2026-09-23 on the Cabauw 150 m startup request: at 0.5 the 4 m/s
# solve fails in IPOPT; 0.75 and 0.85 both converge at 3, 4, 5, 7 and 10 m/s,
# 0.85 landing 5 % lower at 7 m/s (25.3 instead of 26.8 kW). 0.85 by choice.
SPEED_LAW_BLEND_CENTER = 0.85
SPEED_LAW_BLEND_WIDTH = 0.1


def _sp_inv(y):
    """Inverse of the softplus ``ln(1 + e^x)``, i.e. ``ln(e^y - 1)``, floored."""
    y = ca.fmax(y, SP_INV_Y_MIN)
    return y + ca.log(-ca.expm1(-y))


def drivetrain_friction_force(
    speed_radial,
    friction_coulomb,
    friction_viscous,
    speed_smoothing=0.05,
    xp=np,
    tension=0.0,
    load_fraction=0.0,
):
    """Drivetrain friction force at the tether [N], opposing the reel speed.

    ``F_f = (F_c + k T) tanh(v / eps) + c_v v``: Coulomb friction F_c, an
    optional load-proportional part ``k T`` (pulley bearings and rope bending,
    belt, gear mesh: a fixed fraction of the transmitted force) and viscous
    friction c_v, all tether-referred (a motor-side torque tau times n / r_drum).
    The tanh smooths sign(v), so the Coulomb term has no kink at v = 0, where
    v_r crosses twice per cycle and IPOPT needs second derivatives; same role as
    smooth_sign in WinchModels.jl / SymbolicAWEModels.jl.

    Args:
        speed_radial: reel speed v_r [m/s], positive = reel-out.
        friction_coulomb: F_c [N] (system.yaml drums[0].friction_coulomb).
        friction_viscous: c_v [N s/m] (drums[0].friction_viscous).
        speed_smoothing: eps [m/s] of the tanh sign.
        xp: math namespace, ``numpy`` or ``casadi``.
        tension: tether tension T [N] at the ground station (>= 0); only used
            with ``load_fraction``.
        load_fraction: k [-], load-proportional loss as a fraction of T
            (drums[0].friction_load_fraction).
    """
    v = speed_radial
    force = friction_coulomb + load_fraction * tension
    return force * xp.tanh(v / speed_smoothing) + friction_viscous * v


def drivetrain_loss_power(
    speed_radial,
    friction_coulomb,
    friction_viscous,
    speed_smoothing=0.05,
    xp=np,
    tension=0.0,
    load_fraction=0.0,
):
    """Power lost to winch drivetrain friction [W], >= 0 in both reel directions.

    ``P_loss = v F_f(v)`` with F_f = :func:`drivetrain_friction_force`, i.e.
    ``(F_c + k T) v tanh(v / eps) + c_v v^2``. The tanh under-counts F_c |v|
    only for |v| < ~2 eps, by at most 0.28 F_c eps (1.7 W for F_c = 122 N at
    |v| = 0.04 m/s). The drive power is then ``T v - P_loss``: the generator
    gets less than the tether delivers on reel-out, the motor supplies more
    than the tether absorbs on reel-in. Arguments as
    :func:`drivetrain_friction_force`.
    """
    return speed_radial * drivetrain_friction_force(
        speed_radial,
        friction_coulomb,
        friction_viscous,
        speed_smoothing=speed_smoothing,
        xp=xp,
        tension=tension,
        load_fraction=load_fraction,
    )


@dataclass(frozen=True)
class Drivetrain:
    """Winch drivetrain between the tether and the motor, tether-referred.

    The drum obeys ``m dv_r/dt = T - F_f(v_r) - F_motor``: the tether force T
    drives it, friction F_f (:func:`drivetrain_friction_force`) and the
    motor/generator force F_motor brake it. ``inertia_mass`` is the rotating
    inertia seen at the tether, ``m = J_total (n / r_drum)^2`` (J_total
    motor-side, n the gear ratio). All zero = the ideal winch: F_motor = T.

    The winch force law (:class:`Winch`) sets F_motor, as a torque-controlled
    machine does (V3Kite's TorqueControlledMachine), so the tension the kite
    feels is ``T = F_law(v) + F_f(v) + m dv/dt``.
    """

    inertia_mass: float = 0.0
    friction_coulomb: float = 0.0
    friction_viscous: float = 0.0
    load_fraction: float = 0.0
    speed_smoothing: float = 0.05

    @property
    def has_friction(self):
        return bool(self.friction_coulomb or self.friction_viscous or self.load_fraction)

    @property
    def is_ideal(self):
        return not self.inertia_mass and not self.has_friction

    def friction_force(self, speed_radial, tension=0.0, xp=np):
        """F_f [N] at the tether, see :func:`drivetrain_friction_force`."""
        if not self.has_friction:
            return 0.0
        return drivetrain_friction_force(
            speed_radial,
            self.friction_coulomb,
            self.friction_viscous,
            speed_smoothing=self.speed_smoothing,
            xp=xp,
            tension=tension,
            load_fraction=self.load_fraction,
        )

    def loss_power(self, speed_radial, tension=0.0, xp=np):
        """Friction power [W], ``v F_f(v)``, >= 0 (:func:`drivetrain_loss_power`)."""
        if not self.has_friction:
            return 0.0
        return drivetrain_loss_power(
            speed_radial,
            self.friction_coulomb,
            self.friction_viscous,
            speed_smoothing=self.speed_smoothing,
            xp=xp,
            tension=tension,
            load_fraction=self.load_fraction,
        )

    def motor_force(self, tension, speed_radial, acceleration=None, xp=np):
        """F_motor = T - F_f(v) - m dv_r/dt [N], tether-referred.

        ``acceleration`` is the drum's dv_r/dt from the caller's time scheme
        (awetrim.utils.time_scheme: implicit, since the drum's time constant,
        m over the law's slope, is far below a node step); None = steady drum,
        no inertia term.
        """
        force = tension - self.friction_force(speed_radial, tension, xp=xp)
        if self.inertia_mass and acceleration is not None:
            force = force - self.inertia_mass * acceleration
        return force


class Winch:
    def __init__(self, pattern_config, config=DEFAULT_WINCH_CONFIG, drivetrain=None):
        self.max_tether_length = config["max_tether_length"]
        self.min_tether_length = config["min_tether_length"]
        self.max_speed = config["max_speed"]
        self.min_speed = config["min_speed"]
        self.max_acceleration = config["max_acceleration"]
        self.min_acceleration = config["min_acceleration"]

        self.pattern_config = pattern_config
        # Drum inertia + friction between the tether and the motor; the force
        # law sets the MOTOR force (see Drivetrain). Ideal by default.
        self.drivetrain = drivetrain if drivetrain is not None else Drivetrain()

    def tension_curve(self, speed_radial, input_depower=None):
        """Nominal tether force model as a CasADi function f(v_r).

        Uses `pattern_config` to choose the shape and optional smoothing:
        - force_model: "linear" or "quadratic" (default: "quadratic")
        - max_tether_force: required (N)
        - min_tether_force: optional (N, default 0)
        - softplus / softminus: optional boolean flags
        - softplus_beta / softminus_beta: optional sharpness parameters

        Reel-in-capable law ("quadratic" with ``min_tether_force > 0``, which
        is every force-limited reel-out request): instead of softminus, the
        law below ``min_tether_force`` is a straight line through
        ``(offset, min_tether_force)`` and ``(offset + v_reel_in, 0)``, handed
        over to the plain quadratic reel-out law by a smooth maximum
        (sharpness ``reel_in_beta``, default 20) once the quadratic law grows
        past the line. ``v_reel_in`` (< 0, default -2.0) is the reel-in speed
        at zero force. Mirrors WinchControllers.jl's ``calc_vro_soft`` under
        ``soft_lfc``, in the forward direction (this method computes force
        from speed; that one inverts it).

        use_awe_trim (float in [0, 1], default 0, "quadratic" only) blends
        that law with the softminus-floored one it replaced: 0 (the default)
        is the reel-in law above, 1 is the plain quadratic under softminus,
        and values between mix the two forces linearly. The endpoints differ
        most at zero speed, where the reel-in law holds ``min_tether_force``
        while softminus holds ``sp(beta*min_tether_force)/beta`` — 883 N for
        a 350 N ``min_tether_force`` at beta 1e-3, and never below
        ``log(2)/beta`` whatever ``min_tether_force`` is asked for.

        The reel-in law needs no softminus smoothing of its own corner: the
        forward quadratic has zero (not infinite) slope at its own zero, so
        the smooth maximum stays well conditioned for any
        ``reel_in_beta > 0``.

        Depower-dependent offset (the key to flying a full pumping cycle as a
        single phase): the winch's zero-force reeling speed ``offset`` is shifted
        by ``winch_offset_depower_gain * (input_depower - winch_depower_ref)``.
        A single force law then spans both reel-out (powered, low ``l_dp``) and
        reel-in (depowered, high ``l_dp``): with a negative gain the offset moves
        down as the kite is depowered, so the same balance that held high tension
        during reel-out yields a negative reeling speed during reel-in. The shift
        is identity (legacy behaviour) unless ``winch_offset_depower_gain`` is set
        in ``pattern_config`` AND ``input_depower`` is supplied. The config keys
        deliberately avoid the ``offset_winch_`` / ``slope_winch_`` prefixes so
        they are not mistaken for the base offset/slope below.
        """

        model = self.pattern_config.get("force_model", "quadratic")
        max_tf = self.pattern_config.get("max_tether_force", None)
        min_tf = self.pattern_config.get("min_tether_force", 0)
        use_awe_trim = self.pattern_config.get("use_awe_trim", 0.0)
        if use_awe_trim and model != "quadratic":
            raise ValueError(
                "use_awe_trim requires force_model 'quadratic', got "
                f"'{model}'"
            )
        if not (0.0 <= use_awe_trim <= 1.0):
            raise ValueError(f"use_awe_trim must be in [0, 1], got {use_awe_trim}")

        if max_tf is None:
            if model != "custom_spline":
                raise ValueError(
                    "pattern_config must define 'max_tether_force' for tension_curve"
                )
        if min_tf is None:
            if model != "custom_spline":
                raise ValueError(
                    "pattern_config must define 'min_tether_force' for tension_curve"
                )

        if model in ["linear", "quadratic"]:
            slope, offset = self._slope_offset(model, input_depower)

            speed_radial = self._undo_v_sat_clamp(speed_radial)

            if model == "linear":
                T = slope * (speed_radial - offset)
            else:  # quadratic
                T = slope * (speed_radial - offset) * (speed_radial - offset)
        elif model == "custom_spline":
            # User-defined spline model
            if (
                "v_knots" not in self.pattern_config
                or "C_fitted" not in self.pattern_config
            ):
                raise ValueError(
                    "pattern_config must define 'v_knots' and 'C_fitted' for custom_spline force_model"
                )
            spline_model = ca.interpolant(
                "custom_T_spline",
                "bspline",
                [self.pattern_config["v_knots"]],
                np.array(self.pattern_config["C_fitted"]),
            )
            T = spline_model(speed_radial)
        else:
            raise ValueError(f"Unknown force_model '{model}' in pattern_config")

        def softplus_cap(expr):
            if not self.pattern_config.get("softplus", False):
                return expr
            beta = self.pattern_config.get(
                "softplus_beta", DEFAULT_WINCH_CONFIG.get("sharpness_beta", 1e-3)
            )
            return expr - (1 / beta) * ca.log(1 + ca.exp(beta * (expr - max_tf)))

        def softminus_floor(expr):
            if not self.pattern_config.get("softminus", False):
                return expr
            beta = self.pattern_config.get(
                "softminus_beta", DEFAULT_WINCH_CONFIG.get("sharpness_beta", 1e-3)
            )
            return expr + (1 / beta) * ca.log(1 + ca.exp(beta * (min_tf - expr)))

        # A zero min_tether_force leaves no reel-in line to build (its two
        # defining points collapse), so that case keeps the plain law.
        if model == "quadratic" and min_tf:
            v_reel_in = self.pattern_config.get("v_reel_in", -2.0)
            if v_reel_in >= 0:
                raise ValueError(
                    f"the reel-in law requires v_reel_in < 0, got {v_reel_in}"
                )
            reel_in_beta = self.pattern_config.get("reel_in_beta", 20.0)
            m = -v_reel_in / min_tf
            line = min_tf + (speed_radial - offset) / m
            v_eff = ca.fmax(speed_radial - offset, 0)
            quad_ro = slope * v_eff * v_eff
            gap = ca.fabs(line - quad_ro)
            T_reel_in = (
                ca.fmax(line, quad_ro)
                + ca.log(1 + ca.exp(-reel_in_beta * gap)) / reel_in_beta
            )
            T_reel_in = ca.fmax(softplus_cap(T_reel_in), 0)
            if not use_awe_trim:
                return T_reel_in
            return (1 - use_awe_trim) * T_reel_in + use_awe_trim * softminus_floor(
                softplus_cap(T)
            )

        return softminus_floor(softplus_cap(T))

    def _slope_offset(self, model, input_depower=None):
        """``slope_winch_*`` and ``offset_winch_*`` of a linear/quadratic law."""
        # Find offset and slope with winch prefix in pattern_config
        offset = None
        slope = None
        for key in self.pattern_config:
            if key.startswith("offset_winch_"):
                offset = self.pattern_config[key]
            elif key.startswith("slope_winch_"):
                slope = self.pattern_config[key]

        if slope is None:
            raise ValueError(
                f"No slope_winch_* parameter found in pattern_config (required for {model} force model)"
            )

        # Use found offset or default to 0
        offset = 0 if offset is None else offset

        # Depower-dependent shift of the zero-force reeling speed. Lets one
        # force law cover reel-out and reel-in within a single phase (see
        # tension_curve's docstring). gain < 0 -> reel-in as l_dp grows.
        gain = self.pattern_config.get("winch_offset_depower_gain", None)
        if gain is not None and input_depower is not None:
            dep_ref = self.pattern_config.get("winch_depower_ref", 0.0)
            offset = offset + gain * (input_depower - dep_ref)
        return slope, offset

    def uses_speed_law(self):
        """Whether :meth:`radial_equation` blends in :meth:`speed_law`.

        Only for the case that law is the exact inverse of
        :meth:`tension_curve`: a quadratic law under the soft v_sat clamp
        (``v_sat`` and ``v_sat_beta`` set) with ``use_awe_trim == 1``, i.e. the
        plain quadratic under softplus/softminus. The reel-in blend below 1 has
        no closed-form inverse.
        """
        pc = self.pattern_config
        return (
            pc.get("reeling_strategy") == "force"
            and pc.get("force_model", "quadratic") == "quadratic"
            and pc.get("v_sat") is not None
            and pc.get("v_sat_beta") is not None
            and float(pc.get("use_awe_trim", 0.0)) == 1.0
            and bool(pc.get("min_tether_force"))
        )

    def speed_law(self, tension, input_depower=None):
        """The winch controller's own law: reel speed as a function of force.

        The exact inverse of :meth:`tension_curve` on its reel-out branch
        (speed >= offset) when :meth:`uses_speed_law` holds — the law
        WinchControllers.jl's ``calc_vro_soft`` evaluates at runtime. The two
        soft saturations are undone in the reverse of the order they are
        applied (softminus first, then softplus), then the soft v_sat clamp is
        applied. Near ``v_sat`` this curve is FLAT in force where
        :meth:`tension_curve` is vertical in speed.

        Guarded so it stays finite for any tension: below the softminus floor
        the speed bottoms out just above ``offset`` (at the tension
        ``SPEED_LAW_T_MIN``), and at or above
        ``max_tether_force`` it is ``v_sat`` (minus the clamp's own rounding).
        """
        pc = self.pattern_config
        slope, offset = self._slope_offset("quadratic", input_depower)
        max_tf = pc["max_tether_force"]
        min_tf = pc["min_tether_force"]
        t = tension
        if pc.get("softminus", False):
            beta = pc.get("softminus_beta", DEFAULT_WINCH_CONFIG.get("sharpness_beta", 1e-3))
            t = min_tf + _sp_inv(beta * (t - min_tf)) / beta
        if pc.get("softplus", False):
            beta = pc.get("softplus_beta", DEFAULT_WINCH_CONFIG.get("sharpness_beta", 1e-3))
            t = max_tf - _sp_inv(beta * (max_tf - t)) / beta
        v_raw = offset + ca.sqrt(ca.fmax(t, SPEED_LAW_T_MIN) / slope)
        v_sat = pc["v_sat"]
        beta = pc["v_sat_beta"]
        # soft_min(v_raw, v_sat, beta), as WinchControllers.jl's _clamp_v_sat
        return ca.fmin(v_raw, v_sat) - ca.log1p(ca.exp(-beta * ca.fabs(v_raw - v_sat))) / beta

    def _undo_v_sat_clamp(self, speed_radial):
        """Reel speed the law would ask for WITHOUT the soft v_sat clamp.

        WinchControllers.jl's ``calc_vro_soft`` soft-clamps the speed its
        tension-curve inverse commands: ``v = soft_min(v_raw, v_sat, beta)``
        (``_clamp_v_sat``, ``beta = v_sat_beta``). That soft minimum is a
        log-sum-exp, ``exp(-beta*v) = exp(-beta*v_raw) + exp(-beta*v_sat)``,
        so it inverts in closed form:

            v_raw = v - log(1 - exp(-beta*(v_sat - v))) / beta

        Feeding ``v_raw`` to the force law makes the tension rise steeply
        towards ``max_tether_force`` as the speed approaches ``v_sat``, the
        vertical end of the controller's curve, instead of stopping short of it
        at the ``speed_radial`` bound. Identity unless ``pattern_config``
        carries both ``v_sat`` and ``v_sat_beta``.

        ``v_raw`` diverges AT ``v_sat``, which is also the upper speed bound, so
        the gap is floored at ``V_SAT_GAP_MIN``: there ``v_raw`` exceeds ``v``
        by ``-log(beta*V_SAT_GAP_MIN)/beta`` (1.15 m/s at beta = 10), enough
        for the softplus cap to sit within a few newtons of max_tether_force,
        while the result stays finite for IPOPT.
        """
        v_sat = self.pattern_config.get("v_sat", None)
        beta = self.pattern_config.get("v_sat_beta", None)
        if v_sat is None or beta is None:
            return speed_radial
        gap = ca.fmax(v_sat - speed_radial, V_SAT_GAP_MIN)
        return speed_radial - ca.log(-ca.expm1(-beta * gap)) / beta

    def radial_equation(
        self,
        speed_radial=None,
        tension_tether_ground=None,
        input_depower=None,
        speed_radial_rate=None,
    ):
        """Algebraic equation for radial dynamics when using force control.

        Args:
            self: Winch object
            kite_model: Kite object
            input_depower: optional depower input forwarded to
                :meth:`tension_curve` so the force law's offset can depend on the
                depower setting (see that method).
            speed_radial_rate: the drum's dv_r/dt from the caller's time
                scheme, for the inertia of :attr:`drivetrain`; omitted =
                steady drum.
        Returns:
            radial_equation: Algebraic equation for radial dynamics, in newtons

        With a non-ideal :attr:`drivetrain` the law applies to the MOTOR force
        ``T - F_f(v) - m dv/dt`` instead of the tether tension T.

        When :meth:`uses_speed_law` holds, the residual blends the force form
        ``F - T(v)`` with the speed form ``K * (V(F) - v)`` (``V`` =
        :meth:`speed_law`, ``K = max_tether_force / v_sat`` to give it force
        units), by a weight ``w(F)`` rising from 0 to 1 around
        ``SPEED_LAW_BLEND_CENTER`` of ``[min_tether_force, max_tether_force]``. On the reel-out branch both
        forms are zero exactly on the curve and have the same sign off it, so
        the blend has the SAME zero set as the plain force form — only its
        conditioning changes. Each form is used where it is well conditioned:
        the force form at low force, where ``V`` has the infinite slope of a
        square root at the floor, and the speed form near ``v_sat``, where the
        soft v_sat clamp makes ``T(v)`` nearly vertical and ``V(F)`` is flat
        instead. The pure force form with the clamp failed a 7 m/s startup
        solve in IPOPT that converged without the clamp.
        """

        if self.pattern_config["reeling_strategy"] == "force":
            if speed_radial is None or tension_tether_ground is None:
                raise ValueError(
                    "speed_radial and tension_tether_ground must be provided for force control"
                )
            if not self.drivetrain.is_ideal:
                tension_tether_ground = self.drivetrain.motor_force(
                    tension_tether_ground,
                    speed_radial,
                    acceleration=speed_radial_rate,
                    xp=ca,
                )
            # Use the unified tension_curve property
            tension_curve_val = self.tension_curve(
                speed_radial, input_depower=input_depower
            )
            radial_force_law = tension_tether_ground - tension_curve_val
            if self.uses_speed_law():
                pc = self.pattern_config
                max_tf = pc["max_tether_force"]
                min_tf = pc["min_tether_force"]
                k = max_tf / pc["v_sat"]
                f_mid = min_tf + SPEED_LAW_BLEND_CENTER * (max_tf - min_tf)
                width = SPEED_LAW_BLEND_WIDTH * (max_tf - min_tf)
                # logistic weight, written with tanh to stay finite far out
                w = 0.5 * (1 + ca.tanh((tension_tether_ground - f_mid) / (2 * width)))
                speed_form = k * (
                    self.speed_law(tension_tether_ground, input_depower=input_depower)
                    - speed_radial
                )
                radial_force_law = w * speed_form + (1 - w) * radial_force_law
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
        label=None,
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

        ax.plot(v_r, T_vals, label=label)
        ax.set_xlabel("Reeling speed v_r (m/s)")
        ax.set_ylabel("Tension (N)")
        ax.grid(True)
        ax.legend()

        if created_fig and show:
            plt.show()

        return v_r, T_vals


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    # Example usage and test of Winch class and tension_curve plotting
    example_pattern_config = {
        "reeling_strategy": "force",  # "force" or "constant"
        "force_model": "linear",  # "linear" or "quadratic"
        "reeling_speed": 0,  # m/s, only for constant reeling
        "max_tether_force": 8400,  # N, only for force reeling
        "min_tether_force": 1500.0,  # N, only for force reeling
        "softplus": True,
        "softplus_beta": 1e-3,  # bigger is sharper
        "softminus": True,
        "softminus_beta": 1e-3,  # bigger is sharper
        "slope_winch_force": 5555.55,  # N/(m/s)^2 for quadratic, N/(m/s) for linear
        "offset_winch_force": 0.58,  # m/s
    }

    winch = Winch(pattern_config=example_pattern_config)
    fig, ax = plt.subplots()
    winch.plot_tension_curve(vr_min=-2, vr_max=6, n_points=400, show=False, ax=ax)
    # Example usage and test of Winch class and tension_curve plotting
    example_pattern_config = {
        "reeling_strategy": "force",  # "force" or "constant"
        "force_model": "linear",  # "linear" or "quadratic"
        "reeling_speed": 0,  # m/s, only for constant reeling
        "max_tether_force": 8400,  # N, only for force reeling
        "min_tether_force": 1500.0,  # N, only for force reeling
        "softplus": True,
        "softplus_beta": 1e-2,  # bigger is sharper
        "softminus": True,
        "softminus_beta": 1e-2,  # bigger is sharper
        "slope_winch_force": 5555.55,  # N/(m/s)^2 for quadratic, N/(m/s) for linear
        "offset_winch_force": 0.58,  # m/s
    }
    winch = Winch(pattern_config=example_pattern_config)
    winch.plot_tension_curve(vr_min=-2, vr_max=6, n_points=400, show=True, ax=ax)
    plt.show()

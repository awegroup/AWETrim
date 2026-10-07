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

import logging
import warnings

import casadi as ca
from awetrim.aerodynamics.kcu_drag import CT_TURBINE_OPERATING, KcuDragModel, force_drag_kcu
from awetrim.utils.utils import skew_symmetric
from awetrim.utils.reference_frames import (
    transformation_C_from_W,
    transformation_C_from_A,
    transformation_C_from_K,
)

#: Pre-2026-10 ROM parameter keys, in the code's old sign convention
#: ``alpha_w = alpha_b + angle_pitch_depower`` (i.e. angle_pitch_depower =
#: -theta_b), mapped onto the paper-sign keys ``alpha_w = alpha_b - theta_b``
#: (Cayon, van Deursen, Schmehl 2026, Eqs. 1-2). Each maps to (new key, sign).
LEGACY_AERO_PARAM_KEYS = {
    "angle_pitch_depower_0": ("angle_pitch_tether_0", -1.0),
    "delta_pitch_depower": ("slope_angle_pitch_tether_depower", -1.0),
}


def canonical_aero_params(params):
    """ROM ``aerodynamics.params`` with legacy keys translated.

    The geometric pitch ``theta_b`` between the bridle resultant and the wing
    chord is ``angle_pitch_tether_0 + slope_angle_pitch_tether_depower * u_p``
    [rad, rad per u_p unit]. Legacy files state ``angle_pitch_depower_0`` /
    ``delta_pitch_depower`` with the opposite sign; they are converted here
    with a DeprecationWarning. A file carrying both spellings of one
    parameter is ambiguous and rejected.
    """
    params = dict(params)
    for old, (new, sign) in LEGACY_AERO_PARAM_KEYS.items():
        if old not in params:
            continue
        if new in params:
            raise ValueError(
                f"aerodynamics.params states both {old!r} (legacy) and {new!r}"
            )
        warnings.warn(
            f"ROM parameter {old!r} is deprecated; state {new!r} = "
            f"{sign:+.0f} x {old} instead (paper sign, alpha_w = alpha_b - theta_b)",
            DeprecationWarning,
            stacklevel=3,
        )
        params[new] = sign * params.pop(old)
    return params


def stall_blend(alpha, angle_of_attack_stall, width_stall, xp=ca):
    """Smooth attached -> separated switch, 0 well below the stall, 1 above.

    ``sigma = 0.5 * (1 + tanh((alpha - angle_of_attack_stall) / width_stall))``
    [alpha, stall angle and width in rad]. A ROM coefficient term multiplied
    by the ``stall`` variable acts only past the stall, so
    ``C = C_attached + stall * (C_separated - C_attached)`` stays linear in its
    coefficients. ``xp`` is the math namespace (casadi in the ROM, numpy for
    identification and plots) -- this is the one place the formula lives.
    """
    return 0.5 * (1.0 + xp.tanh((alpha - angle_of_attack_stall) / width_stall))


def _uses_variable(coefficients, name):
    """True if any ``coeffs`` term (``var`` or ``vars`` form) uses ``name``."""
    return any(
        term.get("var") == name or name in term.get("vars", {})
        for terms in coefficients.values()
        for term in terms
    )


class Wing:

    def __init__(self, mass_wing, area_wing, aero_input):
        """
        Initialize the kite system with its parameters.
        """
        self.mass_wing = mass_wing
        # A ROM file that states the area its coefficients are referenced to
        # wins over the system's projected area: a coefficient is only
        # meaningful with the area it was normalised by.
        area_reference = aero_input.get("reference_area")
        if area_reference is not None and abs(float(area_reference) - area_wing) > 1e-6:
            logging.info(
                "ROM coefficients referenced to %.4f m2; overriding the system "
                "wing area %.4f m2",
                float(area_reference),
                area_wing,
            )
        self.area_wing = (
            float(area_reference) if area_reference is not None else area_wing
        )
        self.input_steering = ca.MX.sym("input_steering")
        self.input_depower = ca.MX.sym("input_depower")
        # Geometric pitch theta_b(u_p) between the bridle resultant and the
        # wing chord (paper Eq. 2): alpha_w = alpha_b - theta_b.
        params = canonical_aero_params(aero_input.get("params", {}))
        self.angle_pitch_tether_0 = params.get(
            "angle_pitch_tether_0", ca.MX.sym("angle_pitch_tether_0")
        )
        self.slope_angle_pitch_tether_depower = params.get(
            "slope_angle_pitch_tether_depower",
            ca.MX.sym("slope_angle_pitch_tether_depower"),
        )
        self.aero_params = params
        self.aero_input = aero_input
        # Whether the ROM's C_D already contains the KCU drag (a flight-
        # calibrated ROM: the KCU flew with the wing) or the KCU is a separate
        # bluff body the Kite adds from its hardware (an identified wing-only
        # ROM, so a KCU/turbine swap needs no re-identification). Absent =
        # True, the behaviour of every pre-2026-10 file.
        self.kcu_drag_in_coefficients = bool(
            aero_input.get("kcu_drag_in_coefficients", True)
        )
        # Cached drag parameters for easy external tuning
        self._cd0_param = params.get("CD0", 0)
        self._cd_us_param = None  # optional override for u_s drag term
        self._velocity_apparent_wind_wing = None
        self._angle_of_attack = None
        self._lift_coefficient = None
        self._drag_coefficient = None

    def set_drag_params(self, cd0=None, cd_us=None):
        """Optionally override CD0 and the u_s drag coefficient for tuning."""
        if cd0 is not None:
            self._cd0_param = cd0
        if cd_us is not None:
            self._cd_us_param = cd_us

    @property
    def drag_params(self):
        return {"CD0": self._cd0_param, "CD_us": self._cd_us_param}

    @property
    def aerodynamic_force_coefficients(self):
        return self.aerodynamic_force_coefficients_for(self)

    def aerodynamic_force_coefficients_for(self, model):

        aero_input = self.aero_input

        # Define symbolic variables
        variables = {
            "alpha": self.angle_of_attack_for(model),
            "u_s": model.input_steering,
            "u_p": model.input_depower,
        }
        # Also support derived variables
        variables["alpha_squared"] = variables["alpha"] ** 2
        # Apparent wind speed [m/s]: the load level of an aeroelastic
        # (deforming) wing, whose polar shifts with dynamic pressure. Built
        # only when a term uses it, so rigid-wing ROMs keep their graph.
        if _uses_variable(aero_input.get("coefficients", {}), "v_a"):
            vec_va = model.velocity_apparent_wind
            variables["v_a"] = ca.sqrt(ca.mtimes(vec_va.T, vec_va) + 1e-10)
        if _uses_variable(aero_input.get("coefficients", {}), "stall"):
            missing = [k for k in ("angle_of_attack_stall", "width_stall")
                       if k not in self.aero_params]
            if missing:
                raise ValueError(f"ROM 'stall' terms need params {missing}")
            variables["stall"] = stall_blend(
                variables["alpha"],
                self.aero_params["angle_of_attack_stall"],
                self.aero_params["width_stall"],
            )

        # Inviscid model
        if aero_input["model"] == "inviscid":
            e = self.aero_params["oswald_efficiency"]
            AR = self.aero_params["aspect_ratio"]
            CD0 = self.aero_params["CD0"]
            C_L0 = 2 * ca.pi * variables["alpha"] / (1 + 2 / (AR * e))
            C_D = C_L0**2 / (ca.pi * e * AR) + CD0
            # Decompose the wing lift into lift/side components when rolled;
            # both must come from the unrotated C_L0, not the rotated C_L.
            roll = model.input_steering * self.k_steering
            C_L = C_L0 * ca.cos(roll)
            C_S = C_L0 * ca.sin(roll)
            return C_L, C_D, C_S

        # Coeff-based model
        elif aero_input["model"] == "coeffs":
            C_L = self.aero_params.get("CL0", 0)
            C_D = self._cd0_param
            C_S = self.aero_params.get("CS0", 0)

            # Loop over defined terms per coefficient
            for coeff_key, terms in aero_input.get("coefficients", {}).items():
                for term in terms:
                    coef = term["coef"]
                    if "vars" in term:
                        value = 1
                        for var, power in term["vars"].items():
                            if var not in variables:
                                value = None
                                break
                            value *= variables[var] ** power
                    else:
                        var = term["var"]
                        power = term.get("power", 1)
                        if (
                            coeff_key == "CD"
                            and var == "u_s"
                            and self._cd_us_param is not None
                        ):
                            coef = self._cd_us_param
                        value = variables[var] ** power if var in variables else None
                    if value is None:
                        continue
                    # Every term is a plain monomial. ``abs: true`` asks for
                    # |monomial| (e.g. a steering drag symmetric in left/right
                    # u_s); it has a kink at zero, so use it deliberately.
                    # Until 2026-10 EVERY C_D term was |monomial|, which bent
                    # each odd-alpha drag term into a V at alpha = 0.
                    if term.get("abs", False):
                        value = ca.sqrt(value**2 + 1e-10)
                    if coeff_key == "CL":
                        C_L += coef * value
                    elif coeff_key == "CD":
                        C_D += coef * value
                    elif coeff_key == "CS":
                        C_S += coef * value
            return C_L, C_D

        else:
            raise ValueError(
                "Invalid aerodynamic model type. Choose 'inviscid' or 'coeffs'."
            )

    @property
    def lift_coefficient(self):
        if self._lift_coefficient is None:
            self._lift_coefficient = self.aerodynamic_force_coefficients[0]
        return self._lift_coefficient

    def lift_coefficient_for(self, model):
        return self.aerodynamic_force_coefficients_for(model)[0]

    @property
    def drag_coefficient(self):
        if self._drag_coefficient is None:
            self._drag_coefficient = self.aerodynamic_force_coefficients[1]
        return self._drag_coefficient

    def drag_coefficient_for(self, model):
        return self.aerodynamic_force_coefficients_for(model)[1]

    @property
    def angle_pitch_tether(self):
        """Geometric pitch theta_b between the bridle resultant and the wing chord [rad]."""
        return self.angle_pitch_tether_for(self)

    def angle_pitch_tether_for(self, model):
        return (
            self.angle_pitch_tether_0
            + model.input_depower * self.slope_angle_pitch_tether_depower
        )

    @property
    def pitch_bridle(self):
        return self.pitch_bridle_for(self)

    def force_bridle_for(self, model):
        """Resultant the bridle lines carry to the wing: tether + KCU loads.

        The KCU hangs at the bridle point, so its weight, inertia and (when
        explicit) drag enter here, not through the wing's lift and drag.
        """
        force_kcu = (
            model.force_gravity_kcu
            - model.kite.mass_kcu * model.acceleration
            + model.kite.force_drag_kcu_for(model)
        )
        return model.force_tether_at_kite + force_kcu

    def force_drag_kcu_for(self, model):
        """Explicit KCU drag [N]; zero for a bare wing."""
        return ca.MX.zeros(3, 1)

    @staticmethod
    def _force_in_wind_frame(model, force):
        """Components of ``force`` in the apparent-wind frame of paper App. D.

        Axes: x = -v_a/|v_a| (into the wind, paper e_chi'), y = -e_n with
        e_n = e_r x v_a / |.| (paper e_n'), z = e_r' = e_r projected
        perpendicular to v_a. Built from the apparent wind VECTOR, like the
        lift direction in :meth:`force_aerodynamic_wing`, so the bridle pitch
        (paper Eq. D11) and bridle roll (Eqs. D6-D8) are taken about v_a.
        The Euler-angle route (transformation_C_from_A with
        angle_yaw_aerodynamic = -atan(v_y/v_x)) is NOT this frame: its yaw
        sign is mirrored for that composition, so its x axis sat twice the
        sideslip angle off v_a in the tangent plane. That leaked the normal
        component of the bridle resultant (the KCU's inertia in a turn) into
        the bridle pitch -- 2-3 deg of alpha_b with a sign that follows the
        sideslip -- and into the bridle roll (about 1 deg), zero only at zero
        sideslip, i.e. at the centre-window identification states.
        """
        vec_va = model.velocity_apparent_wind
        va = ca.sqrt(ca.mtimes(vec_va.T, vec_va) + 1e-10)
        va_tau = ca.sqrt(vec_va[0] ** 2 + vec_va[1] ** 2 + 1e-10)
        x_axis = -vec_va / va
        y_axis = ca.vertcat(vec_va[1], -vec_va[0], 0.0) / va_tau
        z_axis = ca.vertcat(
            -vec_va[0] * vec_va[2], -vec_va[1] * vec_va[2], va_tau**2
        ) / (va * va_tau)
        return ca.vertcat(
            ca.dot(force, x_axis), ca.dot(force, y_axis), ca.dot(force, z_axis)
        )

    def pitch_bridle_for(self, model):
        """Bridle pitch alpha_b, paper Eq. D11: the angle of the bridle
        resultant from -e_r' toward e_chi' (= -v_a)."""
        tow_line = self._force_in_wind_frame(model, self.force_bridle_for(model))
        angle_bridle = ca.atan2(tow_line[0], -tow_line[2] + 1e-6)
        return angle_bridle

    @property
    def roll_bridle(self):
        return self.roll_bridle_for(self)

    def roll_bridle_for(self, model):
        """Bridle roll phi_b, paper Eqs. D6-D8: the roll of the bridle
        resultant about v_a (0 along -e_r'), the reference the wing's
        steering roll k u_s is measured from."""
        tow_line = self._force_in_wind_frame(model, self.force_bridle_for(model))
        angle_bridle = ca.atan2(-tow_line[1], -tow_line[2] + 1e-6)
        return angle_bridle

    @property
    def angle_of_attack(self):
        """
        Compute the angle of attack based on the air velocity vector and tether angle.
        """
        self._angle_of_attack = self.angle_of_attack_for(self)

        return self._angle_of_attack

    def angle_of_attack_for(self, model):
        # Paper Eq. 1: alpha_w = alpha_b - theta_b.
        return self.pitch_bridle_for(model) - self.angle_pitch_tether_for(model)

    def force_aerodynamic(self, model):
        """Total aerodynamic force on the kite [N, C-frame]."""
        return self.force_aerodynamic_wing(model)

    def force_aerodynamic_wing(self, model):
        """
        Compute the aerodynamic forces based on the aerodynamic coefficients.
        """
        vec_va = model.velocity_apparent_wind
        va_sq = ca.mtimes(vec_va.T, vec_va)
        # Guard the sqrt arguments: the value is finite at zero but the
        # derivative (1 / 2*sqrt(arg)) is NaN, which poisons jac_g and stalls
        # IPOPT if an iterate steps through zero apparent wind. Mirrors the
        # +1e-10 guards on the divisions below and in the C_D coefficient.
        va = ca.sqrt(va_sq + 1e-10)

        CL, CD = self.aerodynamic_force_coefficients_for(model)[:2]

        va_tau = ca.sqrt(vec_va[0] ** 2 + vec_va[1] ** 2 + 1e-10)
        roll_total = self.angle_roll_aerodynamic_for(model) + self.roll_bridle_for(
            model
        )
        lift_direction = ca.vertcat(
            va * vec_va[1] * ca.sin(roll_total)
            - vec_va[2] * vec_va[0] * ca.cos(roll_total),
            -va * vec_va[0] * ca.sin(roll_total)
            - vec_va[2] * vec_va[1] * ca.cos(roll_total),
            va_tau**2 * ca.cos(roll_total),
        ) / (va * va_tau + 1e-10)
        drag_direction = vec_va / (va + 1e-10)
        # Aerodynamic forces
        D = 0.5 * self.rho * va_sq * self.area_wing * CD
        L = 0.5 * self.rho * va_sq * self.area_wing * CL

        aero_forces = D * drag_direction + L * lift_direction
        return aero_forces

    @property
    def force_gravity_wing(self):
        return self.force_gravity_wing_for(self)

    def force_gravity_wing_for(self, model):

        return (
            -self.mass_wing
            * self.g
            * ca.vertcat(
                ca.cos(model.angle_elevation) * ca.cos(model.angle_course),
                ca.cos(model.angle_elevation) * ca.sin(model.angle_course),
                ca.sin(model.angle_elevation),
            )
        )


class Kite(Wing):

    def __init__(
        self,
        mass_wing,
        area_wing,
        aero_input,
        mass_kcu=0,
        length_kcu=0.0,
        diameter_kcu=0.0,
        diameter_turbine=0.0,
        thrust_coefficient_turbine=0.85,
        g=9.81,
        rho=1.225,
        center_aerodynamic_wing=[0, 0, 10],
        center_gravity_wing=[0, 0, 10],
        steering_control="roll",
    ):
        """
        Initialize the kite system with its parameters.
        """

        super().__init__(mass_wing, area_wing, aero_input)
        self.mass_kcu = mass_kcu  # Mass of the kite control unit
        # KCU envelope [m]; only the VSM trims use it, for the bluff-body drag
        # in awetrim.aerodynamics.kcu_drag. 0.0 = unknown = no KCU drag, which
        # keeps every Kite built without these (tests, legacy configs, design
        # tools) exactly as it was.
        self.length_kcu = float(length_kcu or 0.0)
        self.diameter_kcu = float(diameter_kcu or 0.0)
        # Onboard wind turbine (mounted ON TOP of the KCU, facing the
        # apparent wind); only the KCU drag model reads these. 0.0 diameter =
        # no turbine, and its mass is already inside mass_kcu -- these are
        # DRAG parameters only.
        self.diameter_turbine = float(diameter_turbine or 0.0)
        self.thrust_coefficient_turbine = float(thrust_coefficient_turbine or 0.0)
        # Explicit KCU bluff-body drag (awetrim.aerodynamics.kcu_drag, the
        # single source) for a ROM whose C_D excludes it; None = none added.
        self.kcu_drag_model = None
        if not self.kcu_drag_in_coefficients:
            self.kcu_drag_model = KcuDragModel.from_dimensions(
                self.length_kcu,
                self.diameter_kcu,
                diameter_turbine=self.diameter_turbine,
                thrust_coefficient_turbine=(
                    self.thrust_coefficient_turbine or CT_TURBINE_OPERATING
                ),
            )
            if self.kcu_drag_model is None:
                logging.warning(
                    "ROM C_D excludes the KCU drag but the system has no KCU "
                    "length/diameter: the KCU drag is NOT modelled."
                )
        self.steering_control = steering_control
        self.g = g  # Gravitational acceleration
        self.rho = rho  # Air density
        self.center_aerodynamic_wing = (
            center_aerodynamic_wing  # Center of aerodynamic pressure
        )
        self.center_gravity_wing = center_gravity_wing  # Center of gravity

        # Add these missing symbolic variables
        self.pitch_kcu = ca.MX.sym("pitch_kcu")
        self.roll_kcu = ca.MX.sym("roll_kcu")
        self._override_gravity = False
        self._override_centripetal = False
        self._override_coriolis = False

        # Steering -> aerodynamic roll gain, phi_a,w = gain_roll_steering * u_s
        # [rad per u_s] (paper Eq. 43, k_phi,s), for asymmetric steering: an
        # explicit parameter wins, legacy files carry it as minus the CS
        # coefficient of u_s. Roll steering takes u_s as the roll angle itself.
        if self.steering_control == "asymmetric":
            gain = self.aero_params.get("gain_roll_steering")
            if gain is None:
                cs_terms = aero_input.get("coefficients", {}).get("CS", [])
                gain = -next(
                    (term["coef"] for term in cs_terms if term.get("var") == "u_s"),
                    0.0,
                )
            self.gain_roll_steering = gain
        else:
            self.gain_roll_steering = 1.0
        # Sideslip -> aerodynamic roll, phi_a,w += gain_roll_sideslip * beta_a
        # [rad per rad], beta_a = angle_sideslip_for(model): the restoring roll
        # (weathercock / sideforce) of a wing flying in sideslip, which the
        # point-mass lift (perpendicular to v_a in the (v_a, e_r) plane) has
        # no other way to carry. Measured on the 2019 flight as +0.05 rad/rad
        # (held-out R2 of the roll 0.93 -> 0.95); absent/0 = the paper's model.
        self.gain_roll_sideslip = float(self.aero_params.get("gain_roll_sideslip", 0.0) or 0.0)

        self._acceleration_total = None  # Cache for total acceleration

    def force_drag_kcu_for(self, model):
        """KCU drag [N, C-frame], hanging on the tether (axis = e_r).

        The apparent wind at the kite point (the ROM is a point mass; the
        rotational correction of apparent_wind.py is a VSM-trim refinement).
        """
        if self.kcu_drag_model is None:
            return ca.MX.zeros(3, 1)
        return force_drag_kcu(
            model.velocity_apparent_wind,
            [0.0, 0.0, 1.0],
            self.rho,
            self.kcu_drag_model.cd_area_axial,
            self.kcu_drag_model.cd_area_broadside,
            xp=ca,
        )

    def force_aerodynamic(self, model):
        """Wing aerodynamic force plus the explicit KCU drag [N, C-frame]."""
        return self.force_aerodynamic_wing(model) + self.force_drag_kcu_for(model)

    @property
    def k_steering(self):
        """Deprecated alias of :attr:`gain_roll_steering`."""
        return self.gain_roll_steering

    @property
    def angle_roll(self):
        return self.roll_kcu

    @property
    def angle_roll_aerodynamic(self):
        return self.angle_roll_aerodynamic_for(self)

    def angle_roll_aerodynamic_for(self, model):
        roll = model.input_steering * self.gain_roll_steering
        if self.gain_roll_sideslip:
            roll = roll + self.gain_roll_sideslip * self.angle_sideslip_for(model)
        return roll

    @staticmethod
    def angle_sideslip_for(model):
        """Course-frame sideslip beta_a [rad]: the angle of the oncoming
        apparent wind (-v_a) from the course direction, in the tangent plane,
        positive toward -e_n (atan2(-v_a,n, -v_a,chi))."""
        vec_va = model.velocity_apparent_wind
        return ca.atan2(-vec_va[1], -vec_va[0] + 1e-10)

    @property
    def angle_pitch(self):
        return self.pitch_kcu

    @property
    def force_gravity_kcu(self):
        return self.force_gravity_kcu_for(self)

    def force_gravity_kcu_for(self, model):

        T = transformation_C_from_W(
            model.angle_azimuth, model.angle_elevation, model.angle_course
        )
        return T @ ca.vertcat(0, 0, -self.mass_kcu * self.g)

    @property
    def force_gravity(self):
        if self._override_gravity == True:
            return ca.vertcat(0, 0, 0)
        return self.force_gravity_wing + self.force_gravity_kcu

    def force_gravity_for(self, model):
        if getattr(model, "override_gravity", False) == True:
            return ca.vertcat(0, 0, 0)
        return self.force_gravity_wing_for(model) + self.force_gravity_kcu_for(model)

    @property
    def override_gravity(self):
        return self._override_gravity

    @override_gravity.setter
    def override_gravity(self, value):
        if not isinstance(value, bool):
            raise ValueError("override_gravity ha de ser True o False.")
        self._override_gravity = value

    @property
    def override_centripetal(self):
        return self._override_centripetal

    @override_centripetal.setter
    def override_centripetal(self, value):
        if not isinstance(value, bool):
            raise ValueError("override_gravity ha de ser True o False.")
        self._override_centripetal = value

    @property
    def override_coriolis(self):
        return self._override_coriolis

    @override_coriolis.setter
    def override_coriolis(self, value):
        if not isinstance(value, bool):
            raise ValueError("override_gravity ha de ser True o False.")
        self._override_coriolis = value

    @property
    def acceleration_external(self):
        acc = self.force_external / (self.mass_wing + self.mass_kcu)
        vtau = self.speed_tangential
        acc[1] = -acc[1] / (vtau + 1e-6)
        # acc[1] = ca.if_else(
        #     vtau > 1e-3,
        #     -acc[1] / vtau,
        #     -ca.sign(acc[1] + 1e-6) * 1,
        # )
        return acc

    @property
    def acceleration_inertial(self):
        return ca.vertcat(
            -self.speed_tangential * self.speed_radial / self.distance_radial,
            self.speed_tangential
            * ca.sin(self.angle_course)
            * ca.tan(self.angle_elevation)
            / self.distance_radial,
            self.speed_tangential**2 / self.distance_radial,
        )

    @property
    def acceleration_rotation_course(self):
        if self._override_centripetal == True:
            return ca.vertcat(
                self.speed_tangential * self.speed_radial / self.distance_radial, 0, 0
            )
        if self._override_coriolis == True:
            return ca.cross(
                self.velocity_rotation_course_frame, self.velocity_kite
            ) - ca.vertcat(
                2 * self.speed_tangential * self.speed_radial / self.distance_radial,
                0,
                0,
            )
        return ca.cross(self.velocity_rotation_course_frame, self.velocity_kite)

    @property
    def acceleration_local(self):
        return ca.vertcat(self.timeder_speed_tangential, 0, self.timeder_speed_radial)

    @property
    def acceleration(self):
        return self.acceleration_local + self.acceleration_rotation_course

    @property
    def acceleration_total(self):
        if self._acceleration_total is None:
            self._acceleration_total = (
                self.acceleration_inertial + self.acceleration_external
            )
        return self._acceleration_total

    @property
    def angle_yaw(self):
        return self.angle_yaw_aerodynamic

    @property
    def velocity_wind(self):
        """Wind velocity at the kite position in the kite frame."""
        # Wind velocity in wind frame
        return self.wind.velocity_wind(self)


def project_onto_plane(v, n):
    n_norm2 = ca.dot(n, n) + 1e-12  # avoid div by zero
    return v - (ca.dot(n, v) / n_norm2) * n

import casadi as ca
from picawe.utils.utils import skew_symmetric
from picawe.utils.reference_frames import (
    transformation_C_from_W,
    transformation_C_from_A,
    transformation_C_from_K,
)


class Wing:

    def __init__(self, mass_wing, area_wing, aero_input):
        """
        Initialize the kite system with its parameters.
        """
        self.mass_wing = mass_wing
        self.area_wing = area_wing
        self.input_steering = ca.SX.sym("input_steering")
        self.input_depower = ca.SX.sym("input_depower")
        # Aerodynamic inputs
        self.angle_pitch_tether = aero_input["params"].get(
            "angle_pitch_depower_0", ca.SX.sym("angle_pitch_tether")
        )
        self.delta_pitch_depower = aero_input["params"].get(
            "delta_pitch_depower", ca.SX.sym("delta_pitch_depower")
        )
        # self.aerodynamic_coeffs_function(aero_input)
        self.aero_input = aero_input
        self._velocity_apparent_wind_wing = None
        self._angle_of_attack = None
        self._lift_coefficient = None
        self._drag_coefficient = None

    @property
    def aerodynamic_force_coefficients(self):
        import casadi as ca

        aero_input = self.aero_input

        # Define symbolic variables
        variables = {
            "alpha": self.angle_of_attack,
            "u_s": self.input_steering,
            "u_p": self.input_depower,
            "yaw_rate": self.timeder_angle_yaw / ca.norm_2(self.velocity_apparent_wind),
            "sideslip": self.angle_sideslip,
        }
        # Also support derived variables
        variables["alpha_squared"] = variables["alpha"] ** 2

        # Inviscid model
        if aero_input["model"] == "inviscid":
            e = aero_input["params"]["oswald_efficiency"]
            AR = aero_input["params"]["aspect_ratio"]
            CD0 = aero_input["params"]["CD0"]
            C_L = 2 * ca.pi * variables["alpha"]
            C_D = C_L**2 / (ca.pi * e * AR) + CD0
            C_L = C_L * ca.cos(self.input_steering * self.k_steering)
            C_S = C_L * ca.sin(self.input_steering * self.k_steering)
            return C_L, C_D, C_S

        # Coeff-based model
        elif aero_input["model"] == "coeffs":
            C_L = aero_input["params"].get("CL0", 0)
            C_D = aero_input["params"].get("CD0", 0)
            C_S = aero_input["params"].get("CS0", 0)

            # Loop over defined terms per coefficient
            for coeff_key, terms in aero_input.get("coefficients", {}).items():
                for term in terms:
                    var = term["var"]
                    power = term.get("power", 1)
                    coef = term["coef"]
                    if var in variables:
                        value = variables[var] ** power
                        if coeff_key == "CL":
                            C_L += coef * value
                        elif coeff_key == "CD":
                            C_D += coef * ca.fabs(value)
                        elif coeff_key == "CS":
                            C_S += coef * value
                    # alpha_min = 0 / 180 * ca.pi
                    # alpha_max = 15 / 180 * ca.pi
                    # C_L = ca.if_else(
                    #     variables["alpha"] <= alpha_max,
                    #     C_L,
                    #     1.2,
                    # )
            C_L = C_L * ca.cos(self.input_steering * self.k_steering)
            C_S = C_L * ca.sin(self.input_steering * self.k_steering)
            return C_L, C_D, C_S

        else:
            raise ValueError(
                "Invalid aerodynamic model type. Choose 'inviscid' or 'coeffs'."
            )
            # elif coeff_type == "k_cs":
            # if self.steering_control == "asymmetric":
            #     C_L = C_L*ca.cos(aero_input["dependencies"]["u_s"]["k_cs"]*self.input_steering)
            #     C_S = C_L*ca.sin(aero_input["dependencies"]["u_s"]["k_cs"]*self.input_steering)

    @property
    def lift_coefficient(self):
        if self._lift_coefficient is None:
            self._lift_coefficient = self.aerodynamic_force_coefficients[0]
        return self._lift_coefficient

    @property
    def drag_coefficient(self):
        if self._drag_coefficient is None:
            self._drag_coefficient = self.aerodynamic_force_coefficients[1]
        return self._drag_coefficient

    @property
    def aerodynamic_moment_coefficients(self):
        aero_input = self.aero_input
        # Define symbolic variables
        variables = {
            "alpha": self.angle_of_attack,
            "alpha_squared": self.angle_of_attack**2,
            "u_s": self.input_steering,
            "u_p": self.input_depower,
            # Dynamically add other variables as dependencies
            "yaw_rate": self.timeder_angle_yaw / ca.norm_2(self.velocity_apparent_wind),
            "sideslip": self.angle_sideslip,
        }

        C_m = aero_input["params"].get("C_m_base", 0)  # Pitch moment coefficient
        C_l = aero_input["params"].get("C_l_base", 0)  # Roll moment coefficient
        C_n = aero_input["params"].get("C_n_base", 0)  # Yaw moment coefficient

        # Apply dependencies dynamically for CL, CD, CS, C_m, C_l, and C_n
        for var, coeffs in aero_input.get("dependencies", {}).items():
            for coeff_type, coeff_value in coeffs.items():
                if coeff_type == "k_cm":
                    C_m += coeff_value * variables[var]
                elif coeff_type == "k_cl_roll":
                    C_l += coeff_value * variables[var]
                elif coeff_type == "k_cn":
                    C_n += coeff_value * variables[var]

        return C_m, C_l, C_n

    @property
    def angle_pitch_depower(self):
        """
        Compute the tether angle based on the powered angle and the tether angle at t=0.
        """
        return self.angle_pitch_tether + self.input_depower * self.delta_pitch_depower

    @property
    def angle_sideslip(self):
        """
        Compute the angle_sideslip
        """
        return self.angle_yaw - self.angle_yaw_aerodynamic

    @property
    def angle_of_attack(self):
        """
        Compute the angle of attack based on the air velocity vector and tether angle.
        """
        # print("angle_pitch_aerodynamic:",self.angle_pitch_aerodynamic)
        angle_of_attack = (
            self.angle_pitch_aerodynamic + self.angle_pitch_depower - self.angle_pitch
        )
        if self._angle_of_attack is None:
            self._angle_of_attack = angle_of_attack

        return angle_of_attack

    @property
    def velocity_apparent_wind(self):
        # print("velocity_apparent_wind:", self.velocity_kite)
        # print(self.wind)

        return self.wind.velocity_wind(self) - self.velocity_kite

    @property
    def velocity_rotation_wing(self):
        return ca.vertcat(
            self.timeder_angle_roll, self.timeder_angle_pitch, self.timeder_angle_yaw
        )

    @property
    def velocity_apparent_wind_wing(self):

        velocity_wing_rotation = ca.cross(
            self.velocity_rotation_wing, self.center_gravity_wing_course
        )
        velocity_apparent_wind_wing = (
            self.velocity_apparent_wind - velocity_wing_rotation
        )
        if self._velocity_apparent_wind_wing is None:
            self._velocity_apparent_wind_wing = velocity_apparent_wind_wing

        return velocity_apparent_wind_wing

    @property
    def angle_pitch_aerodynamic(self):
        # print(self.velocity_apparent_wind)
        return ca.atan2(
            self.velocity_apparent_wind_wing[2],
            ca.sqrt(
                self.velocity_apparent_wind_wing[0] ** 2
                + self.velocity_apparent_wind_wing[1] ** 2
            ),
        )

    @property
    def angle_yaw_aerodynamic(self):
        return -ca.atan(
            self.velocity_apparent_wind_wing[1] / self.velocity_apparent_wind_wing[0]
        )

    @property
    def force_aerodynamic(self):
        """
        Compute the aerodynamic forces based on the aerodynamic coefficients.
        """
        V_a_sq = ca.mtimes(self.velocity_apparent_wind.T, self.velocity_apparent_wind)

        CL, CD, CS = self.aerodynamic_force_coefficients

        # Aerodynamic forces
        D = 0.5 * self.rho * V_a_sq * self.area_wing * CD
        L = 0.5 * self.rho * V_a_sq * self.area_wing * CL
        S = 0.5 * self.rho * V_a_sq * self.area_wing * CS

        R = transformation_C_from_A(
            self.angle_pitch_aerodynamic,
            self.angle_yaw_aerodynamic,
            0,
        )
        aero_forces = R @ ca.vertcat(-D, S, L)
        return aero_forces

    @property
    def force_gravity_wing(self):

        return transformation_C_from_W(
            self.angle_azimuth, self.angle_elevation, self.angle_course
        ) @ ca.vertcat(0, 0, -self.mass_wing * self.g)


class Kite(Wing):

    def __init__(
        self,
        mass_wing,
        area_wing,
        aero_input,
        mass_kcu=0,
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
        self.steering_control = steering_control
        self.g = g  # Gravitational acceleration
        self.rho = rho  # Air density
        self.center_aerodynamic_wing = (
            center_aerodynamic_wing  # Center of aerodynamic pressure
        )
        self.center_gravity_wing = center_gravity_wing  # Center of gravity
        self._override_gravity = False
        self._override_centripetal = False
        self._override_coriolis = False
        self._angle_yaw = ca.SX.sym("angle_yaw")
        self._angle_pitch = ca.SX.sym("angle_pitch")
        self._angle_roll = ca.SX.sym("angle_roll")
        self.timeder_angle_yaw = ca.SX.sym("timeder_angle_yaw")
        self.timeder_angle_pitch = ca.SX.sym("timeder_angle_pitch")
        self.timeder_angle_roll = ca.SX.sym("timeder_angle_roll")
        self.acceleration_angle_yaw = ca.SX.sym("acceleration_angle_yaw")
        self.acceleration_angle_pitch = ca.SX.sym("acceleration_angle_pitch")
        self.acceleration_angle_roll = ca.SX.sym("acceleration_angle_roll")
        # print(aero_input)
        if self.steering_control == "asymmetric":
            cs_terms = aero_input["coefficients"].get("CS", [])
            k_steering = -next(
                (term["coef"] for term in cs_terms if term["var"] == "u_s"), 0.0
            )
            self.k_steering = k_steering
        else:
            self.k_steering = 1.0

        self._acceleration_total = None  # Cache for total acceleration

    @property
    def angle_roll(self):
        if self.dof == 6:
            return self._angle_roll
        elif self.dof == 3:
            return self.roll_kcu

    @property
    def angle_roll_aerodynamic(self):
        return self.input_steering * self.k_steering

    @property
    def angle_pitch(self):
        if self.dof == 6:
            return self._angle_pitch
        elif self.dof == 3:
            return self.pitch_kcu

    @property
    def force_gravity_kcu(self):

        T = transformation_C_from_W(
            self.angle_azimuth, self.angle_elevation, self.angle_course
        )
        return T @ ca.vertcat(0, 0, -self.mass_kcu * self.g)

    @property
    def force_gravity(self):
        if self._override_gravity == True:
            return ca.vertcat(0, 0, 0)
        return self.force_gravity_wing + self.force_gravity_kcu

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
    def velocity_rotation(self):
        return ca.vertcat(
            self.timeder_angle_pitch, self.timeder_angle_roll, self.timeder_angle_yaw
        )

    @property
    def acceleration_rotation_kite(self):
        return ca.vertcat(
            self.acceleration_angle_roll,
            self.acceleration_angle_pitch,
            self.acceleration_angle_yaw,
        )

    @property
    def force_external(self):
        # print("force_external:", self.force_aerodynamic, self.force_gravity)

        return self.force_aerodynamic + self.force_gravity + self.force_tether_at_kite

    @property
    def acceleration_external(self):
        acc = self.force_external / (self.mass_wing + self.mass_kcu)
        vtau = self.speed_tangential

        acc[1] = ca.if_else(
            vtau > 1e-3,
            -acc[1] / vtau,
            -ca.sign(acc[1]) * 1,
        )
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
    def acceleration_total(self):
        if self._acceleration_total is None:
            self._acceleration_total = (
                self.acceleration_inertial + self.acceleration_external
            )
        return self._acceleration_total

    @property
    def force_residual(self):
        """
        Compute the residual for the kite system dynamics.
        """
        # LHS and RHS
        lhs = (self.mass_wing + self.mass_kcu) * self.acceleration
        # Residual
        # print(self.force_external)
        # print(lhs)
        return -lhs + self.force_external

    @property
    def aero_moment(self):

        R_C_from_K = transformation_C_from_K(self.angle_pitch, self.angle_roll)

        center_aerodynamic_wing = R_C_from_K @ ca.vertcat(*self.center_aerodynamic_wing)

        Cm, Cl, Cn = self.aerodynamic_moment_coefficients
        aero_moment = ca.cross(center_aerodynamic_wing, self.force_aerodynamic)
        aero_moment[1] += (
            Cm
            * self.area_wing
            * self.rho
            * ca.mtimes(self.velocity_apparent_wind.T, self.velocity_apparent_wind)
            * 2
        )  ### Not the correct formula
        aero_moment[2] += (
            Cn
            * self.area_wing
            * self.rho
            * ca.mtimes(self.velocity_apparent_wind.T, self.velocity_apparent_wind)
            * 2
        )  ### Not the correct formula
        return aero_moment

    @property
    def gravity_moment(self):

        R_C_from_K = transformation_C_from_K(self.angle_pitch, self.angle_roll)

        center_gravity_wing = R_C_from_K @ ca.vertcat(*self.center_gravity_wing)

        return ca.cross(center_gravity_wing, self.force_gravity_wing)

    @property
    def inertia_moment(self):

        R_C_from_K = transformation_C_from_K(self.angle_pitch, self.angle_roll)

        center_gravity_wing = R_C_from_K @ ca.vertcat(*self.center_gravity_wing)

        return ca.cross(center_gravity_wing, (self.mass_wing) * self.acceleration())

    @property
    def moment_external(self):

        return self.aero_moment + self.gravity_moment

    @property
    def rb_residual(self):
        """
        Compute the residual for the kite system dynamics.
        Join the force and moment residuals.
        """
        x_cg_c_cross = skew_symmetric(self.center_gravity_wing_course)
        omega_cross = skew_symmetric(self.velocity_rotation_course_frame)
        I = self.inertia_matrix_course
        m = self.mass_wing + self.mass_kcu
        m_w = self.mass_wing
        # Create the block matrix
        M = ca.vertcat(
            ca.horzcat(m * ca.SX.eye(3), -m_w * x_cg_c_cross),
            ca.horzcat(m_w * x_cg_c_cross, I),
        )

        ROT = ca.vertcat(
            ca.horzcat(omega_cross, ca.SX.zeros(3, 3)),
            ca.horzcat(ca.SX.zeros(3, 3), omega_cross),
        )

        acceleration = ca.vertcat(
            self.acceleration_local, self.acceleration_rotation_kite
        )
        velocity = ca.vertcat(self.velocity_kite, self.velocity_rotation)

        lhs = M @ acceleration + ROT @ M @ velocity
        rhs = ca.vertcat(self.force_external, self.moment_external)

        return lhs - rhs

    @property
    def inertia_matrix_body(self):
        # TODO: Calculate based on the cg position
        return ca.diag([1, 1, 0]) * self.mass_wing * self.center_gravity_wing[2] ** 2

    @property
    def inertia_matrix_course(self):
        return (
            transformation_C_from_K(self.angle_pitch, self.angle_roll)
            @ self.inertia_matrix_body
        )

    @property
    def center_gravity_wing_course(self):
        return transformation_C_from_K(self.angle_pitch, self.angle_roll) @ ca.vertcat(
            *self.center_gravity_wing
        )

    @property
    def angle_yaw(self):

        if self.dof == 3:
            return self.angle_yaw_aerodynamic

        elif self.dof == 6:
            return self._angle_yaw

    @property
    def pitch_kcu(self):

        numerator = self.mass_kcu * (
            self.g * ca.cos(self.angle_elevation) * ca.cos(self.angle_course)
            + (self.speed_tangential * self.speed_radial) / self.distance_radial
        )
        denominator = self.tension_kite + self.mass_kcu * (
            self.g * ca.sin(self.angle_elevation)
            - (self.speed_tangential**2) / self.distance_radial
        )
        return ca.arctan(numerator / denominator)

    @property
    def roll_kcu(self):
        numerator = self.mass_kcu * (
            -(self.speed_tangential**2)
            / self.distance_radial
            * ca.sin(self.angle_course)
            * ca.tan(self.angle_elevation)
            + self.speed_tangential * self.timeder_angle_course
            - self.g * ca.cos(self.angle_elevation) * ca.sin(self.angle_course)
        )
        denominator = self.tension_kite + self.mass_kcu * (
            self.g * ca.sin(self.angle_elevation)
            - (self.speed_tangential**2) / self.distance_radial
        )
        return ca.arctan(numerator / denominator)

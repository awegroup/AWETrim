import casadi as ca


class KiteKinematics:

    def __init__(self):
        self._timeder_speed_tangential = ca.SX.sym("timeder_speed_tangential")
        self._timeder_speed_radial = ca.SX.sym("timeder_speed_radial")
        self._define_symbolic_variables_kin()

    def _define_symbolic_variables_kin(self):
        """
        Define symbolic variables used in the model.
        """
        base_symbolic_variables = {
            "speed_tangential": "speed_tangential",
            "speed_radial": "speed_radial",
            "timeder_angle_course": "timeder_angle_course",
            "distance_radial": "distance_radial",
            "angle_course": "angle_course",
            "angle_elevation": "angle_elevation",
            "angle_azimuth": "angle_azimuth",
        }
        for var_name in base_symbolic_variables.keys():
            setattr(self, var_name, ca.SX.sym(var_name))

    @property
    def timeder_angle_elevation(self):
        return self.speed_tangential * ca.cos(self.angle_course) / self.distance_radial

    @property
    def timeder_angle_azimuth(self):
        return (
            self.speed_tangential
            * ca.sin(self.angle_course)
            / (self.distance_radial * ca.cos(self.angle_elevation))
        )

    @property
    def velocity_kite(self):
        return ca.vertcat(self.speed_tangential, 0, self.speed_radial)

    @property
    def timeder_speed_tangential(self):
        return self._timeder_speed_tangential

    @timeder_speed_tangential.setter
    def timeder_speed_tangential(self, value):
        self._timeder_speed_tangential = value

    @property
    def timeder_speed_radial(self):
        return self._timeder_speed_radial

    @timeder_speed_radial.setter
    def timeder_speed_radial(self, value):
        self._timeder_speed_radial = value

    @property
    def velocity_rotation_course_frame(self):
        return ca.vertcat(
            0,
            self.speed_tangential / self.distance_radial,
            self.speed_tangential
            / self.distance_radial
            * ca.tan(self.angle_elevation)
            * ca.sin(self.angle_course)
            - self.timeder_angle_course,
        )


    

class ParametrizedKinematics:

    def __init__(self, pattern, quasi_steady=False):
        self.pattern = pattern

        self.s = ca.SX.sym("s")
        self.t = ca.SX.sym("t")
        self.s_dot = ca.SX.sym("s_dot")
        if quasi_steady:
            self.s_ddot = 0
        else:
            self.s_ddot = ca.SX.sym("s_ddot")

    @property
    def beta(self):
        return self.pattern.elevation(self.t, self.s)
    
    @property
    def phi(self):
        return self.pattern.azimuth(self.t, self.s)
    
    @property
    def r(self):
        return self.pattern.r(self.t)

    @property
    def dtheta_ds(self):
        return (
            ca.gradient(self.phi, self.s)
            + ca.gradient(self.phi, self.t) / self.s_dot
        )

    @property
    def dbeta_ds(self):
        return (
            ca.gradient(self.beta, self.s)
            + ca.gradient(self.beta, self.t) / self.s_dot
        )

    @property
    def dr_ds(self):
        return self.pattern.vr / self.s_dot

    @property
    def dr_ds2(self):
        return ca.gradient(self.dr_ds, self.s)

    @property
    def dR_ds(self):
        return ca.vertcat(
            self.r
            * self.dtheta_ds
            * ca.cos(self.beta),
            self.r * self.dbeta_ds,
            self.dr_ds,
        )

    @property
    def vr(self):
        return self.pattern.vr

    @property
    def vk(self):
        return ca.norm_2(self.dR_ds) * self.s_dot

    @property
    def vtau(self):
        return ca.sqrt(self.vk**2 - self.vr**2)

    @property
    def dot_vr(self):
        return self.dr_ds2 * self.s_dot**2 + self.s_ddot * self.dr_ds

    @property
    def dot_vtau(self):
        return self.sqrt_A * (
            self.s_dot**2 * self.dr_ds + self.s_ddot * self.r
        ) + self.s_dot * self.r * self.dot_A / (2 * self.sqrt_A)

    @property
    def dbeta_ds2(self):
        return ca.gradient(self.dbeta_ds, self.s)

    @property
    def dtheta_ds2(self):
        return ca.gradient(self.dtheta_ds, self.s)

    @property
    def chi(self):
        return ca.atan2(
            self.dtheta_ds
            * self.s_dot
            * ca.cos(self.beta),
            self.dbeta_ds * self.s_dot,
        )

    @property
    def dot_chi(self):
        return ca.gradient(self.chi, self.s) * self.s_dot

    @property
    def sqrt_A(self):
        return self.vtau / (self.s_dot * self.r)

    @property
    def dot_A(self):
        return (
            2
            * self.s_dot
            * (
                self.dbeta_ds * self.dbeta_ds2
                + self.dtheta_ds * self.dtheta_ds2 * ca.cos(self.beta) ** 2
                - self.dtheta_ds**2 * self.dbeta_ds * ca.sin(self.beta) * ca.cos(self.beta)
            )
        )

    def extract_function(self, attr_name):
        """
        Returns a CasADi function for a given symbolic attribute name.
        
        :param attr_name: Name of the attribute (string)
        :return: CasADi function
        """
        if not hasattr(self, attr_name):
            raise AttributeError(f"'ParametrizedKinematics' has no attribute '{attr_name}'")

        expression = getattr(self, attr_name)  # Get the symbolic expression
        if not isinstance(expression, ca.MX) and not isinstance(expression, ca.SX):
            raise TypeError(f"'{attr_name}' is not a CasADi symbolic expression")

        # Extract all symbolic variables used in the expression
        variables = list(ca.symvar(expression))
        variables.sort(key=lambda x: x.name())  # Ensure consistent ordering

        return ca.Function(attr_name, variables, [expression], [var.name() for var in variables], [attr_name])
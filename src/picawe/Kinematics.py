import casadi as ca
class KiteKinematics:

    def __init__(self):
        
        self._define_symbolic_variables_kin()
    
    def _define_symbolic_variables_kin(self):
        """
        Define symbolic variables used in the model.
        """
        base_symbolic_variables = {
            'timeder_speed_tangential': 'timeder_speed_tangential',
            'timeder_angle_course': 'timeder_angle_course',
            'timeder_speed_radial': 'timeder_speed_radial',
            'speed_tangential': 'speed_tangential',
            'speed_radial': 'speed_radial',
            'distance_radial': 'distance_radial',
            'angle_course': 'angle_course',
            'angle_elevation': 'angle_elevation',
            'angle_azimuth': 'angle_azimuth',
            }
        for var_name in base_symbolic_variables.keys():
            setattr(self, var_name, ca.SX.sym(var_name))

    @property
    def timeder_angle_elevation(self):
        return self.speed_tangential * ca.cos(self.angle_course) / self.distance_radial
    
    @property
    def timeder_angle_azimuth(self):
        return self.speed_tangential * ca.sin(self.angle_course) / (self.distance_radial* ca.cos(self.angle_elevation))
    
    @property
    def velocity_kite(self):
        return ca.vertcat(self.speed_tangential, 0, self.speed_radial)

    @property
    def acceleration_elevation(self):
        r = self.distance_radial
        r_dot = self.speed_radial
        speed_tangential = self.speed_tangential
        speed_tangential_dot = self.timeder_speed_tangential
        angle_course = self.angle_course
        angle_course_dot = self.timeder_angle_course

        angle_elevation_dot = (
            (
               r * speed_tangential_dot * ca.cos(angle_course) - r * speed_tangential * angle_course_dot * ca.sin(angle_course) - r_dot * speed_tangential * ca.cos(angle_course)
            ) / r**2
        )

        return angle_elevation_dot

    @property
    def acceleration_azimuth(self):
        r = self.distance_radial
        r_dot = self.speed_radial
        v_tau = self.speed_tangential
        v_tau_dot = self.force_residual[0]
        beta = self.angle_elevation
        beta_dot = self.timeder_angle_elevation
        chi = self.angle_course
        chi_dot = self.timeder_angle_course

        phi_ddot = (
            (
                r*ca.cos(beta) * (v_tau_dot * ca.sin(chi) + v_tau * chi_dot * ca.cos(chi)) -
                v_tau * ca.sin(chi) * (r_dot * ca.cos(beta) - r * beta_dot * ca.sin(beta))
            ) / (
                r**2 * ca.cos(beta)**2
            )
        )

        return phi_ddot
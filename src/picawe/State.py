import casadi as ca
import numpy as np
from picawe.Tether import Tether
from picawe.Kinematics import KiteKinematics
from picawe.Wind import Wind
from picawe.Kite import Kite


class State(KiteKinematics, Tether, Wind, Kite):

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
        E=132e9,
        diameter=0.008,
        density=970,
        dof = 6
    ):
        """
        Initialize the kite system with its parameters.
        """
        # Define symbolic variables for the function inputs
        KiteKinematics.__init__(self)
        Wind.__init__(self)
        Kite.__init__(
            self,
            mass_wing,
            area_wing,
            aero_input,
            mass_kcu,
            g,
            rho,
            center_aerodynamic_wing,
            center_gravity_wing,
        )
        Tether.__init__(self, E, diameter, density)
        if dof == 3:
            self.angle_pitch = 0
            self.angle_roll = 0
            self.angle_yaw = 0
        
        self.dof = dof

    def ode(self):
        dot_r = self.speed_radial
        dot_beta = self.timeder_angle_elevation
        dot_theta = self.timeder_angle_azimuth
        dot_chi = self.timeder_angle_course
        dot_vr = self.timeder_speed_radial
        dot_vt = self.timeder_speed_tangential
        dot_lt = self.timeder_length_tether

        ode = ca.vertcat(dot_r, dot_beta, dot_theta, dot_chi, dot_vr, dot_vt, dot_lt)
        return ode


    def algebraic(self):
        return self.force_residual()

    def establish_residual(self):
        if self.dof == 6:
            self.residual = self.rb_residual
        elif self.dof == 3:
            self.residual = self.force_residual()
        
    def solve_quasi_steady_state(
        self, current_state, unknown_vars, x0, solver_options={}
    ):
        """
        Solve the quasi-steady state equations for the kite system.

        :param known_state: Dictionary of known state variables and their values.
        :param unknown_vars: List of unknown state variables to solve for.
        :return: Dictionary of unknown state variables and their values.
        """
        if self.dof == 6:
            # Solve the system of equations
            lbx = [
                current_state["distance_radial"] - 5,
                -10,
                -1,
                -np.pi / 4,
                -np.pi / 4,
                -np.pi / 4,
            ]  # Lower bounds for T, u_s, speed_tangential, phi_k, theta_k
            ubx = [
                current_state["distance_radial"],
                10,
                500,
                np.pi / 4,
                np.pi / 4,
                np.pi / 4,
            ]  # Upper bounds for T, u_s, speed_tangential, phi_k, theta_k
        elif self.dof == 3:
            # Solve the system of equations
            lbx = [
                current_state["distance_radial"] - 5,
                -10,
                -10,
            ]  # Lower bounds for T, u_s, speed_tangential, phi_k, theta_k
            ubx = [
                current_state["distance_radial"],
                10,
                500,
            ]  # Upper bounds for T, u_s, speed_tangential, phi_k, theta_k

        sym_list = [getattr(self, name) for name in unknown_vars]
        residual = self.residual
        for state_name, state_value in current_state.items():
            if state_name not in unknown_vars:
                residual = ca.substitute(
                    residual, getattr(self, state_name), state_value
                )
        # NLP problem definition
        nlp = {
            "x": ca.vertcat(*sym_list),
            "f": 0,
            "g": residual,
        }  # 'f' is set to 0 for root-finding

        # Define the NLP solver
        solver = ca.nlpsol("solver", "ipopt", nlp, solver_options)

        # Bounds for the constraints
        lbg = [0] * residual.size1()  # Lower bounds (0 for residuals)
        ubg = [0] * residual.size1()  # Upper bounds (0 for residuals)

        # try:
        # Solve the system
        sol = solver(x0=x0, lbg=lbg, ubg=ubg, lbx=lbx, ubx=ubx)  # Initial guess
        converged = True
        if ca.norm_1(sol["g"]) > 1:
            print("Quasi-steady state not found")
            converged = False

        return sol["x"], converged

    def integrate(self, x0, time, time_step, quasi_steady = False):

        x = ca.vertcat(
            self.distance_radial,
            self.angle_elevation,
            self.angle_azimuth,
            self.angle_course,
            self.speed_radial,
            self.speed_tangential,
            self.length_tether,
        )
        
        if quasi_steady:
            ode = {'x': x, 'ode': self.ode()}
            intg = ca.integrator('intg', 'cvodes', ode, time, time + time_step)
            res = intg(x0=x0)
            return res['xf']
        else:
            z = ca.vertcat(
                self.timeder_speed_radial,
                self.timeder_angle_course,
                self.timeder_speed_tangential,
            )
            
            dae = {"x": x, "z": z, "ode": self.ode(), "alg": self.algebraic()}

            intg = ca.integrator(
                "intg", "idas", dae, time, time + time_step
            )

            res = intg(x0=x0)

            new_state = res["xf"]
            zf = res["zf"]
            # new_state.update(current_state)
            return new_state, zf


    def extract_function(self, attribute_name):
        """Extract a CasADi function dynamically based on the attribute name."""
        # Get the expression from the attribute name
        if not hasattr(self, attribute_name):
            raise AttributeError(f"'State' object has no attribute '{attribute_name}'")
        
        expression = getattr(self, attribute_name)
        
        # Find all symbolic variables used in the expression
        variables = ca.symvar(expression)
        
        # Sort the variables by name for consistent ordering
        variables.sort(key=lambda x: x.name())

        names = [var.name() for var in variables]
        
        # Create and return the CasADi function
        return ca.Function(attribute_name, variables, [expression], names, [attribute_name])
import casadi as ca
import numpy as np
from picawe.system.tether import RigidLinkTether, FlexibleLinkTether
from picawe.system.kite import Kite
from picawe.kinematics.Kinematics import KiteKinematics
from picawe.environment.Wind import Wind
from picawe.utils.defaults import DEFAULT_BOUNDS
import inspect


class SystemModel(KiteKinematics):

    def __init__(
        self,
        dof=6,
        quasi_steady=False,
        wind_model="logarithmic",
        tether=None,
        kite = None,
    ):
        """
        Initialize the kite system with its parameters.
        """
        # Define symbolic variables for the function inputs
        KiteKinematics.__init__(self)
        self.define_wind_model(wind_model)
        self.define_kite_model(kite)
        self.define_tether_model(tether)
        
        self.steering_control = self.steering_control

        if self.steering_control not in ["asymmetric", "roll"]:
            raise ValueError("Invalid steering_control. Choose 'asymmetric' or 'roll'.")
        if dof == 3:

            self.timeder_angle_pitch = 0
            self.timeder_angle_roll = 0
            self.timeder_angle_yaw = 0
            self.acceleration_angle_pitch = 0
            self.acceleration_angle_roll = 0
            self.acceleration_angle_yaw = 0



        if quasi_steady:
            self.timeder_angle_roll = 0
            self.timeder_angle_pitch = 0
            self.timeder_angle_yaw = 0
            self.acceleration_angle_roll = 0
            self.acceleration_angle_pitch = 0
            self.acceleration_angle_yaw = 0
            self.timeder_length_tether = self.speed_radial
            self.timeder_speed_tangential = 0
            self.timeder_speed_radial = 0

        self.dof = dof
        self.quasi_steady = quasi_steady


    def define_kite_model(self, kite):
        if kite is None:
            kite = Kite(mass_wing= 20,
                        area_wing= 20,
                        aero_input=   {
                        "model": "inviscid",
                        "params": {
                            "CD0": 0.05,
                            "aspect_ratio": 10,
                            "oswald_efficiency": 1,
                            "angle_pitch_depower_0": 0,
                        }}) 
            print("Kite model not defined. Using default kite model.")
                
        # Inject all tether attributes into SystemModel so they can be accessed directly
        for attr_name, attr_value in vars(kite).items():
            setattr(self, attr_name, attr_value)
        # Copy properties from the component's class and its base classes
        for cls in inspect.getmro(kite.__class__):
            for name, obj in cls.__dict__.items():
                if isinstance(obj, property) and not hasattr(self.__class__, name):
                    setattr(self.__class__, name, obj)

    def define_tether_model(self, tether):
        if tether is None:
            tether = FlexibleLinkTether()
            print("Tether model not defined. Using default tether model.")
        self.tether = tether
        # Inject all tether attributes into SystemModel so they can be accessed directly
        for attr_name, attr_value in vars(tether).items():
            setattr(self, attr_name, attr_value)
        # Copy properties from the component's class and its base classes
        for cls in inspect.getmro(tether.__class__):
            for name, obj in cls.__dict__.items():
                if isinstance(obj, property) and not hasattr(self.__class__, name):
                    setattr(self.__class__, name, obj)

    def define_wind_model(self, wind_model):
        if wind_model == "logarithmic"or wind_model == "uniform":
            self.wind = Wind(wind_model)
        else:
            raise ValueError("Invalid wind model. Choose 'logarithmic' or 'uniform'.")

    def ode_function(self):
        dot_r = self.speed_radial
        dot_beta = self.timeder_angle_elevation
        dot_theta = self.timeder_angle_azimuth
        dot_vt = self.acceleration_total[0]
        dot_chi = self.acceleration_total[1]
        dot_vr = self.acceleration_total[2]
        dot_lt = self.timeder_length_tether
        ode = ca.vertcat(dot_r, dot_beta, dot_theta, dot_vt, dot_chi, dot_vr, dot_lt)
        if self.dof == 6 and not self.quasi_steady:
            ode_add = ca.vertcat(
                self.timeder_angle_roll,
                self.timeder_angle_pitch,
                self.timeder_angle_yaw,
                self.acceleration_angle_roll,
                self.acceleration_angle_pitch,
                self.acceleration_angle_yaw,
            )
            ode = ca.vertcat(ode, ode_add)
        return ode

    def algebraic_function(self):
        if self.dof == 6:
            return self.rb_residual
        else:
            return self.force_residual

    def establish_residual(self):
        if self.dof == 6:
            self.residual = self.rb_residual
        elif self.dof == 3:
            self.residual = self.force_residual

    def setup_qs_solver(
        self,
        unknown_vars=["speed_tangential", "timeder_angle_course", "length_tether"],
        solver_options=None,
    ):
        """
        Solve the quasi-steady state equations for the kite system.

        :param known_state: Dictionary of known state variables and their values.
        :param unknown_vars: List of unknown state variables to solve for.
        :return: Dictionary of unknown state variables and their values.
        """
        self.establish_residual()
        x = [getattr(self, name) for name in unknown_vars]

        inputs = []
        for var in ca.symvar(self.residual):
            if var.name() not in unknown_vars:
                inputs.append(var)
        inputs_name = [name.name() for name in inputs]
        
        # NLP problem definition
        nlp = {
            "x": ca.vertcat(*x),
            "f": 0,
            "g": self.residual,
            "p": ca.vertcat(*inputs),
        }  # 'f' is set to 0 for root-finding

        # Define the solver options
        if solver_options is None:
            solver_options = self.solver_options()
        # Define the NLP solver
        solver = ca.nlpsol("solver", "ipopt", nlp, solver_options)

        return solver, inputs_name

    def get_boundaries(self, unkown_vars):

        lbx = []
        ubx = []
        for var in unkown_vars:
            lbx.append(DEFAULT_BOUNDS[var][0])
            ubx.append(DEFAULT_BOUNDS[var][1])

        # Bounds for the constraints
        lbg = [0] * self.residual.size1()  # Lower bounds (0 for residuals)
        ubg = [0] * self.residual.size1()  # Upper bounds (0 for residuals)

        return lbx, ubx, lbg, ubg

    @property
    def mechanical_power(self):
        """
        Compute the mechanical power of the kite system.
        """
        return self.tension_tether_ground * self.speed_radial

    @property
    def state_vector(self):
        """
        Get the state vector of the kite system.
        """
        return ca.vertcat(
            self.distance_radial,
            self.angle_elevation,
            self.angle_azimuth,
            self.speed_tangential,
            self.angle_course,
            self.speed_radial,
            self.length_tether,

        )

    @property
    def input_vector(self):
        """
        Get the input vector of the kite system.
        """
        return ca.vertcat(
            self.input_steering,
            self.input_depower,
            self.timeder_length_tether,
        )
    def integrator(self, time_step):


        if self.dof == 6 and not self.quasi_steady:
            x_add = ca.vertcat(
                self.angle_roll,
                self.angle_pitch,
                self.angle_yaw,
                self.timeder_angle_roll,
                self.timeder_angle_pitch,
                self.timeder_angle_yaw,
            )
            x = ca.vertcat(x, x_add)
            z = ca.vertcat(
                z,
                self.acceleration_angle_roll,
                self.acceleration_angle_pitch,
                self.acceleration_angle_yaw,
            )

        if self.quasi_steady:
            x = ca.vertcat(self.state_vector[0:3], self.state_vector[4])
            ode = ca.vertcat(self.ode[0:3], self.ode[4])
            z = ca.vertcat(self.speed_tangential, self.timeder_angle_course, self.length_tether)

            p = ca.vertcat(self.input_vector[0:2], self.speed_radial)
            alg = self.algebraic
            dae = {"x": x, "p": p, "z": z, "p": p, "ode": ode, "alg": alg}
            # Create the integrator
            intg = ca.integrator("intg", "idas", dae, 0, time_step)
            return intg
            
        else:
            p = self.input_vector
            ode = {"x": self.state_vector, "p": p, "ode": self.ode}

            return ca.integrator("intg", "cvodes", ode, 0, time_step)

    def establish_ode(self):
        """
        Establish the ordinary differential equations for the kite system.
        """
        self.ode = self.ode_function()

    def establish_algebraic(self):
        """
        Establish the algebraic equations for the kite system.
        """
        self.algebraic = self.algebraic_function()

    def extract_function(self, attribute_name):
        """Extract a CasADi function dynamically based on the attribute name."""

        # Ensure the attribute exists
        if not hasattr(self, attribute_name):
            raise AttributeError(f"'State' object has no attribute '{attribute_name}'")

        expression = getattr(self, attribute_name)

        # If the expression is a DM (numerical constant), return a constant function
        if isinstance(expression, ca.DM) or isinstance(expression, (int, float)):
            return ca.Function(attribute_name, [], [expression], [], [attribute_name])

        # If the expression is neither SX nor MX, it is not symbolic and should be handled
        if not isinstance(expression, (ca.SX, ca.MX)):
            raise TypeError(
                f"Expected symbolic expression (SX or MX), but got {type(expression)} for '{attribute_name}'"
            )

        # Extract symbolic variables from the expression
        variables = ca.symvar(expression)

        # Sort variables by name for consistent ordering
        variables.sort(key=lambda x: x.name())

        names = [var.name() for var in variables]

        # Create and return the CasADi function
        return ca.Function(
            attribute_name,
            variables,
            [expression],
            names,
            [attribute_name],
            {"allow_duplicate_io_names": True},
        )

    def solver_options(self):
        """
        Define the solver options for the NLP problem.

        :param print_level: Verbosity level of the solver.
        :return: Dictionary of solver options.
        """
        return {
            "ipopt": {
                "print_level": 0,  # Suppresses IPOPT output
                # 'max_iter': 200,  # Maximum number of iterations
                "sb": "yes",  # Suppresses more detailed solver information
            },
            "print_time": False,  # Disables CasADi's internal timing output
        }


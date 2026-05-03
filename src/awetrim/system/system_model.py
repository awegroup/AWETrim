import casadi as ca
import copy
import numpy as np
from awetrim.system.tether import RigidLinkTether, FlexibleLinkTether
from awetrim.system.kite import Kite
from awetrim.system.expressions import build_expression_registry
from awetrim.system.state import State
from awetrim.kinematics.Kinematics import KiteKinematics
from awetrim.environment.Wind import Wind
from awetrim.utils.defaults import DEFAULT_BOUNDS
import logging

logger = logging.getLogger(__name__)


class SystemModel(KiteKinematics):

    def __init__(
        self,
        dof=3,
        quasi_steady=False,
        neglect_radial_acceleration=True,
        wind_model=None,
        tether=None,
        kite=None,
        acceleration_winch=2,
        depower_rate=0.2,
    ):
        """
        Initialize the kite system with its parameters.
        """
        # Define symbolic variables for the function inputs
        KiteKinematics.__init__(self)
        self.define_wind_model(wind_model)
        self.define_kite_model(kite)
        self.define_tether_model(tether)

        self._override_gravity = False
        self._override_centripetal = False
        self._override_coriolis = False

        self.acceleration_winch = acceleration_winch
        self.depower_rate = depower_rate
        # self.steering_control = self.steering_control

        if self.kite.steering_control not in ["asymmetric", "roll"]:
            raise ValueError("Invalid steering_control. Choose 'asymmetric' or 'roll'.")

        if quasi_steady:
            self.timeder_speed_tangential = 0
            #     if neglect_radial_acceleration:
            self.timeder_speed_radial = 0
        #     self.timeder_angle_roll = 0
        #     self.timeder_angle_pitch = 0
        #     self.timeder_angle_yaw = 0
        #     self.acceleration_angle_roll = 0
        #     self.acceleration_angle_pitch = 0
        #     self.acceleration_angle_yaw = 0
        self.timeder_length_tether = self.speed_radial

        # else:
        #     self.timeder_length_tether = ca.MX.sym("timeder_length_tether")

        # self.quasi_steady = quasi_steady
        self._qs_solver = None
        self._qs_vars = None
        self._qs_inputs = None
        self.ode = None
        self.algebraic = None
        if self.is_tether_rigid:
            self.default_unknown_vars = [
                "speed_tangential",
                "timeder_angle_course",
                "tension_tether_ground",
            ]
        else:
            self.default_unknown_vars = [
                "speed_tangential",
                "input_steering",
                "length_tether",
            ]
        self.derived_function_names = [
            "angle_of_attack",
            "tension_tether_ground",
            "lift_coefficient",
            "drag_coefficient",
            "angle_course",
            "timeder_angle_course",
            "angle_elevation",
            "angle_azimuth",
            "speed_apparent_wind",
        ]
        self._derived_functions = None
        self._expressions = self._build_expression_registry()

    def define_kite_model(self, kite):
        if kite is None:
            kite = Kite(
                mass_wing=20,
                area_wing=20,
                aero_input={
                    "model": "inviscid",
                    "params": {
                        "CD0": 0.05,
                        "aspect_ratio": 10,
                        "oswald_efficiency": 1,
                        "angle_pitch_depower_0": 0,
                    },
                },
            )
            print("Kite model not defined. Using default kite model.")

        self.kite = kite

    def define_tether_model(self, tether):
        if tether is None:
            tether = FlexibleLinkTether()
            print("Tether model not defined. Using default tether model.")
        self.tether = tether

    def define_wind_model(self, wind_model):
        if wind_model is None:
            self.wind = Wind("uniform", direction_wind=0)
        else:
            self.wind = wind_model

    def _build_expression_registry(self):
        """Named symbolic outputs available for extraction and post-processing."""
        return build_expression_registry(self)

    def refresh_expression_registry(self):
        self._expressions = self._build_expression_registry()

    def expression_registry(self):
        return dict(self._expressions)

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for key, value in self.__dict__.items():
            if key != "_expressions":
                setattr(result, key, copy.deepcopy(value, memo))
        result.refresh_expression_registry()
        return result

    def available_expressions(self):
        return tuple(sorted(self._expressions))

    def has_expression(self, name):
        return name in self._expressions

    def expression(self, name):
        try:
            return self._expressions[name]()
        except KeyError as exc:
            raise AttributeError(f"'SystemModel' has no expression '{name}'") from exc

    @property
    def tension_tether_equation(self):
        # TODO: Write explicit equation for tether force
        lhs = (self.kite.mass_wing + self.kite.mass_kcu) * self.acceleration
        return (
            -lhs[2]
            + self.expression("force_aerodynamic")[2]
            + self.expression("force_gravity")[2]
            + self.expression("drag_tether_at_kite")[2]
            + self.expression("force_gravity_tether_at_kite")[2]
        )

    @property
    def input_steering(self):
        return self.kite.input_steering

    @input_steering.setter
    def input_steering(self, value):
        self.kite.input_steering = value

    @property
    def input_depower(self):
        return self.kite.input_depower

    @input_depower.setter
    def input_depower(self, value):
        self.kite.input_depower = value

    @property
    def g(self):
        return self.kite.g

    @g.setter
    def g(self, value):
        self.kite.g = value

    @property
    def rho(self):
        return self.kite.rho

    @rho.setter
    def rho(self, value):
        self.kite.rho = value

    @property
    def override_gravity(self):
        return self._override_gravity

    @override_gravity.setter
    def override_gravity(self, value):
        if not isinstance(value, bool):
            raise ValueError("override_gravity must be True or False.")
        self._override_gravity = value

    @property
    def override_centripetal(self):
        return self._override_centripetal

    @override_centripetal.setter
    def override_centripetal(self, value):
        if not isinstance(value, bool):
            raise ValueError("override_centripetal must be True or False.")
        self._override_centripetal = value

    @property
    def override_coriolis(self):
        return self._override_coriolis

    @override_coriolis.setter
    def override_coriolis(self, value):
        if not isinstance(value, bool):
            raise ValueError("override_coriolis must be True or False.")
        self._override_coriolis = value

    @property
    def is_tether_rigid(self):
        return self.tether.is_tether_rigid

    @property
    def length_tether(self):
        return self.tether.length_tether

    @length_tether.setter
    def length_tether(self, value):
        self.tether.length_tether = value

    @property
    def timeder_length_tether(self):
        return self.tether.timeder_length_tether

    @timeder_length_tether.setter
    def timeder_length_tether(self, value):
        self.tether.timeder_length_tether = value

    @property
    def tension_tether_ground(self):
        if hasattr(self.tether, "tension_tether_ground_for"):
            return self.tether.tension_tether_ground_for(self)
        return self.tether.tension_tether_ground

    @tension_tether_ground.setter
    def tension_tether_ground(self, value):
        self.tether.tension_tether_ground = value

    @property
    def force_tether_at_kite(self):
        return self.tether.force_tether_at_kite_for(self)

    @property
    def force_gravity_kcu(self):
        return self.kite.force_gravity_kcu_for(self)

    @property
    def force_external(self):
        # print("force_external:", self.force_aerodynamic, self.force_gravity)

        return (
            self.expression("force_aerodynamic")
            + self.expression("force_gravity")
            + self.expression("force_tether_at_kite")
        )

    @property
    def force_residual(self):
        """
        Compute the residual for the kite system dynamics.
        """
        # LHS and RHS
        lhs = (self.kite.mass_wing) * self.acceleration
        # Residual
        # print(self.force_external)
        # print(lhs)
        return -lhs + self.force_external

    def establish_ode_function(self):
        dot_r = self.speed_radial
        dot_beta = self.timeder_angle_elevation
        dot_theta = self.timeder_angle_azimuth
        dot_vt = self.acceleration_total[0]
        dot_chi = self.acceleration_total[1]
        dot_vr = self.acceleration_total[2]
        dot_lt = self.timeder_length_tether
        ode = ca.vertcat(dot_r, dot_beta, dot_theta, dot_vt, dot_chi, dot_vr, dot_lt)
        self._ode = ode

    def algebraic_function(self):
        return self.force_residual

    def establish_residual(self):
        self.residual = self.force_residual

    def setup_qs_solver(
        self,
        unknown_vars=None,
        solver_options=None,
    ):
        """
        Solve the quasi-steady state equations for the kite system.

        :param known_state: Dictionary of known state variables and their values.
        :param unknown_vars: List of unknown state variables to solve for.
        :return: Dictionary of unknown state variables and their values.
        """
        if unknown_vars is None:
            unknown_vars = self.default_unknown_vars
        self.establish_residual()

        x = []
        for name in unknown_vars:
            if hasattr(self, name):
                x.append(getattr(self, name))
            else:
                try:
                    x.append(getattr(self.wind, name))
                except AttributeError:
                    raise ValueError(
                        f"Unknown variable '{name}' is not a valid attribute."
                    )
        # x = [getattr(self, name) for name in unknown_vars]

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
        self._qs_solver, self._qs_inputs, self._qs_vars = (
            solver,
            inputs_name,
            unknown_vars,
        )
        # return solver, inputs_name, unknown_vars

    def solve_quasi_steady(self, state_obj, unknown_vars=None):
        if unknown_vars is None:
            unknown_vars = self.default_unknown_vars

        state_dict = state_obj.to_dict()

        if self._qs_solver is None or self._qs_vars != unknown_vars:
            self.setup_qs_solver(unknown_vars)

        p = [state_dict[name] for name in self._qs_inputs]
        print("Input names:", self._qs_inputs)
        lbx, ubx, lbg, ubg = self.get_boundaries(state_dict, unknown_vars)

        x0 = [safe_value(state_dict.get(var, 1.0)) for var in unknown_vars]
        # Solve the quasi-steady state equations

        print("Initial guess:", x0)
        print("Inputs (p):", p)
        sol = self._qs_solver(x0=x0, p=p, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)

        if np.linalg.norm(sol["g"]) > 1:
            logger.warning(
                "Quasi-steady solver did not converge. Residual norm: %.4f",
                np.linalg.norm(sol["g"]),
            )
            return None

        # Update with solved variables
        for i, var in enumerate(unknown_vars):
            state_dict[var] = float(sol["x"][i])

        if self._derived_functions is None:
            self._derived_functions = {
                name: self.extract_function(name)
                for name in self.derived_function_names
            }
        for name, func in self._derived_functions.items():
            args = [state_dict[n] for n in func.name_in()]
            state_dict[name] = float(func(*args))

        return State(**state_dict)

    def get_boundaries(
        self,
        current_state,
        unknown_vars=[
            "speed_tangential",
            "timeder_angle_course",
            "length_tether",
            "speed_radial",
        ],
    ):

        lbx = []
        ubx = []
        for var in unknown_vars:
            if var == "length_tether":
                lbx.append(current_state["distance_radial"] * 0.9)
                ubx.append(current_state["distance_radial"])
            else:
                lbx.append(DEFAULT_BOUNDS[var][0])
                ubx.append(DEFAULT_BOUNDS[var][1])

        # Bounds for the constraints
        lbg = [0] * len(unknown_vars)
        ubg = [0] * len(unknown_vars)

        return lbx, ubx, lbg, ubg

    # def get_derived_functions(self):

    #     return self._derived_functions

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
        if self.is_tether_rigid:
            return ca.vertcat(
                self.distance_radial,
                self.angle_elevation,
                self.angle_azimuth,
                self.speed_tangential,
                self.angle_course,
                self.speed_radial,
            )
        else:
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

    def integrator(self, time_step, quasi_steady=True, inputs=None):
        if quasi_steady:
            self.timeder_speed_radial = 0
            self.timeder_speed_tangential = 0
        if self.ode is None:
            self.establish_ode_function()
        if self.algebraic is None:
            self.establish_algebraic()

        if quasi_steady:

            p = ca.vertcat(
                self.timeder_angle_course, self.input_depower, self.speed_radial
            )

            x = ca.vertcat(
                self.distance_radial,
                self.angle_elevation,
                self.angle_azimuth,
                self.angle_course,
            )
            ode = ca.vertcat(
                self._ode[0],
                self._ode[1],
                self._ode[2],
                self._ode[4],
            )
            if self.is_tether_rigid:
                z = ca.vertcat(
                    self.speed_tangential,
                    self.input_steering,
                    self.tension_tether_ground,
                )
            else:
                z = ca.vertcat(
                    self.speed_tangential,
                    self.input_steering,
                    self.length_tether,
                )
            alg = self.algebraic

            dae = {"x": x, "p": p, "z": z, "p": p, "ode": ode, "alg": alg}
            # Create the integrator
            opts = {
                "abstol": 1e-6,
                "reltol": 1e-6,
                "max_num_steps": 20000,
                "max_step_size": 0.01,  # Or even 1e-3 if very stiff
            }

            # intg = ca.integrator("intg", "idas", dae, opts)
            intg = ca.integrator("intg", "idas", dae, 0, time_step, opts)
            return intg

        else:
            p = self.input_vector
            ode = {"x": self.state_vector, "p": p, "ode": self._ode}
            return ca.integrator("intg", "cvodes", ode, 0, time_step)

    def establish_algebraic(self):
        """
        Establish the algebraic equations for the kite system.
        """
        self.algebraic = self.algebraic_function()

    def extract_function(self, attribute_name):
        """Extract a CasADi function dynamically based on the attribute name."""

        if self.has_expression(attribute_name):
            expression = self.expression(attribute_name)
        elif hasattr(self, attribute_name):
            expression = getattr(self, attribute_name)
        else:
            raise AttributeError(f"'State' object has no attribute '{attribute_name}'")

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
                "max_iter": 200,  # Maximum number of iterations
                "sb": "yes",  # Suppresses more detailed solver information
            },
            "print_time": False,  # Disables CasADi's internal timing output
        }

    def reset_solver(self):
        """
        Reset the solver to its initial state.
        """
        self._qs_solver = None
        self._qs_vars = None
        self._qs_inputs = None
        self._derived_functions = None


def safe_value(val):
    return 0.0 if val is None else val

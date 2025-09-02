from matplotlib import pyplot as plt
from picawe.timeseries.timeseries import TimeSeries
from picawe.kinematics.parametrized_patterns import create_pattern_from_dict
from picawe import SystemModel
from picawe.kinematics.Kinematics import ParametrizedKinematics
import casadi as ca
import numpy as np
from picawe.utils.defaults import DEFAULT_PATTERN_CONFIG, DEFAULT_OPTI_LIMITS
import copy
from picawe.system.tether import RigidLinkTether
from picawe import State
from picawe.system.kite import Kite

import logging


logger = logging.getLogger(__name__)


class PhaseParameterized(TimeSeries):
    def __init__(
        self,
        kite_model: SystemModel,
        quasi_steady: bool = False,
        pattern_config: dict = DEFAULT_PATTERN_CONFIG,
    ):
        """
        Args:

        """

        super().__init__(
            kite_model=kite_model,
        )
        self.pattern_config = pattern_config
        self.quasi_steady = quasi_steady

        self.kite_model = kite_model
        self.target_drag_coefficient = None
        self.target_lift_coefficient = None
        self.s = ca.MX.sym("s")
        self.t = ca.MX.sym("t")
        self.s_dot = ca.MX.sym("s_dot")
        self.s_ddot = ca.MX.sym("s_ddot")
        # self.find_optimal_angle_pitch_tether()

    def run_simulation(self, start_state, allow_failure=True, return_states=False):

        print("Starting state:", start_state)
        self.substitute_parametrized_kinematics()
        self.states = []
        self.kite_model.reset_solver()

        if self.quasi_steady:
            unknown_vars = ["length_tether", "input_steering", "s_dot"]
        else:
            unknown_vars = ["length_tether", "input_steering", "s_ddot"]

        if self.kite_model.is_tether_rigid:
            unknown_vars[0] = "tension_tether_ground"
        # Initialize state
        if isinstance(start_state, dict):
            state_obj = State(**start_state)
        else:
            state_obj = start_state

        N = self.pattern_config["n_points"]
        time_step = self.pattern_config["end_time"] / self.pattern_config["n_points"]
        intg = self.integrator(time_step=time_step, inputs=None)
        new_state = self.kite_model.solve_quasi_steady(state_obj, unknown_vars)
        print("New state:", new_state)
        if self.quasi_steady:
            x0 = [new_state.s, new_state.distance_radial]
            z0 = ca.vertcat(
                new_state.tension_tether_ground,
                new_state.input_steering,
                new_state.s_dot,
                new_state.speed_radial,
            )
        else:
            x0 = [
                new_state.s,
                new_state.s_dot,
                new_state.distance_radial,
            ]
            z0 = ca.vertcat(
                new_state.tension_tether_ground,
                new_state.input_steering,
                new_state.s_ddot,
                new_state.speed_radial,
            )
        self.states.append(new_state.to_dict())
        t = self.pattern_config["start_time"]
        for i in range(N):
            # print(f"Time: {t}, State: {x0}, Inputs: {z0}")
            try:
                sol = intg(
                    x0=x0,
                    p=t,
                    z0=z0,
                )
            except Exception as e:
                print(f"Error occurred: {e}")
                if not allow_failure:
                    raise
                break
            x0 = sol["xf"]
            z0 = sol["zf"]
            if self.quasi_steady:
                new_state = State(
                    t=t,
                    s=x0[0],
                    input_steering=float(z0[1]),
                    tension_tether_ground=float(z0[0]),
                    s_dot=float(z0[2]),
                    distance_radial=float(x0[1]),
                    speed_radial=float(z0[3]),
                )
            else:
                new_state = State(
                    t=t,
                    s=x0[0],
                    s_dot=float(x0[1]),
                    input_steering=float(z0[1]),
                    tension_tether_ground=float(z0[0]),
                    s_ddot=float(z0[2]),
                    distance_radial=float(x0[2]),
                    speed_radial=float(z0[3]),
                )
            t += time_step
            self.states.append(new_state.to_dict())

    def run_simulation_opti(self, start_state):

        self.states = []
        self.kite_model.reset_solver()

        N = self.pattern_config["n_points"]
        time_step = self.pattern_config["end_time"] / N
        t0 = 0

        self.run_simulation(start_state, return_states=True)
        pattern_inputs, pattern = self.substitute_parametrized_kinematics(True)
        opti = ca.Opti()
        # pattern_inputs = self.substitute_parametrized_kinematics(True)

        self.optimization_vars = {}  # Store optimization variables

        # Create optimization variables for parameters to optimize
        for var in self.pattern_config["optimization_parameters"]:

            val = np.atleast_1d(
                self.pattern_config["parameters"][var]
            )  # guarantees array, even for scalar

            if len(val) > 1:
                self.optimization_vars[var] = opti.variable(len(val))
            else:
                self.optimization_vars[var] = (
                    opti.variable()
                )  # No bounds if not specified
            print(var, self.optimization_vars[var])

            # If the optimization variable is a vector, set as a list of variables
            opt_var = self.optimization_vars[var]
            setattr(pattern, var, opt_var)
            print(getattr(pattern, var))

        if "k_vr" in self.optimization_vars:
            k_vr = self.optimization_vars["k_vr"]
            pattern_inputs.append(k_vr)
        else:
            k_vr = 1
        opti_variables = {
            "s": opti.variable(N + 1),
            "s_dot": opti.variable(N + 1),
            "input_steering": opti.variable(N + 1),
            "tension_tether_ground": opti.variable(N + 1),
            "speed_radial": opti.variable(N + 1),
            "distance_radial": opti.variable(N + 1),
        }

        # Add optimization parameters
        for var in self.optimization_vars:
            opti_variables[var] = self.optimization_vars[var]

        opti.set_initial(opti_variables["s"], self.return_variable("s"))
        opti.set_initial(opti_variables["s_dot"], self.return_variable("s_dot"))
        opti.set_initial(
            opti_variables["input_steering"], self.return_variable("input_steering")
        )
        opti.set_initial(
            opti_variables["tension_tether_ground"],
            self.return_variable("tension_tether_ground"),
        )
        opti.set_initial(
            opti_variables["speed_radial"], self.return_variable("speed_radial")
        )
        opti.set_initial(
            opti_variables["distance_radial"], self.return_variable("distance_radial")
        )

        opti.subject_to(opti_variables["s"][0] == start_state.s)
        opti.subject_to(
            opti_variables["distance_radial"][0]
            == self.return_variable("distance_radial")[0]
        )
        # opti.subject_to(
        #     opti_variables["speed_radial"][0] == self.return_variable("speed_radial")[0]
        # )
        if not self.quasi_steady:
            opti.subject_to(
                opti_variables["s_dot"][0] == self.return_variable("s_dot")[0]
            )
        # opti.subject_to(
        #     opti_variables["input_steering"][0]
        #     == self.return_variable("input_steering")[0]
        # )
        # opti.subject_to(
        #     opti_variables["tension_tether_ground"][0]
        #     == self.return_variable("tension_tether_ground")[0]
        # )

        self.kite_model.establish_residual()
        flat = [ca.vertcat(*pattern_inputs)]

        if self.quasi_steady:
            residual = ca.Function(
                "residual",
                [
                    self.kite_model.s,
                    self.kite_model.s_dot,
                    self.kite_model.input_steering,
                    self.kite_model.tension_tether_ground,
                    self.kite_model.speed_radial,
                    self.kite_model.distance_radial,
                ]
                + flat,
                [self.kite_model.residual],
            )
            tether_tension_eq = ca.Function(
                "tether_tension_eq",
                [
                    self.kite_model.s,
                    self.kite_model.s_dot,
                    self.kite_model.input_steering,
                    self.kite_model.speed_radial,
                    self.kite_model.distance_radial,
                ]
                + flat,
                [self.kite_model.tension_tether_equation],
            )
        else:
            residual = ca.Function(
                "residual",
                [
                    self.kite_model.t,
                    self.kite_model.s,
                    self.kite_model.s_dot,
                    self.kite_model.s_ddot,
                    self.kite_model.input_steering,
                    self.kite_model.tension_tether_ground,
                ]
                + flat,
                [self.kite_model.residual],
            )

        # # Set the time values
        time_array = np.arange(N + 1) * time_step + t0
        # height = pattern.z(time_array, opti_variables["s"])
        # opti.subject_to(height >= 50)
        # radius_curvature = pattern.radius_curvature(time_array, opti_variables["s"])

        P_scale = opti.parameter()
        # set from your initial trajectory (rough but effective):
        T0 = self.return_variable("tension_tether_ground")
        vr0 = self.return_variable("speed_radial")
        s = self.return_variable("s")
        mask = (s > s[0]) & (s < s[0] + 2 * np.pi)
        P0 = (np.dot(T0[mask], vr0[mask]) * time_step) / max(
            time_array[mask]
        )  # crude scale for T*vr
        print(f"Initial P0: {P0}")
        opti.set_value(P_scale, max(abs(P0), 1.0))
        total_time = opti.parameter()
        opti.set_value(total_time, self.pattern_config["end_time"])
        energy = 0
        t_eff = 0
        for i in range(N + 1):
            opt_par_values = [opti_variables[var] for var in self.optimization_vars]
            flat = [ca.vertcat(*opt_par_values)]

            if self.quasi_steady:
                tether_inputs = [
                    opti_variables["s"][i],
                    opti_variables["s_dot"][i],
                    opti_variables["input_steering"][i],
                    opti_variables["speed_radial"][i],
                    opti_variables["distance_radial"][i],
                ] + flat
                tether_tension = tether_tension_eq(*tether_inputs)
                opti.subject_to(
                    (
                        opti_variables["speed_radial"][i] ** 2
                        * k_vr
                        * self.pattern_config["parameters"]["k_vr"]
                        - tether_tension
                    )
                    == 0
                )
                opti_variables["tension_tether_ground"][i] = tether_tension

                residual_inputs = [
                    opti_variables["s"][i],
                    opti_variables["s_dot"][i],
                    opti_variables["input_steering"][i],
                    tether_tension,
                    opti_variables["speed_radial"][i],
                    opti_variables["distance_radial"][i],
                ] + flat

            else:
                # For dynamic case, include s_ddot
                residual_inputs = [
                    time_array[i],
                    opti_variables["s"][i],
                    opti_variables["s_dot"][i],
                    opti_variables["s_ddot"][i],
                    opti_variables["input_steering"][i],
                    opti_variables["tension_tether_ground"][i],
                ] + flat
            res = residual(*residual_inputs)

            W = ca.diag(
                ca.vertcat(0.01, 0.01, 0.001)
            )  # tune so |W r| ~ 1 near feasible
            # opti.subject_to(W @ res == 0)
            opti.subject_to(res[0] / 100 == 0)
            opti.subject_to(res[1] / 100 == 0)

            if i < N:
                if self.quasi_steady:
                    opti.subject_to(
                        opti_variables["s"][i + 1]
                        == opti_variables["s"][i]
                        + opti_variables["s_dot"][i] * time_step
                    )
                    opti.subject_to(
                        opti_variables["distance_radial"][i + 1]
                        == opti_variables["distance_radial"][i]
                        + opti_variables["speed_radial"][i] * time_step
                    )

                else:
                    raise NotImplementedError("Dynamic case not implemented")

            # Only accumulate power when s is between 0 and 2π
        w = smooth_gate_interval(
            opti_variables["s"], start_state.s, start_state.s + 2 * np.pi
        )
        w = ca.reshape(w, (-1, 1))
        T = ca.reshape(opti_variables["tension_tether_ground"], (-1, 1))
        energy = ca.sum1(
            w * (T**1.5) / ca.sqrt(k_vr * self.pattern_config["parameters"]["k_vr"])
        )

        # t_eff = time_step * ca.sum1(w)
        power = energy / ca.sum1(w)
        opti.minimize(-power / P_scale)  # drop duplicate min with -power/P_scale
        # power = ca.dot(
        #     w, opti_variables["tension_tether_ground"] * opti_variables["speed_radial"]
        # )

        rho = 1  # demand at least 95% of warm-start power
        sigma = opti.variable()  # slack ≥ 0
        opti.subject_to(sigma >= 0)
        opti.subject_to(power >= rho * P_scale - sigma)

        # Take the mean

        # energy = time_step * ca.dot(
        #     opti_variables["tension_tether_ground"], opti_variables["speed_radial"]
        # )
        # power = power / (time_array[-1] - time_array[0])

        # # Check gradient with respect to optimization variables
        # for var in self.optimization_vars:
        #     grad = ca.gradient(power, opti_variables[var])
        # print(f"Gradient for {var}: {grad}")

        w_sigma = 1e3  # big penalty on violating the floor
        opti.minimize(opti.f + w_sigma * sigma)  # 'opti.f' is your current objective

        # Average power only over s∈[0,2π]
        # power = energy / (t_eff + 1e-12)
        opti.solver(
            "ipopt",
            {
                "ipopt": {
                    # "max_iter": 100,
                    "bound_relax_factor": 1e-8,
                    "tol": 1e-4,  # Main tolerance
                    # "acceptable_iter": 3,  # Accept if solution is good for 3 iter
                    "acceptable_tol": 1e-4,  # Acceptable early termination
                    "constr_viol_tol": 1e-4,  # Constraint violation tolerance
                    "dual_inf_tol": 1e-4,  # Dual infeasibility
                    # "honor_original_bounds": "yes",
                    "hessian_approximation": "limited-memory",
                    "mu_strategy": "adaptive",
                    # "linear_solver": "mumps",
                }
            },
        )

        # Set initial conditions for optimization parameters
        for var in self.optimization_vars:
            print(
                f"Setting initial value for {var}: {self.pattern_config['parameters'][var]}"
            )
            print(self.optimization_vars[var])
            if var == "k_vr":
                opti.set_initial(self.optimization_vars[var], 1)
            else:
                opti.set_initial(
                    self.optimization_vars[var], self.pattern_config["parameters"][var]
                )
        ### APPLY CONSTRAINTS DYNAMICALLY FROM DEFAULT_OPTI_LIMITS ###
        for var_name, opti_var in opti_variables.items():
            if isinstance(opti_var, ca.MX):
                if var_name in DEFAULT_OPTI_LIMITS:
                    if var_name in self.optimization_vars:
                        print(f"Applying constraints for {var_name}")
                        lb, ub = DEFAULT_OPTI_LIMITS[var_name]
                        opti.subject_to(lb <= opti_var)
                        opti.subject_to(opti_var <= ub)
                    else:
                        print(f"Applying constraints for {var_name}")
                        lb, ub = DEFAULT_OPTI_LIMITS[var_name]
                        opti.subject_to(lb <= opti_var[:])
                        opti.subject_to(opti_var[:] <= ub)

        try:
            solution = opti.solve()
            # Print optimized values for variables in the pattern
            print("\n Optimized Pattern Variables:")
            for var_name, var in self.optimization_vars.items():
                print(f"  {var_name}: {solution.value(var)}")
                optimized_config = self.pattern_config.copy()

                if var_name == "k_vr":
                    optimized_config["parameters"].update(
                        {
                            "k_vr": opti.debug.value(var)
                            * self.pattern_config["parameters"]["k_vr"]
                        }
                    )
                else:
                    optimized_config["parameters"].update(
                        {var_name: opti.debug.value(var)}
                    )
                self.pattern_config = optimized_config
                self.substitute_parametrized_kinematics()

            print(solution.value(power))
        except Exception as e:
            # Print debug optimization information
            print("Debug optimization information:")
            for var_name, var in self.optimization_vars.items():
                print(f"  {var_name}: {opti.debug.value(var)}")
            print("Optimization failed:", e)

        # plt.plot(solution.value(opti_variables["input_steering"]))
        # plt.show()
        print("Optimization status:", solution)
        s_vals = solution.value(opti_variables["s"])  # shape: (N+1,)
        s_dot_vals = solution.value(opti_variables["s_dot"])  # shape: (N+1,)
        tension_vals = solution.value(
            opti_variables["tension_tether_ground"]
        )  # shape: (N+1,)
        input_steering_vals = solution.value(
            opti_variables["input_steering"]
        )  # shape: (N+1,)
        self.states = []
        for i in range(N + 1):

            new_state = State(
                t=float(time_array[i]),
                s=float(s_vals[i]),
                input_steering=float(input_steering_vals[i]),
                tension_tether_ground=float(tension_vals[i]),
                s_dot=float(s_dot_vals[i]),
                distance_radial=float(
                    solution.value(opti_variables["distance_radial"][i])
                ),
                speed_radial=float(solution.value(opti_variables["speed_radial"][i])),
            )

            self.states.append(new_state.to_dict())

    # def _flatten_for_function_call(vals):
    #     flat = []
    #     for v in vals:
    #         if isinstance(v, ca.MX) and v.numel() > 1:
    #             # ensure column, then split into scalars
    #             flat += list(ca.vertsplit(ca.reshape(v, (-1, 1))))
    #         else:
    #             flat.append(v)
    #     return flat

    def substitute_parametrized_kinematics(self, optimize=False):

        pattern = create_pattern_from_dict(self.pattern_config, optimize=optimize)

        kinematics = ParametrizedKinematics(pattern, self)

        self.kite_model.s = kinematics.s
        self.kite_model.s_dot = kinematics.s_dot
        self.kite_model.s_ddot = kinematics.s_ddot

        self.kite_model.angle_course = kinematics.chi
        self.kite_model.angle_elevation = kinematics.beta
        # Optimal analytical solution for speed_radial should be part of the pattern class
        # self.kite_model.speed_radial = self.kite_model.speed_radial
        # print(self.kite_model.speed_radial)
        # self.kite_model.speed_radial = kinematics.vr
        self.kite_model.speed_tangential = kinematics.vtau
        self.kite_model.timeder_angle_course = kinematics.dot_chi
        if not self.quasi_steady:
            self.kite_model.timeder_speed_radial = kinematics.dot_vr
            self.kite_model.timeder_speed_tangential = kinematics.dot_vtau
        else:
            self.kite_model.timeder_speed_radial = 0
            self.kite_model.timeder_speed_tangential = 0

        self.kite_model.angle_azimuth = kinematics.phi
        self.kite_model.angle_elevation = kinematics.beta

        if optimize:
            return list(kinematics.pattern.optimization_vars.values()), pattern

    # @property
    # def target_lift_coefficient(self):
    #     return self._target_lift_coefficient

    # @target_lift_coefficient.setter
    # def target_lift_coefficient(self, value):
    #     self._target_lift_coefficient = value

    # @property
    # def target_drag_coefficient(self):
    #     return self._target_drag_coefficient

    # @target_drag_coefficient.setter
    # def target_drag_coefficient(self, value):
    #     self._target_drag_coefficient = value

    def integrator(self, time_step, inputs=None):
        self.kite_model.establish_residual()
        k_vr = self.pattern_config["parameters"]["k_vr"]
        if self.quasi_steady:
            x = ca.vertcat(self.kite_model.s, self.kite_model.distance_radial)
            if self.kite_model.is_tether_rigid:
                z = ca.vertcat(
                    self.kite_model.tension_tether_ground,
                    self.kite_model.input_steering,
                    self.kite_model.s_dot,
                    self.kite_model.speed_radial,
                )
            else:
                z = ca.vertcat(
                    self.kite_model.length_tether,
                    self.kite_model.input_steering,
                    self.kite_model.s_dot,
                    self.kite_model.speed_radial,
                )

            ode = ca.vertcat(
                self.kite_model.s_dot,
                self.kite_model.speed_radial,
            )
            alg = ca.vertcat(
                self.kite_model.residual,
                self.kite_model.tension_tether_ground
                - k_vr * self.kite_model.speed_radial**2,
            )
        else:
            x = ca.vertcat(
                self.kite_model.s,
                self.kite_model.s_dot,
                self.kite_model.distance_radial,
            )
            if self.kite_model.is_tether_rigid:
                z = ca.vertcat(
                    self.kite_model.tension_tether_ground,
                    self.kite_model.input_steering,
                    self.kite_model.s_ddot,
                    self.kite_model.speed_radial,
                )
            else:
                z = ca.vertcat(
                    self.kite_model.length_tether,
                    self.kite_model.input_steering,
                    self.kite_model.s_ddot,
                    self.kite_model.speed_radial,
                )

            ode = ca.vertcat(
                self.kite_model.s_dot,
                self.kite_model.s_ddot,
                self.kite_model.speed_radial,
            )
            alg = ca.vertcat(
                self.kite_model.residual,
                self.kite_model.tension_tether_ground
                - k_vr * self.kite_model.speed_radial**2,
            )

        dae = {"x": x, "z": z, "ode": ode, "alg": alg}
        # Create the integrator
        opts = {
            "abstol": 1e-6,
            "reltol": 1e-6,
            # "max_num_steps": 20000,
            # "max_step_size": 0.01,  # Or even 1e-3 if very stiff
        }

        # intg = ca.integrator("intg", "idas", dae, opts)
        intg = ca.integrator("intg", "idas", dae, 0, time_step, opts)
        return intg

    # def find_optimal_angle_pitch_tether(self):
    #     copy_kite = copy.deepcopy(self.kite_model)
    #     copy_kite.angle_elevation = 0
    #     copy_kite.angle_azimuth = 0
    #     copy_kite.angle_course = np.pi/2
    #     copy_kite.timeder_speed_tangential = 0
    #     copy_kite.distance_radial = 200
    #     copy_kite.wind.wind_model = 'uniform'
    #     copy_kite.wind.speed_wind_ref = 10
    #     copy_kite.speed_radial = 0
    #     copy_kite.timeder_angle_course = 0
    #     # copy_kite.angle_roll = 0
    #     copy_kite.timeder_speed_radial = 0
    #     copy_kite.delta_pitch_depower = 0
    #     copy_kite.input_depower = 0
    #     copy_kite.tether = RigidLinkTether()
    #     copy_kite.angle_pitch_tether = ca.MX.sym("angle_pitch_tether")
    #     copy_kite.speed_tangential = ca.MX.sym("speed_tangential")
    #     # copy_kite.aero_input["dependencies"]["u_s"] = {}
    #     # copy_kite.input_steering = 0
    #     # copy_kite.speed_radial = ca.MX.sym("speed_radial")
    #     print(copy_kite.lift_coefficient)
    #     cl_func = copy_kite.extract_function("lift_coefficient")
    #     cd_func = copy_kite.extract_function("drag_coefficient")
    #     aoa_func = copy_kite.extract_function("angle_of_attack")

    #     copy_kite.establish_residual()

    #     residual = ca.Function("residual", [copy_kite.speed_tangential, copy_kite.input_steering, copy_kite.length_tether, copy_kite.angle_pitch_tether], [copy_kite.residual], ["vtau", "steering", "length_tether", "angle_pitch"], ["residual"])
    #     print(residual)

    #     opti = ca.Opti()
    #     vtau = opti.variable()
    #     steering = opti.variable()
    #     lt = opti.variable()
    #     angle_pitch = opti.variable()
    #     opti.subject_to(vtau >= 0)
    #     opti.subject_to(vtau <= 300)
    #     opti.subject_to(steering >= -np.pi/2)
    #     opti.subject_to(steering <= np.pi/2)
    #     opti.subject_to(lt <= copy_kite.distance_radial)
    #     opti.subject_to(angle_pitch >= np.radians(-5))
    #     opti.subject_to(angle_pitch <= np.radians(15))
    #     opti.subject_to(residual(vtau = vtau, steering = steering, length_tether = lt, angle_pitch = angle_pitch)["residual"] == 0)

    #     opti.set_initial(vtau, 100)
    #     opti.set_initial(steering, 0)
    #     opti.set_initial(lt, copy_kite.distance_radial)
    #     opti.set_initial(angle_pitch, 0)

    #     opti.minimize(-lt)
    #     solver_opts = {"ipopt.print_level": 0, "print_time": 0}
    #     opti.solver("ipopt", solver_opts)
    #     try:
    #         sol = opti.solve()
    #         vtau = sol.value(vtau)
    #         angle_pitch = sol.value(angle_pitch)
    #         steering = sol.value(steering)
    #     except:
    #         print("Solver failed")
    #         print(opti.debug.value(vtau))
    #         print(opti.debug.value(steering))
    #         print(opti.debug.value(lt))
    #         print(opti.debug.value(angle_pitch))
    #     print(cl_func)
    #     if "u_s" in copy_kite.aero_input.get("dependencies", {}):
    #         self.target_lift_coefficient = cl_func(speed_tangential=vtau, angle_pitch_tether=angle_pitch, input_steering=steering)["lift_coefficient"]
    #         self.target_drag_coefficient = cd_func(speed_tangential=vtau, angle_pitch_tether=angle_pitch, input_steering=steering)["drag_coefficient"]
    #     else:
    #         self.target_lift_coefficient = cl_func(speed_tangential=vtau, angle_pitch_tether=angle_pitch)["lift_coefficient"]
    #         self.target_drag_coefficient = cd_func(speed_tangential=vtau, angle_pitch_tether=angle_pitch)["drag_coefficient"]
    #     self.target_angle_of_attack = aoa_func(speed_tangential=vtau, angle_pitch_tether=angle_pitch)["angle_of_attack"]
    #     self.optimal_angle_pitch_tether = float(angle_pitch)
    #     print(self.target_lift_coefficient, self.target_drag_coefficient, self.target_angle_of_attack*180/np.pi)

    # # def set_optimal_angle_pitch_tether(self):
    # #     print(f"Angle respect to the tether set to: {np.degrees(self.optimal_angle_pitch_tether)}")
    # #     self.kite_model.angle_pitch_tether = self.optimal_angle_pitch_tether

    # # def set_optimal_speed_radial(self):
    # #     # if self.target_drag_coefficient is None or self.target_lift_coefficient is None or self.target_angle_of_attack is None:

    # #     print(f"Optimal speed radial set according to the target  CL: {self.target_lift_coefficient} CD: {self.target_drag_coefficient} at aoa: {np.degrees(self.target_angle_of_attack)}")

    # #     CR_target = ca.sqrt(self.target_lift_coefficient**2 + self.target_drag_coefficient**2)

    # #     # self.kinematics.vr = ca.sqrt(
    # #     #     self.kite_model.tension_tether_ground / (
    # #     #         2 * 1.225 * self.kite_model.area_wing * CR_target *
    # #     #         (1 + (self.target_lift_coefficient / self.target_drag_coefficient)**2)
    # #     #     )
    # #     # )
    # #     # Calculate the optimal speed_radial
    # #     self.kite_model.speed_radial = ca.sqrt(
    # #         self.kite_model.tension_tether_ground / (
    # #             2 * 1.225 * self.kite_model.area_wing * CR_target *
    # #             (1 + (self.target_lift_coefficient / self.target_drag_coefficient)**2)
    # #         )
    # #     )
    # #     print(self.kite_model.speed_radial)
    # #     # print(f"Optimal speed radial set according to the target  CL: {self.target_lift_coefficient} CD: {self.target_drag_coefficient} at aoa: {np.degrees(self.target_angle_of_attack)}")


def smooth_gate_interval(s, s0, sf, eps=1e-1):
    """
    Smooth gate active only on [s0, sf], 0 outside.
    eps controls the ramp width at both ends.
    Works with MX/SX.
    """
    # guard: sf must be > s0
    # (you can add an assert or handle wrap if you need periodic behavior)

    def ramp(u):
        # 0 for u<=0, 1 for u>=eps, smooth in between
        u_clip = ca.fmin(ca.fmax(u / eps, 0), 1)
        return 0.5 * (1 - ca.cos(ca.pi * u_clip))

    w_lo = ramp(s - s0)  # rises near s0
    w_hi = ramp(sf - s)  # falls near sf
    return w_lo * w_hi


# # Plotting code
# import matplotlib.pyplot as plt

# s_vals = np.linspace(-2 * np.pi, 6 * np.pi, 500)
# # Evaluate using CasADi function
# s_sym = ca.MX.sym("s")
# gate_func = ca.Function(
#     "gate", [s_sym], [smooth_gate_interval(s_sym, np.pi / 2, 2 * np.pi)]
# )
# w_vals = np.array([float(gate_func(s)) for s in s_vals])

# plt.plot(s_vals, w_vals)
# plt.xlabel("s")
# plt.ylabel("smooth_gate_interval(s, 0, 2π)")
# plt.title("Smooth Gate from 0 to 2π")
# plt.grid(True)
# plt.show()

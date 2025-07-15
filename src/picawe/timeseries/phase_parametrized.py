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
        # self.find_optimal_angle_pitch_tether()

    def run_simulation(self, start_state=None, allow_failure=True):

        if start_state is None:
            start_state = {
                "t": 0,
                "s": 0,
                "s_dot": 2,
                "s_ddot": 0,
                "input_steering": 0,
                "tension_tether_ground": 1e5,
                "input_depower": self.pattern_config["control"]["input_depower"],
            }

        self.substitute_parametrized_kinematics()
        self.states = []
        self.kite_model.reset_solver()

        if self.pattern_config["start_path_angle"] is not None:
            s_array = np.linspace(
                self.pattern_config["start_path_angle"],
                self.pattern_config["end_path_angle"],
                self.pattern_config["n_points"],
            )
            time_array = None
        elif self.pattern_config["start_time"] is not None:
            time_array = np.linspace(
                self.pattern_config["start_time"],
                self.pattern_config["end_time"],
                self.pattern_config["n_points"],
            )
            s_array = None
        else:
            raise ValueError(
                "Either path_angle or time bounds must be provided in the pattern_config."
            )

        if self.quasi_steady:
            unknown_vars = ["length_tether", "input_steering", "s_dot"]
        else:
            unknown_vars = ["length_tether", "input_steering", "s_ddot"]

        if self.kite_model.dof == 6:
            unknown_vars += ["angle_roll", "angle_pitch", "angle_yaw"]

        if self.kite_model.is_tether_rigid:
            unknown_vars[0] = "tension_tether_ground"
        # Initialize state
        if isinstance(start_state, dict):
            state_obj = State(**start_state)
        else:
            state_obj = start_state

        speed_radial_func = self.kite_model.extract_function("speed_radial")
        input_length = speed_radial_func.n_in()
        distance_radial_func = self.kite_model.extract_function("distance_radial")

        if time_array is not None:
            s = state_obj.s
            s_dot = state_obj.s_dot
            for i in range(len(time_array)):
                if distance_radial_func.n_in() == 0:
                    state_obj.distance_radial = float(
                        distance_radial_func()["distance_radial"]
                    )
                else:
                    state_obj.distance_radial = float(
                        distance_radial_func(
                            *[
                                getattr(state_obj, name)
                                for name in distance_radial_func.name_in()
                            ]
                        )
                    )
                new_state = self.kite_model.solve_quasi_steady(state_obj, unknown_vars)
                if new_state:
                    if i < len(time_array) - 1:
                        time_step = time_array[i + 1] - time_array[i]
                    else:
                        time_step = 0

                    if self.quasi_steady:
                        s += new_state.s_dot * time_step
                        new_state.s = s
                        new_state.t = time_array[i]
                        new_state.s_ddot = 0
                    else:
                        s_dot += new_state.s_ddot * time_step
                        s += s_dot * time_step
                        new_state.s_dot = s_dot
                        new_state.s = s
                        new_state.t = time_array[i]

                    # if input_length > 0:
                    #     distance_radial += speed_radial_func(new_state.tension_tether_ground) * time_step
                    #     new_state.distance_radial = distance_radial

                    self.states.append(new_state.to_dict())
                    state_obj = new_state
                else:
                    logger.warning("Solver did not converge at time index %d", i)
                    if not allow_failure:
                        break
        kite_model = copy.deepcopy(self.kite_model)
        if s_array is not None:
            t = state_obj.t
            s_dot = state_obj.s_dot
            for i in range(len(s_array)):
                state_obj.s = s_array[i]
                if distance_radial_func.n_in() == 0:
                    state_obj.distance_radial = float(
                        distance_radial_func()["distance_radial"]
                    )
                else:
                    state_obj.distance_radial = float(
                        distance_radial_func(
                            *[
                                getattr(state_obj, name)
                                for name in distance_radial_func.name_in()
                            ]
                        )
                    )
                new_state = kite_model.solve_quasi_steady(state_obj, unknown_vars)
                if new_state:
                    if i < len(s_array) - 1:
                        delta_s = s_array[i + 1] - s_array[i]

                        if self.quasi_steady:
                            # Use direct relation from known s_dot
                            s_dot = new_state.s_dot
                            time_step = delta_s / s_dot if s_dot != 0 else 0.01

                        else:
                            # Constant acceleration: solve for time_step using quadratic formula
                            s_dot = new_state.s_dot
                            s_ddot = (
                                new_state.s_ddot
                                if new_state.s_ddot is not None
                                else 0.0
                            )

                            # Equation: Δs = v * Δt + 0.5 * a * Δt^2 → 0.5*a*dt^2 + v*dt - Δs = 0
                            a = 0.5 * s_ddot
                            b = s_dot
                            c = -delta_s

                            discriminant = b**2 - 4 * a * c
                            if discriminant >= 0 and abs(a) > 1e-8:
                                sqrt_disc = discriminant**0.5
                                dt1 = (-b + sqrt_disc) / (2 * a)
                                dt2 = (-b - sqrt_disc) / (2 * a)
                                # Pick positive, realistic time step
                                time_step = dt1 if dt1 > 0 else dt2
                                if time_step <= 0:
                                    time_step = 0.01  # fallback
                            else:
                                # If acceleration is zero or near-zero, fallback to linear
                                time_step = delta_s / s_dot if s_dot != 0 else 0.01

                    # Time and state update
                    if self.quasi_steady:
                        t += time_step
                        new_state.t = t
                        new_state.s = s_array[i]
                        new_state.s_ddot = 0

                    else:
                        s_ddot = new_state.s_ddot
                        s_dot += s_ddot * time_step
                        t += time_step
                        new_state.s_dot = s_dot
                        new_state.s = s_array[i]
                        new_state.t = t
                    # if input_length > 0:
                    #     distance_radial += speed_radial_func(new_state.tension_tether_ground) * time_step
                    #     new_state.distance_radial = distance_radial

                    self.states.append(new_state.to_dict())
                    state_obj = new_state
                else:
                    logger.warning("Solver did not converge at s index %d", i)
                    if not allow_failure:
                        break

    def optimize_pattern(self, start_state):
        # self.set_optimal_speed_radial()
        # self.set_optimal_angle_pitch_tether()

        self.kite_model_opt = copy.deepcopy(self.kite_model)

        if self.pattern_config["start_path_angle"] is not None:
            s_array = np.linspace(
                self.pattern_config["start_path_angle"],
                self.pattern_config["end_path_angle"],
                self.pattern_config["n_points"],
            )
            time_array = None
        elif self.pattern_config["start_time"] is not None:
            time_array = np.linspace(
                self.pattern_config["start_time"],
                self.pattern_config["end_time"],
                self.pattern_config["n_points"],
            )
            s_array = None
        else:
            raise ValueError(
                "Either path_angle or time bounds must be provided in the pattern_config."
            )
        self.run_simulation(start_state)
        # self.kite_model.reset_solver()
        pattern_inputs = self.substitute_parametrized_kinematics(True)

        # print(self.kite_model.force_aerodynamic)

        opti = ca.Opti()
        self.optimization_vars = {}  # Store optimization variables

        # Create optimization variables for parameters to optimize
        for var in self.pattern_config["optimization_parameters"]:
            self.optimization_vars[var] = opti.variable()  # No bounds if not specified

        if time_array is not None:
            # Create other required variables
            N = len(time_array)
            path_angle = opti.variable(len(time_array))
            time = time_array
        elif s_array is not None:
            N = len(s_array)
            path_angle = s_array
            time = opti.variable(len(s_array))
            sf = ca.SX.sym("sf")
            si = ca.SX.sym("si")
            s_dot_sym = ca.SX.sym("s_dot")
            ts = (sf - si) / s_dot_sym
            timestep_func = ca.Function("t_func", [si, sf, s_dot_sym], [ts])

        # Store optimization variables dynamically
        opti_variables = {
            "t": time,
            "s": path_angle,
            "s_dot": opti.variable(N),
            "input_steering": opti.variable(N),
            "tension_tether_ground": opti.variable(N),
        }
        if self.quasi_steady:
            opti_variables["s_ddot"] = ca.MX.zeros(N)  # Quasi-steady, no acceleration
        else:
            opti_variables["s_ddot"] = opti.variable(N)

        # Add optimization parameters
        for var in self.optimization_vars:
            opti_variables[var] = self.optimization_vars[var]

        # Define the residual function
        # Choose the correct variable depending on tether type
        if self.kite_model.is_tether_rigid:
            unknown_var = self.kite_model.tension_tether_ground
        else:
            unknown_var = self.kite_model.length_tether

        print(f"Unknown variable: {unknown_var}")
        self.kite_model.establish_residual()
        if self.quasi_steady:
            residual = ca.Function(
                "residual",
                [
                    self.kite_model.t,
                    self.kite_model.s,
                    self.kite_model.s_dot,
                    self.kite_model.input_steering,
                    unknown_var,
                ]
                + pattern_inputs,
                [self.kite_model.residual],
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
                    unknown_var,
                ]
                + pattern_inputs,
                [self.kite_model.residual],
            )

        # Define angle elevation function
        angle_elevation_fun = ca.Function(
            "angle_elevation",
            [self.kite_model.t, self.kite_model.s] + pattern_inputs,
            [self.kite_model.angle_elevation],
        )
        vr_func = self.kite_model.extract_function("speed_radial")
        print(vr_func)
        tension_tether_func = self.kite_model.extract_function("tension_tether_ground")
        print(tension_tether_func)
        distance_radial_func = self.kite_model.extract_function("distance_radial")
        print(distance_radial_func)

        energy = 0  # Initialize power
        angle_elevation = ca.MX.zeros(N)

        def call_distance_radial_func_casadi(func, t, vr=None):
            if func.n_in() == 2:
                return func(t, vr)
            return func(t)

        def call_tension_tether_func(func, length_tether, t, vr=None):
            if func.n_in() == 3:
                return func(length_tether, t, vr)
            return func(length_tether, t)

        def call_vr_func(func, vr):
            if func.n_in() == 1:
                return func(vr)
            return vr

        if "vr" not in opti_variables.keys():
            opti_variables["vr"] = self.pattern_config["parameters"]["vr"]

        for i in range(N):
            # Dynamically pass opti variables into residual function
            if self.quasi_steady:
                residual_inputs = [
                    opti_variables["t"][i],
                    opti_variables["s"][i],
                    opti_variables["s_dot"][i],
                    opti_variables["input_steering"][i],
                    opti_variables["tension_tether_ground"][i],
                ] + [opti_variables[var] for var in self.optimization_vars]
            else:
                # For dynamic case, include s_ddot
                residual_inputs = [
                    opti_variables["t"][i],
                    opti_variables["s"][i],
                    opti_variables["s_dot"][i],
                    opti_variables["s_ddot"][i],
                    opti_variables["input_steering"][i],
                    opti_variables["tension_tether_ground"][i],
                ] + [opti_variables[var] for var in self.optimization_vars]
            res = residual(*residual_inputs)
            # opti.subject_to(ca.norm_2(res) <= 1e-2)
            # print(res)
            opti.subject_to(res[0] == 0)
            opti.subject_to(res[1] == 0)
            opti.subject_to(res[2] == 0)

            if time_array is not None:

                time_step = (
                    (time_array[i + 1] - time_array[i])
                    if i < len(time_array) - 1
                    else 0.1
                )

                # Add dynamic constraint on path angle
                if i < len(time_array) - 1:
                    opti.subject_to(
                        path_angle[i + 1]
                        == path_angle[i] + time_step * opti_variables["s_dot"][i]
                    )
                    if not self.quasi_steady:
                        opti.subject_to(
                            opti_variables["s_ddot"][i]
                            == (
                                opti_variables["s_dot"][i + 1]
                                - opti_variables["s_dot"][i]
                            )
                            / time_step
                        )

            elif s_array is not None:

                if i < len(s_array) - 1:
                    delta_s = s_array[i + 1] - s_array[i]

                    # Avoid divide-by-zero
                    epsilon = 1e-6
                    time_step = delta_s / (opti_variables["s_dot"][i] + epsilon)

                    if not self.quasi_steady:
                        # Euler update for s_dot
                        opti.subject_to(
                            opti_variables["s_dot"][i + 1]
                            == opti_variables["s_dot"][i]
                            + opti_variables["s_ddot"][i] * time_step
                        )

                    # Euler update for time
                    opti.subject_to(time[i + 1] == time[i] + time_step)

                    # # Optional: explicitly relate s_ddot to s_dot differences (redundant, but valid if needed)
                    # opti.subject_to(s_ddot[i] == (s_dot[i + 1] - s_dot[i]) / time_step)
            energy += (
                opti_variables["tension_tether_ground"][i]
                * time_step
                * call_vr_func(vr_func, opti_variables["vr"])
            )

            # Compute angle elevation
            angle_elevation[i] = angle_elevation_fun(
                opti_variables["t"][i],
                opti_variables["s"][i],
                *[opti_variables[var] for var in self.optimization_vars],
            )

        # reelin_speed = 6
        # reelin_time = 0  # distance / reelin_speed
        # # Normalize power
        # power = (
        #     (power - 1000 * reelin_speed * reelin_time)
        #     / (time[-1] + reelin_time)
        #     / 1000
        # )
        # Normalize power
        power = energy / (time[-1] - time[0]) / 1000

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

        # Constraint for angle_elevation
        # opti.subject_to(angle_elevation >= DEFAULT_OPTI_LIMITS["angle_elevation"][0])
        # opti.subject_to(angle_elevation <= DEFAULT_OPTI_LIMITS["angle_elevation"][1])

        ### APPLY INITIAL VALUES FROM RESULTS ###
        opti.subject_to(time[0] == self.return_variable("t")[0])
        opti.subject_to(opti_variables["s_dot"][0] == self.return_variable("s_dot")[0])
        opti.set_initial(opti_variables["t"], self.return_variable("t"))

        # opti.set_initial(opti_variables["s"], self.return_variable("s"))
        opti.set_initial(opti_variables["s_dot"], self.return_variable("s_dot"))
        opti.set_initial(
            opti_variables["input_steering"], self.return_variable("input_steering")
        )
        opti.set_initial(
            opti_variables["tension_tether_ground"],
            self.return_variable("tension_tether_ground"),
        )

        opti.subject_to(opti_variables["input_steering"] <= 1)
        opti.subject_to(opti_variables["input_steering"] >= -1)
        opti.subject_to(opti_variables["tension_tether_ground"] <= 1e8)
        opti.subject_to(opti_variables["tension_tether_ground"] >= 100)
        # opti.set_initial(path_angle, self.return_variable("s"))

        # Set initial conditions for optimization parameters
        for var in self.optimization_vars:
            print(
                f"Setting initial value for {var}: {self.pattern_config['parameters'][var]}"
            )
            opti.set_initial(
                self.optimization_vars[var], self.pattern_config["parameters"][var]
            )

        # opti.subject_to(time[:]>=0)
        # opti.subject_to(power >= 0)  # Power must be positive
        opti.minimize(-power)  # Minimize negative power

        opti.solver(
            "ipopt",
            {
                "ipopt": {
                    "max_iter": 100,
                    "bound_relax_factor": 0,
                    "tol": 1e-3,  # Main tolerance
                    "acceptable_iter": 3,  # Accept if solution is good for 3 iter
                    "acceptable_tol": 1e-5,  # Acceptable early termination
                    "constr_viol_tol": 1e-6,  # Constraint violation tolerance
                    "dual_inf_tol": 1e-6,  # Dual infeasibility}#,"mu_init": 1e-2},
                }
                # "max_iter": 500,
                # "bound_relax_factor": 0,  # <--- critical
                # "mu_init": 1e-2,
                # "acceptable_tol": 1e-4
            },
        )

        # opti.solver("fatrop")
        # sol = opti.solve()
        # Reset kite_model

        try:
            sol = opti.solve()
            print("\n Solution found!")

            # Print optimized values for variables in the pattern
            print("\n Optimized Pattern Variables:")
            for var_name, var in self.optimization_vars.items():
                print(f"  {var_name}: {opti.debug.value(var):.6f}")
                optimized_config = self.pattern_config.copy()
                optimized_config["parameters"].update({var_name: opti.debug.value(var)})
                self.pattern_config = optimized_config
                # self.pattern_config["parameters"].update({var_name: opti.debug.value(var)})
            import matplotlib.pyplot as plt

            plt.plot(sol.value(time), sol.value(opti_variables["s"]), label="s")
            plt.plot(
                self.return_variable("t"), self.return_variable("s"), label="s_initial"
            )
            plt.xlabel("Time (s)")
            plt.ylabel("s (m)")
            plt.show()

            plt.figure()
            plt.plot(sol.value(time), sol.value(opti_variables["s_dot"]), label="s_dot")
            plt.plot(
                self.return_variable("t"),
                self.return_variable("s_dot"),
                label="s_dot_initial",
            )
            plt.show()

            # Print the final power value
            final_power = sol.value(power)
            print(f"\n Final Power: {final_power:.6f}")

            # self.run_simulation(start_state, s_array= s_array, time_array= time_array)

        except RuntimeError as e:
            print("\n Solver failed with error:", e)

            # Debugging information
            print("\n Debugging Variables:")

            # Print current values of optimization variables
            # print(self.optimization_vars)
            for var_name, var in self.optimization_vars.items():

                try:
                    print(f"  {var_name}: {opti.debug.value(var):.6f}")
                    self.pattern_config["parameters"].update(
                        {var_name: opti.debug.value(var)}
                    )
                    # self.run_simulation(start_state, s_array, time_array)
                except:
                    print(f"  {var_name}: (No value available)")
            return self.pattern_config
            # Print the last power computation attempt
            try:
                print(
                    f"\n Power Value (last computed before failure): {opti.debug.value(power):.6f}"
                )
            except:
                print("\n Power Value: (No value available)")

    def substitute_parametrized_kinematics(self, optimize=False):

        pattern = create_pattern_from_dict(self.pattern_config, optimize=optimize)
        kinematics = ParametrizedKinematics(pattern, self.quasi_steady)

        self.kite_model.s = kinematics.s
        self.kite_model.t = kinematics.t
        self.kite_model.s_dot = kinematics.s_dot
        self.kite_model.s_ddot = kinematics.s_ddot

        self.kite_model.distance_radial = kinematics.r
        self.kite_model.angle_course = kinematics.chi
        self.kite_model.angle_elevation = kinematics.beta
        # Optimal analytical solution for speed_radial should be part of the pattern class
        # self.kite_model.speed_radial = self.kite_model.speed_radial
        # print(self.kite_model.speed_radial)
        self.kite_model.speed_radial = kinematics.vr
        self.kite_model.speed_tangential = kinematics.vtau
        self.kite_model.timeder_angle_course = kinematics.dot_chi
        self.kite_model.timeder_speed_radial = kinematics.dot_vr
        self.kite_model.timeder_speed_tangential = kinematics.dot_vtau

        self.kite_model.angle_azimuth = kinematics.phi
        self.kite_model.angle_elevation = kinematics.beta

        if optimize:
            return list(kinematics.pattern.optimization_vars.values())

    @property
    def target_lift_coefficient(self):
        return self._target_lift_coefficient

    @target_lift_coefficient.setter
    def target_lift_coefficient(self, value):
        self._target_lift_coefficient = value

    @property
    def target_drag_coefficient(self):
        return self._target_drag_coefficient

    @target_drag_coefficient.setter
    def target_drag_coefficient(self, value):
        self._target_drag_coefficient = value

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
    #     copy_kite.angle_pitch_tether = ca.SX.sym("angle_pitch_tether")
    #     copy_kite.speed_tangential = ca.SX.sym("speed_tangential")
    #     # copy_kite.aero_input["dependencies"]["u_s"] = {}
    #     # copy_kite.input_steering = 0
    #     # copy_kite.speed_radial = ca.SX.sym("speed_radial")
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

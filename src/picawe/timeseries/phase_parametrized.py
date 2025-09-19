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
        sharpness_beta: float = 1e-4,
        tension_min: float = 0.0,
        tension_max: float = 1e5,
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
        self.sharpness_beta = sharpness_beta
        self.tension_min = tension_min
        self.tension_max = tension_max

        # self.find_optimal_angle_pitch_tether()

    def run_simulation(self, start_state, allow_failure=True, return_states=False):

        # print("Starting state:", start_state)
        km_copy = self.substitute_parametrized_kinematics()
        self.states = []
        km_copy.reset_solver()
        self.km_param = km_copy

        if self.quasi_steady:
            unknown_vars = ["length_tether", "input_steering", "s_dot"]
        else:
            unknown_vars = ["length_tether", "input_steering", "s_ddot"]

        if km_copy.is_tether_rigid:
            unknown_vars[0] = "tension_tether_ground"
        # Initialize state
        if isinstance(start_state, dict):
            state_obj = State(**start_state)
        else:
            state_obj = start_state

        N = self.pattern_config["n_points"]
        time_step = self.pattern_config["end_time"] / self.pattern_config["n_points"]
        intg = self.integrator(time_step=time_step, kite_model=km_copy)
        qs_solver = self.residual_solver(km_copy)

        # print("New state:", qs_solver)
        if self.quasi_steady:
            x0 = ca.vertcat(
                state_obj.tension_tether_ground,
                state_obj.input_steering,
                state_obj.s_dot,
                state_obj.speed_radial,
            )
            p = ca.vertcat(state_obj.s, state_obj.distance_radial)
            lbx, ubx, lbg, ubg = km_copy.get_boundaries(state_obj, unknown_vars)
            lbx.append(0)
            ubx.append(10)
            lbg.append(0)
            ubg.append(0)
            sol = qs_solver(x0=x0, p=p, lbg=lbg, ubg=ubg, lbx=lbx, ubx=ubx)
            x0 = p
            z0 = sol["x"]
        else:
            x0 = ca.vertcat(
                state_obj.tension_tether_ground,
                state_obj.input_steering,
                state_obj.s_dot,
                state_obj.speed_radial,
            )
            p = ca.vertcat(state_obj.s, state_obj.s_dot, state_obj.distance_radial)
            lbx, ubx, lbg, ubg = km_copy.get_boundaries(state_obj, unknown_vars)
            lbx.append(0)
            ubx.append(10)
            lbg.append(0)
            ubg.append(0)
            sol = qs_solver(x0=x0, p=p, lbg=lbg, ubg=ubg, lbx=lbx, ubx=ubx)
            x0 = p
            z0 = sol["x"]
        # self.states.append(new_state.to_dict())
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

    def run_simulation_phase(
        self, start_state, allow_failure=True, return_states=False
    ):
        """
        March along an s-grid. At each grid point:
        - solve residuals for unknowns (z)
        - record state at current (t, s_i)
        - if not last grid point, compute dt from ds, v, a and advance x, t.

        Conventions:
        QS   : a_s = 0  -> ds = v_s * dt
        Dyn  : ds = v_s * dt + 0.5 * a_s * dt^2  (stable quadratic root used)
        """

        # --- setup / housekeeping
        self.kite_model.reset_solver()
        km_copy = self.substitute_parametrized_kinematics()
        self.km_param = km_copy
        self.states = []

        # unknowns to solve at each s-node
        if self.quasi_steady:
            unknown_vars = ["length_tether", "input_steering", "s_dot"]
        else:
            unknown_vars = ["length_tether", "input_steering", "s_ddot"]

        if km_copy.is_tether_rigid:
            unknown_vars[0] = "tension_tether_ground"

        # initial state object
        state_obj = (
            State(**start_state) if isinstance(start_state, dict) else start_state
        )

        # grid and solver
        N = int(self.pattern_config["n_points"])
        s_grid = np.linspace(
            self.pattern_config["start_angle"],
            self.pattern_config["end_angle"],
            N + 1,
        )
        qs_solver = self.residual_solver(km_copy)

        # pack initial guesses / states
        if self.quasi_steady:
            # z = [tension_tether_ground, input_steering, s_dot, speed_radial]
            z = ca.vertcat(
                state_obj.tension_tether_ground,
                state_obj.input_steering,
                state_obj.s_dot,
                state_obj.speed_radial,
            )
            # x = [s, distance_radial]
            x = ca.vertcat(s_grid[0], state_obj.distance_radial)
        else:
            # z = [tension_tether_ground, input_steering, s_ddot, speed_radial]
            z = ca.vertcat(
                state_obj.tension_tether_ground,
                state_obj.input_steering,
                0.01,  # initial guess for s_ddot
                state_obj.speed_radial,
            )
            # x = [s, s_dot, distance_radial]
            x = ca.vertcat(s_grid[0], state_obj.s_dot, state_obj.distance_radial)

        lbx, ubx, lbg, ubg = self.get_boundaries(state_obj, unknown_vars, km_copy)
        t = float(self.pattern_config.get("start_time", 0.0))

        # --- helper: stable Δt from ds, v, a  (ds = v*dt + 0.5*a*dt^2)
        def _dt_from_ds_v_a(ds_scalar, v_s, a_s):
            """
            Numerically stable positive root:
                dt = 2*ds / ( v + sqrt(v*v + 2*a*ds) )
            - uses CasADi ops so it works with DM/MX/SX.
            - if discriminant < 0: clip to 0 if allow_failure else raise.
            """
            disc = v_s * v_s + 2.0 * a_s * ds_scalar
            if allow_failure:
                disc = ca.fmax(
                    disc, 0.0
                )  # clip; produces the limiting solution if negative
            else:
                # optional hard check
                if isinstance(disc, (float, int)) and disc < 0:
                    raise ValueError(f"Negative discriminant: v^2+2*a*ds={disc}")
            denom = v_s + ca.sqrt(disc)
            # add tiny epsilon to avoid divide-by-zero when v≈0 and a→0
            return 2.0 * ds_scalar / (denom + 1e-12)

        # --- main loop
        for i in range(N):
            # 1) solve residuals at current s-grid node
            sol = qs_solver(x0=z, p=x, lbg=lbg, ubg=ubg, lbx=lbx, ubx=ubx)
            z = sol["x"]  # CasADi DM

            # 2) record current state (BEFORE stepping to next s)
            if self.quasi_steady:
                curr_state = State(
                    t=t,
                    s=float(x[0]),
                    input_steering=float(z[1]),
                    tension_tether_ground=float(z[0]),
                    s_dot=float(z[2]),
                    distance_radial=float(x[1]),
                    speed_radial=float(z[3]),
                )
            else:
                curr_state = State(
                    t=t,
                    s=float(x[0]),
                    s_dot=float(x[1]),
                    input_steering=float(z[1]),
                    tension_tether_ground=float(z[0]),
                    s_ddot=float(z[2]),
                    distance_radial=float(x[2]),
                    speed_radial=float(z[3]),
                )
            self.states.append(curr_state.to_dict())

            # 4) step to next s using appropriate time increment
            ds = float(s_grid[i + 1] - s_grid[i])  # scalar number

            if self.quasi_steady:
                # a_s = 0 => dt = ds / v_s
                v_s = z[2]  # s_dot from QS solve
                dt = ds / (v_s + 1e-12)  # small epsilon to avoid division by zero
                next_r = x[1] + z[3] * dt
                x = ca.vertcat(s_grid[i + 1], next_r)
            else:
                # dynamic: ds = v*dt + 0.5*a*dt^2
                v_s = x[1]  # current s_dot (state)
                a_s = z[2]  # current s_ddot (solve result)
                dt = _dt_from_ds_v_a(ds, v_s, a_s)

                next_s_dot = v_s + a_s * dt
                next_r = x[2] + z[3] * dt
                x = ca.vertcat(s_grid[i + 1], next_s_dot, next_r)

            # 5) advance time (dt is a CasADi scalar DM; cast to float)
            t += float(dt)

        print("Total time:", t)
        return self.states if return_states else None

    def run_simulation_euler(
        self, start_state, allow_failure=True, return_states=False
    ):

        # print("Starting state:", start_state)
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
        qs_solver = self.residual_solver()

        if self.quasi_steady:
            z0 = ca.vertcat(
                state_obj.tension_tether_ground,
                state_obj.input_steering,
                state_obj.s_dot,
                state_obj.speed_radial,
            )
            x0 = ca.vertcat(state_obj.s, state_obj.distance_radial)
        else:
            z0 = ca.vertcat(
                state_obj.tension_tether_ground,
                state_obj.input_steering,
                0,
                state_obj.speed_radial,
            )
            x0 = ca.vertcat(
                state_obj.s,
                state_obj.s_dot,
                state_obj.distance_radial,
            )
        lbx, ubx, lbg, ubg = self.get_boundaries(state_obj, unknown_vars)

        t = self.pattern_config["start_time"]
        for i in range(N):
            # print(f"Time: {t}, State: {x0}, Inputs: {z0}")

            if self.quasi_steady:
                sol = qs_solver(x0=z0, p=x0, lbg=lbg, ubg=ubg, lbx=lbx, ubx=ubx)
                z0 = sol["x"]
                new_s = x0[0] + z0[2] * time_step
                new_r = x0[1] + z0[3] * time_step
                x0 = ca.vertcat(new_s, new_r)
            else:
                sol = qs_solver(x0=z0, p=x0, lbg=lbg, ubg=ubg, lbx=lbx, ubx=ubx)
                z0 = sol["x"]
                new_s = x0[0] + x0[1] * time_step
                new_s_dot = x0[1] + z0[2] * time_step
                new_r = x0[2] + z0[3] * time_step
                x0 = ca.vertcat(new_s, new_s_dot, new_r)
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
            # If the optimization variable is a vector, set as a list of variables
            opt_var = self.optimization_vars[var]
            setattr(pattern, var, opt_var)
            # print(getattr(pattern, var))

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

        if not self.quasi_steady:
            opti.subject_to(
                opti_variables["s_dot"][0] == self.return_variable("s_dot")[0]
            )
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
        height = pattern.z(opti_variables["distance_radial"], opti_variables["s"])
        opti.subject_to(height >= 50)
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

    def run_simulation_opti_phase(self, start_state):
        import casadi as ca, numpy as np

        self.states = []
        self.kite_model.reset_solver()

        N = int(self.pattern_config["n_points"])
        s_grid = np.linspace(
            self.pattern_config["start_angle"], self.pattern_config["end_angle"], N + 1
        )

        # warm-start with a simulated trajectory (QS uses dt = ds / s_dot)
        self.run_simulation_phase(start_state, return_states=True)
        pattern_inputs, pattern, km_copy = self.substitute_parametrized_kinematics(True)
        self.km_param = km_copy
        opti = ca.Opti()
        self.optimization_vars = {}

        # --- Optimization/design parameters
        for var in self.pattern_config["optimization_parameters"]:
            val = np.atleast_1d(self.pattern_config["parameters"][var])
            self.optimization_vars[var] = (
                opti.variable(len(val)) if len(val) > 1 else opti.variable()
            )
            setattr(pattern, var, self.optimization_vars[var])

        # k_vr effective = base * factor (if factor is optimized)
        kvr_base = float(self.pattern_config["parameters"].get("k_vr", 1.0))
        if "k_vr" in self.optimization_vars:
            kvr_eff = self.optimization_vars["k_vr"] * kvr_base
            pattern_inputs.append(self.optimization_vars["k_vr"])
        else:
            kvr_eff = kvr_base

        # --- Decision variables per node (N nodes for intervals 0..N-1)
        opti_vars = {
            "s_dot": opti.variable(N),  # tangential speed
            "input_steering": opti.variable(N),
            "speed_radial": opti.variable(N),  # reel speed v_r
            "distance_radial": opti.variable(N),  # radius r
            "tension_tether_ground": opti.variable(N),  # tether tension T
        }
        # expose design params too
        for var in self.optimization_vars:
            opti_vars[var] = self.optimization_vars[var]

        # Warm starts from simulation
        opti.set_initial(opti_vars["s_dot"], self.return_variable("s_dot"))
        opti.set_initial(
            opti_vars["input_steering"], self.return_variable("input_steering")
        )
        opti.set_initial(
            opti_vars["speed_radial"], self.return_variable("speed_radial")
        )
        opti.set_initial(
            opti_vars["distance_radial"], self.return_variable("distance_radial")
        )
        opti.set_initial(
            opti_vars["tension_tether_ground"],
            self.return_variable("tension_tether_ground"),
        )
        # Fix initial radius
        opti.subject_to(
            opti_vars["distance_radial"][0]
            == self.return_variable("distance_radial")[0]
        )

        # Build model functions
        km_copy.establish_residual()
        flat = [ca.vertcat(*pattern_inputs)]

        residual = ca.Function(
            "residual",
            [
                self.s,
                self.s_dot,
                km_copy.input_steering,
                km_copy.tension_tether_ground,
                km_copy.speed_radial,
                km_copy.distance_radial,
            ]
            + flat,
            [km_copy.residual],
        )
        tether_tension_eq = ca.Function(
            "tether_tension_eq",
            [
                self.s,
                self.s_dot,
                km_copy.input_steering,
                km_copy.speed_radial,
                km_copy.distance_radial,
                km_copy.tension_tether_ground,
            ]
            + flat,
            [km_copy.tension_tether_equation],
        )

        # Safety / geometry constraint
        height = pattern.z(opti_vars["distance_radial"], s_grid[:-1])  # N entries
        opti.subject_to(height >= 50)

        # --- Power scale based on the simulated trajectory (LEFT RULE, consistent)
        P_scale = opti.parameter()
        t_hist = self.return_variable("t")  # length N (QS) or N+1
        P_hist = self.return_variable("mechanical_power")  # same length
        dt_hist = np.diff(t_hist)  # length N-1
        E0 = float(np.sum(P_hist[:-1] * dt_hist))  # left Riemann sum
        T0 = float(np.sum(dt_hist))
        P0 = E0 / (T0 + 1e-12)
        opti.set_value(P_scale, max(abs(P0), 1.0))
        print(f"Initial P0: {P0}")
        # Helpful bounds to keep NLP well-posed
        sdot_min = 1e-2  # ensures dt>0
        opti.subject_to(opti_vars["s_dot"] >= sdot_min)
        if "speed_radial" in DEFAULT_OPTI_LIMITS:
            lb, ub = DEFAULT_OPTI_LIMITS["speed_radial"]
            opti.subject_to(opti_vars["speed_radial"] >= lb)
            opti.subject_to(opti_vars["speed_radial"] <= ub)
        if "distance_radial" in DEFAULT_OPTI_LIMITS:
            lb, ub = DEFAULT_OPTI_LIMITS["distance_radial"]
            opti.subject_to(opti_vars["distance_radial"] >= lb)
            opti.subject_to(opti_vars["distance_radial"] <= ub)

        # --- Objective assembly with SAME quadrature as simulation (left rule)
        energy = 0
        t_eff = 0
        T_min = self.tension_min
        T_max = self.tension_max

        for i in range(N):
            # Current parameter pack (MX)
            opt_par_values = [opti_vars[var] for var in self.optimization_vars]
            flat = [ca.vertcat(*opt_par_values)]

            # Model tension at node i
            T_i = tether_tension_eq(
                s_grid[i],
                opti_vars["s_dot"][i],
                opti_vars["input_steering"][i],
                opti_vars["speed_radial"][i],
                opti_vars["distance_radial"][i],
                opti_vars["tension_tether_ground"][i],
                *flat,
            )

            T_model = T_min + kvr_eff * opti_vars["speed_radial"][i] ** 2
            beta = self.sharpness_beta
            softplus = (1 / beta) * ca.log(1 + ca.exp(beta * (T_model - T_max)))
            # Physics tie: v_r^2 * kvr_eff == T_i   (implicit coupling via T_i(v_r))
            opti.subject_to(T_i == T_model - softplus)

            # Residual equations (unscaled)
            res_i = residual(
                s_grid[i],
                opti_vars["s_dot"][i],
                opti_vars["input_steering"][i],
                T_i,
                opti_vars["speed_radial"][i],
                opti_vars["distance_radial"][i],
                *flat,
            )
            opti.subject_to(res_i[0] == 0)
            opti.subject_to(res_i[1] == 0)
            opti.subject_to(res_i[2] == 0)

            # Left-rule dt_i = Δs_i / s_dot[i]
            if i < N - 1:
                ds_i = s_grid[i + 1] - s_grid[i]
                dt_i = ds_i / (opti_vars["s_dot"][i] + 1e-12)

                # r_{i+1} propagation
                opti.subject_to(
                    opti_vars["distance_radial"][i + 1]
                    == opti_vars["distance_radial"][i]
                    + opti_vars["speed_radial"][i] * dt_i
                )

                # Accumulate energy and time: power_i = T_i * v_r_i
                energy += T_i * opti_vars["speed_radial"][i] * dt_i
                t_eff += dt_i

        power = energy / (t_eff + 1e-12)
        opti.minimize(-power / P_scale)

        # --- Solver
        opti.solver(
            "ipopt",
            {
                "ipopt": {
                    "bound_relax_factor": 1e-8,
                    "tol": 1e-4,
                    "acceptable_iter": 3,
                    "acceptable_tol": 1e-4,
                    "constr_viol_tol": 1e-4,
                    "dual_inf_tol": 1e-4,
                    "hessian_approximation": "limited-memory",
                    "mu_strategy": "adaptive",
                }
            },
        )

        # Initials for optimization parameters
        for var, mx in self.optimization_vars.items():
            init_val = 1 if var == "k_vr" else self.pattern_config["parameters"][var]
            opti.set_initial(mx, init_val)

        # Default limits for vector vars (if provided)
        for var_name, mx in opti_vars.items():
            if isinstance(mx, ca.MX) and var_name in DEFAULT_OPTI_LIMITS:
                print(f"Applying constraints for {var_name}")
                lb, ub = DEFAULT_OPTI_LIMITS[var_name]
                # vector decisions:
                if mx.shape[0] == N:
                    opti.subject_to(lb <= mx[:])
                    opti.subject_to(mx[:] <= ub)
                else:
                    opti.subject_to(lb <= mx)
                    opti.subject_to(mx <= ub)

        try:
            solution = opti.solve()
            print("Optimized average power:", solution.value(power))
            print("\nOptimized Pattern Variables:")
            for var_name, mx in self.optimization_vars.items():
                val = solution.value(mx)
                print(f"  {var_name}: {val}")

                # write back optimized parameters
                optimized_config = self.pattern_config.copy()
                if var_name == "k_vr":
                    optimized_config["parameters"]["k_vr"] = (
                        float(solution.value(mx)) * kvr_base
                    )
                else:
                    optimized_config["parameters"][var_name] = solution.value(mx)
                self.pattern_config = optimized_config
                # self.substitute_parametrized_kinematics()
        except Exception as e:
            print("Debug optimization information:")
            for var_name, mx in self.optimization_vars.items():
                try:
                    print(f"  {var_name}: {opti.debug.value(mx)}")
                except Exception:
                    pass
            print("Optimization failed:", e)

    def substitute_parametrized_kinematics(self, optimize=False):

        pattern = create_pattern_from_dict(self.pattern_config, optimize=optimize)

        kinematics = ParametrizedKinematics(pattern, self)

        km_copy = copy.deepcopy(self.kite_model)

        km_copy.angle_course = kinematics.chi
        # Optimal analytical solution for speed_radial should be part of the pattern class
        # km_copy.speed_radial = km_copy.speed_radial
        # print(km_copy.speed_radial)
        # km_copy.speed_radial = kinematics.vr
        km_copy.speed_tangential = kinematics.vtau
        km_copy.timeder_angle_course = kinematics.dot_chi
        if not self.quasi_steady:
            km_copy.timeder_speed_radial = kinematics.dot_vr
            km_copy.timeder_speed_tangential = kinematics.dot_vtau
        else:
            km_copy.timeder_speed_radial = 0
            km_copy.timeder_speed_tangential = 0

        km_copy.angle_azimuth = kinematics.phi
        km_copy.angle_elevation = kinematics.beta

        if optimize:
            return list(kinematics.pattern.optimization_vars.values()), pattern, km_copy
        else:
            return km_copy

    def integrator(self, time_step, kite_model=None):
        if kite_model is None:
            kite_model = self.kite_model
        kite_model.establish_residual()
        k_vr = self.pattern_config["parameters"]["k_vr"]
        T_min = self.tension_min
        T_max = self.tension_max
        if self.quasi_steady:
            x = ca.vertcat(self.s, kite_model.distance_radial)
            if kite_model.is_tether_rigid:
                z = ca.vertcat(
                    kite_model.tension_tether_ground,
                    kite_model.input_steering,
                    self.s_dot,
                    kite_model.speed_radial,
                )
            else:
                z = ca.vertcat(
                    kite_model.length_tether,
                    kite_model.input_steering,
                    self.s_dot,
                    kite_model.speed_radial,
                )

            ode = ca.vertcat(
                self.s_dot,
                kite_model.speed_radial,
            )
            T_model = T_min + k_vr * kite_model.speed_radial**2
            softplus = (1 / self.sharpness_beta) * ca.log(
                1 + ca.exp(self.sharpness_beta * (T_model - T_max))
            )
            alg = ca.vertcat(
                kite_model.residual,
                kite_model.tension_tether_ground - T_model + softplus,
            )

        else:
            x = ca.vertcat(
                self.s,
                self.s_dot,
                kite_model.distance_radial,
            )
            if kite_model.is_tether_rigid:
                z = ca.vertcat(
                    kite_model.tension_tether_ground,
                    kite_model.input_steering,
                    self.s_ddot,
                    kite_model.speed_radial,
                )
            else:
                z = ca.vertcat(
                    kite_model.length_tether,
                    kite_model.input_steering,
                    self.s_ddot,
                    kite_model.speed_radial,
                )

            ode = ca.vertcat(
                self.s_dot,
                self.s_ddot,
                kite_model.speed_radial,
            )
            T_model = T_min + k_vr * kite_model.speed_radial**2
            softplus = (1 / self.sharpness_beta) * ca.log(
                1 + ca.exp(self.sharpness_beta * (T_model - T_max))
            )
            alg = ca.vertcat(
                kite_model.residual,
                kite_model.tension_tether_ground - T_model + softplus,
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

    def residual_solver(self, km_copy=None):
        if km_copy is None:
            km_copy = self.kite_model

        km_copy.establish_residual()
        k_vr = self.pattern_config["parameters"]["k_vr"]
        T_min = self.tension_min
        T_max = self.tension_max
        if self.quasi_steady:
            if km_copy.is_tether_rigid:
                z = ca.vertcat(
                    km_copy.tension_tether_ground,
                    km_copy.input_steering,
                    self.s_dot,
                    km_copy.speed_radial,
                )
            else:
                z = ca.vertcat(
                    km_copy.length_tether,
                    km_copy.input_steering,
                    km_copy.s_dot,
                    km_copy.speed_radial,
                )
            p = ca.vertcat(
                self.s,
                km_copy.distance_radial,
            )

            T_model = T_min + k_vr * km_copy.speed_radial**2
            softplus = (1 / self.sharpness_beta) * ca.log(
                1 + ca.exp(self.sharpness_beta * (T_model - T_max))
            )
            alg = ca.vertcat(
                km_copy.residual,
                km_copy.tension_tether_ground - T_model + softplus,
            )
        else:
            if km_copy.is_tether_rigid:
                z = ca.vertcat(
                    km_copy.tension_tether_ground,
                    km_copy.input_steering,
                    self.s_ddot,
                    km_copy.speed_radial,
                )
            else:
                z = ca.vertcat(
                    km_copy.length_tether,
                    km_copy.input_steering,
                    self.s_ddot,
                    km_copy.speed_radial,
                )
            T_model = T_min + k_vr * km_copy.speed_radial**2
            softplus = (1 / self.sharpness_beta) * ca.log(
                1 + ca.exp(self.sharpness_beta * (T_model - T_max))
            )
            alg = ca.vertcat(
                km_copy.residual,
                km_copy.tension_tether_ground - T_model + softplus,
            )
            p = ca.vertcat(
                self.s,
                self.s_dot,
                km_copy.distance_radial,
            )
        nlp = {
            "x": z,
            "f": 0,
            "g": alg,
            "p": p,
        }
        solver_options = {
            "ipopt": {
                "print_level": 0,  # Suppresses IPOPT output
                "max_iter": 200,  # Maximum number of iterations
                "sb": "yes",  # Suppresses more detailed solver information
            },
            "print_time": False,  # Disables CasADi's internal timing output
        }
        return ca.nlpsol("solver", "ipopt", nlp, solver_options)

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

    def get_boundaries(self, state_obj, unknown_vars, km_copy):
        lbx, ubx, lbg, ubg = km_copy.get_boundaries(state_obj, unknown_vars)
        lbx.append(0)
        ubx.append(10)
        lbg.append(0)
        ubg.append(0)
        return lbx, ubx, lbg, ubg

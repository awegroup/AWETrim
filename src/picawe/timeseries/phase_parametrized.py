from picawe.timeseries.timeseries import TimeSeries
from picawe.kinematics.parametrized_patterns import  create_pattern_from_dict
from picawe import SystemModel
from picawe.kinematics.Kinematics import ParametrizedKinematics
import casadi as ca
import numpy as np
from picawe.utils.defaults import DEFAULT_PATTERN_CONFIG, DEFAULT_OPTI_LIMITS
import copy
from picawe.system.tether import RigidLinkTether

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
        pattern = create_pattern_from_dict(self.pattern_config, optimize=True)
        self.kinematics = ParametrizedKinematics(pattern, self.quasi_steady)
        self.find_optimal_angle_pitch_tether()
        
    
    def run_simulation(self,start_state, s_array = None, time_array = None):
        self.substitute_parametrized_kinematics()
        self.states = []
        if self.quasi_steady:
            unknown_vars = ["length_tether", "input_steering", "s_dot"]
        else:
            unknown_vars = ["length_tether", "input_steering", "s_ddot"]

        if self.kite_model.dof == 6:
            unknown_vars += ["angle_roll", "angle_pitch", "angle_yaw"]
        for var in unknown_vars:
            if var not in start_state:
                raise ValueError(f"Start state must contain {var}")
        qs_guess = [start_state[name] for name in unknown_vars]
        solve_func, inputs_name = self.kite_model.setup_qs_solver(
            unknown_vars)
        current_state = start_state
        speed_radial_func = self.kite_model.extract_function("speed_radial")
        input_length = speed_radial_func.n_in()
        distance_radial = self.kinematics.r
        # TODO: Implement the s array
        if time_array is not None:
            s = start_state["s"]
            s_dot = start_state["s_dot"]
            for i in range(len(time_array)):
                # print(i)
                p = [current_state[name] for name in inputs_name]
                lbx,ubx,lbg,ubg = self.kite_model.get_boundaries(unknown_vars)
                sol = solve_func(x0=qs_guess, p=p, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
                if sol["g"][0] < 1e-3:
                    # print('Sol found')
                    qs_guess = sol["x"]
                    qs_state = {name: float(sol["x"][i]) for i, name in enumerate(unknown_vars)}
                    if i < len(time_array)-1:
                        time_step = time_array[i+1]-time_array[i]

                    if self.quasi_steady:
                        s += float(sol["x"][2])*time_step
                        current_state = {**qs_state, "s": s, "t": time_array[i], "s_dot": float(sol["x"][2]), "s_ddot": 0}
                    else:
                        dyn_state = {name: float(sol["x"][i]) for i, name in enumerate(unknown_vars)}
                        s_dot += float(sol["x"][2])*time_step
                        s += s_dot*time_step
                        current_state = {**dyn_state, "s_dot": s_dot, "s": s, "t": time_array[i], "s_ddot": float(sol["x"][2])}
                    
                    if input_length >0:
                        distance_radial +=  speed_radial_func(current_state["tension_tether_ground"])*time_step
                        current_state["distance_radial"] = distance_radial
                    self.states.append(current_state)
                else:
                    print("Warning: Solver did not converge")

        
        if s_array is not None:
            t = start_state["t"]
            s_dot = start_state["s_dot"]
            for i in range(len(s_array)):
                # print(i)
                p = [current_state[name] for name in inputs_name]
                lbx,ubx,lbg,ubg = self.kite_model.get_boundaries(unknown_vars)
                sol = solve_func(x0=qs_guess, p=p, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
                if sol["g"][0] < 1e-3:
                    # print('Sol found')
                    qs_guess = sol["x"]
                    qs_state = {name: float(sol["x"][i]) for i, name in enumerate(unknown_vars)}

                    if self.quasi_steady:
                        if i < len(s_array)-1:
                            time_step = (s_array[i+1]-s_array[i])/(float(sol["x"][2]))
                        t += time_step
                        current_state = {**qs_state, "s": s_array[i], "t": t,  "s_ddot": 0}
                    else:
                        dyn_state = {name: float(sol["x"][i]) for i, name in enumerate(unknown_vars)}
                            # Calculate timestep based on known s and current s_dot
                        if i < len(s_array)-1:
                            delta_s = s_array[i+1] - s_array[i]
                        time_step = delta_s / s_dot if s_dot != 0 else 0.01  # Avoid division by zero
                        s_dot += float(sol["x"][2])*time_step
                        t += time_step
                        current_state = {**dyn_state, "s_dot": s_dot, "s": s_array[i], "t": t}
                    if input_length >0:
                        distance_radial +=  speed_radial_func(current_state["tension_tether_ground"])*time_step
                        current_state["distance_radial"] = distance_radial
                    self.states.append(current_state)
                else:
                    print("Warning: Solver did not converge")

    def optimize_pattern(self, start_state, s_array=None, time_array=None):
        self.set_optimal_speed_radial()
        self.set_optimal_angle_pitch_tether()
        self.run_simulation(start_state, s_array, time_array)
        self.substitute_optimized_kinematics()
        

        opti = ca.Opti()
        self.optimization_vars = {}  # Store optimization variables

        # Create optimization variables for parameters to optimize
        for var in self.pattern_config["optimization_parameters"]:
            self.optimization_vars[var] = opti.variable()  # No bounds if not specified

        self.kite_model.establish_residual()

        # Retrieve symbolic variables from self.kinematics.pattern
        pattern_inputs = list(self.kinematics.pattern.optimization_vars.values())


        
        
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
            ts = (sf-si)/s_dot_sym
            timestep_func = ca.Function("t_func", [si,sf,s_dot_sym], [ts])

        path_angle_dot = opti.variable(N)
        input_steering = opti.variable(N)
        tension_tether = opti.variable(N)

        # Store optimization variables dynamically
        opti_variables = {
            "t": time,
            "s": path_angle,
            "s_dot": path_angle_dot,
            "input_steering": input_steering,
            "tension_tether_ground": tension_tether,
        }

        # Add optimization parameters
        for var in self.optimization_vars:
            opti_variables[var] = self.optimization_vars[var]

        # Define the residual function
        residual = ca.Function(
            "residual",
            [self.kite_model.t, self.kite_model.s, self.kite_model.s_dot,
            self.kite_model.input_steering, self.kite_model.tension_tether_ground] + pattern_inputs,  
            [self.kite_model.residual]
        )

        # Define angle elevation function
        angle_elevation_fun = ca.Function(
            "angle_elevation",
            [self.kite_model.t, self.kite_model.s] + pattern_inputs,
            [self.kite_model.angle_elevation]
        )
        vr_func = ca.Function(
            "vr",
            [self.kite_model.tension_tether_ground],
            [self.kite_model.speed_radial]
        )

        power = 0  # Initialize power
        angle_elevation = ca.MX.zeros(N)

        for i in range(N):
            # Dynamically pass opti variables into residual function
            residual_inputs = [opti_variables["t"][i], opti_variables["s"][i], opti_variables["s_dot"][i],
                            opti_variables["input_steering"][i], opti_variables["tension_tether_ground"][i]] \
                            + [opti_variables[var] for var in self.optimization_vars]

            opti.subject_to(residual(*residual_inputs) == 0)

            if time_array is not None:

                # Compute power
                time_step = (time_array[i + 1] - time_array[i]) if i < len(time_array) - 1 else 1.0
                
                # Add dynamic constraint on path angle
                if i < len(time_array) - 1:
                    opti.subject_to(path_angle[i + 1] == path_angle[i] + time_step * path_angle_dot[i])

            elif s_array is not None:

                if i < len(s_array)-1:
                    time_step = timestep_func(opti_variables["s"][i], opti_variables["s"][i+1], opti_variables["s_dot"][i])
                    opti.subject_to(time[i+1] == time[i] + time_step)

            power += tension_tether[i] * time_step * vr_func(tension_tether[i])

            # Compute angle elevation
            angle_elevation[i] = angle_elevation_fun(opti_variables["t"][i], opti_variables["s"][i],
                                                    *[opti_variables[var] for var in self.optimization_vars])


        # Normalize power
        power = power/time[-1]

        ### APPLY CONSTRAINTS DYNAMICALLY FROM DEFAULT_OPTI_LIMITS ###
        for var_name, opti_var in opti_variables.items():
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

        #Constraint for angle_elevation
        opti.subject_to(angle_elevation >= DEFAULT_OPTI_LIMITS["angle_elevation"][0])
        opti.subject_to(angle_elevation <= DEFAULT_OPTI_LIMITS["angle_elevation"][1])


        ### APPLY INITIAL VALUES FROM RESULTS ###
        opti.set_initial(opti_variables["t"], self.return_variable("t"))
        # opti.set_initial(opti_variables["s"], self.return_variable("s"))
        opti.set_initial(opti_variables["s_dot"], self.return_variable("s_dot"))
        opti.set_initial(opti_variables["input_steering"], self.return_variable("input_steering"))
        opti.set_initial(opti_variables["tension_tether_ground"], self.return_variable("tension_tether_ground"))

        # Set initial conditions for optimization parameters
        for var in self.optimization_vars:
            opti.set_initial(self.optimization_vars[var], self.pattern_config["initial_parameters"][var])

        opti.subject_to(time[:]>=0)
        opti.subject_to(power >= 0)  # Power must be positive
        opti.minimize(-power)  # Minimize negative power

        opti.solver("ipopt")
        # opti.solver("sqpmethod")
        # sol = opti.solve()
        try:
            sol = opti.solve()
            print("\n Solution found!")

            # Print optimized values for variables in the pattern
            print("\n Optimized Pattern Variables:")
            for var_name, var in self.optimization_vars.items():
                print(f"  {var_name}: {sol.value(var):.6f}")
                optimized_params = self.pattern_config["initial_parameters"]
                optimized_params.update({var_name: sol.value(var)})
                self.pattern_config["initial_parameters"] = optimized_params

            # Print the final power value
            final_power = sol.value(power)
            print(f"\n Final Power: {final_power:.6f}")
            self.run_simulation(start_state, s_array, time_array)
            return self.pattern_config

        except RuntimeError as e:
            print("\n Solver failed with error:", e)

            # Debugging information
            print("\n Debugging Variables:")

            # Print current values of optimization variables
            # print(self.optimization_vars)
            for var_name, var in self.optimization_vars.items():
                
                try:
                    print(f"  {var_name}: {opti.debug.value(var):.6f}")
                    optimized_params = self.pattern_config["initial_parameters"]
                    optimized_params.update({var_name: opti.debug.value(var)})
                    self.pattern_config["initial_parameters"] = optimized_params
                    self.run_simulation(start_state, s_array, time_array)
                except:
                    print(f"  {var_name}: (No value available)")
            return self.pattern_config
            # Print the last power computation attempt
            try:
                print(f"\n Power Value (last computed before failure): {opti.debug.value(power):.6f}")
            except:
                print("\n Power Value: (No value available)")





    


    def substitute_parametrized_kinematics(self):
        pattern = create_pattern_from_dict(self.pattern_config, optimize=False)
        kinematics = ParametrizedKinematics(pattern, self.quasi_steady)
        self.kite_model.angle_course = kinematics.chi
        self.kite_model.angle_elevation = kinematics.beta
        self.kite_model.speed_radial = self.kite_model.speed_radial
        # self.kite_model.speed_radial = kinematics.vr
        self.kite_model.speed_tangential = kinematics.vtau
        self.kite_model.timeder_angle_course = kinematics.dot_chi
        self.kite_model.timeder_speed_radial = kinematics.dot_vr
        self.kite_model.timeder_speed_tangential = kinematics.dot_vtau
        self.kite_model.distance_radial = kinematics.r
        self.kite_model.angle_azimuth = kinematics.phi
        self.kite_model.angle_elevation = kinematics.beta
        self.kite_model.s = kinematics.s
        self.kite_model.t = kinematics.t
        self.kite_model.s_dot = kinematics.s_dot
        self.kite_model.s_ddot = kinematics.s_ddot

    def substitute_optimized_kinematics(self):

        pattern = create_pattern_from_dict(self.pattern_config, optimize=True)
        self.kinematics = ParametrizedKinematics(pattern, self.quasi_steady)
        self.kite_model.angle_course = self.kinematics.chi
        self.kite_model.angle_elevation = self.kinematics.beta
        # Optimal analytical solution for speed_radial should be part of the pattern class
        self.kite_model.speed_radial = self.kite_model.speed_radial
        # self.kite_model.speed_radial = self.kinematics.vr
        self.kite_model.speed_tangential = self.kinematics.vtau
        self.kite_model.timeder_angle_course = self.kinematics.dot_chi
        self.kite_model.timeder_speed_radial = self.kinematics.dot_vr
        self.kite_model.timeder_speed_tangential = self.kinematics.dot_vtau
        self.kite_model.distance_radial = self.kinematics.r
        self.kite_model.angle_azimuth = self.kinematics.phi
        self.kite_model.angle_elevation = self.kinematics.beta
        self.kite_model.s = self.kinematics.s
        self.kite_model.t = self.kinematics.t
        self.kite_model.s_dot = self.kinematics.s_dot
        self.kite_model.s_ddot = self.kinematics.s_ddot

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


    def find_optimal_angle_pitch_tether(self):
        copy_kite = copy.deepcopy(self.kite_model)
        copy_kite.angle_elevation = 0
        copy_kite.angle_azimuth = 0
        copy_kite.angle_course = np.pi/2
        copy_kite.timeder_speed_tangential = 0
        copy_kite.distance_radial = 200
        copy_kite.wind.wind_model = 'uniform'
        copy_kite.wind.speed_wind_ref = 10
        copy_kite.speed_radial = 0
        copy_kite.timeder_angle_course = 0
        # copy_kite.angle_roll = 0
        copy_kite.timeder_speed_radial = 0
        copy_kite.delta_pitch_depower = 0
        copy_kite.input_depower = 0
        copy_kite.tether = RigidLinkTether()
        copy_kite.angle_pitch_tether = ca.SX.sym("angle_pitch_tether")
        # copy_kite.aero_input["dependencies"]["u_s"] = {}
        # copy_kite.input_steering = 0
        # copy_kite.speed_radial = ca.SX.sym("speed_radial")
        print(copy_kite.lift_coefficient)
        cl_func = copy_kite.extract_function("lift_coefficient")
        cd_func = copy_kite.extract_function("drag_coefficient")
        aoa_func = copy_kite.extract_function("angle_of_attack")

        copy_kite.establish_residual()
        
        residual = ca.Function("residual", [copy_kite.speed_tangential, copy_kite.input_steering, copy_kite.length_tether, copy_kite.angle_pitch_tether], [copy_kite.residual], ["vtau", "steering", "length_tether", "angle_pitch"], ["residual"])
        print(residual)


        opti = ca.Opti()
        vtau = opti.variable()
        steering = opti.variable()
        lt = opti.variable()
        angle_pitch = opti.variable()
        opti.subject_to(vtau >= 0)
        opti.subject_to(vtau <= 300)
        opti.subject_to(steering >= -np.pi/2)
        opti.subject_to(steering <= np.pi/2)
        opti.subject_to(lt <= copy_kite.distance_radial)
        opti.subject_to(angle_pitch >= np.radians(-5))
        opti.subject_to(angle_pitch <= np.radians(15))
        opti.subject_to(residual(vtau = vtau, steering = steering, length_tether = lt, angle_pitch = angle_pitch)["residual"] == 0)

        opti.set_initial(vtau, 50)
        opti.set_initial(steering, 0)
        opti.set_initial(lt, copy_kite.distance_radial)
        opti.set_initial(angle_pitch, 0)

        opti.minimize(-lt)
        solver_opts = {"ipopt.print_level": 0, "print_time": 0}
        opti.solver("ipopt", solver_opts)
        try:
            sol = opti.solve()
            vtau = sol.value(vtau)
            angle_pitch = sol.value(angle_pitch)
            steering = sol.value(steering)
        except:
            print("Solver failed")
            print(opti.debug.value(vtau))
            print(opti.debug.value(steering))
            print(opti.debug.value(lt))
            print(opti.debug.value(angle_pitch))
        print(cl_func)
        if "u_s" in copy_kite.aero_input.get("dependencies", {}):
            self.target_lift_coefficient = cl_func(speed_tangential=vtau, angle_pitch_tether=angle_pitch, input_steering=steering)["lift_coefficient"]
            self.target_drag_coefficient = cd_func(speed_tangential=vtau, angle_pitch_tether=angle_pitch, input_steering=steering)["drag_coefficient"]
        else:
            self.target_lift_coefficient = cl_func(speed_tangential=vtau, angle_pitch_tether=angle_pitch)["lift_coefficient"]
            self.target_drag_coefficient = cd_func(speed_tangential=vtau, angle_pitch_tether=angle_pitch)["drag_coefficient"]
        self.target_angle_of_attack = aoa_func(speed_tangential=vtau, angle_pitch_tether=angle_pitch)["angle_of_attack"]
        self.optimal_angle_pitch_tether = float(angle_pitch)
        # print(self.target_lift_coefficient, self.target_drag_coefficient, self.target_angle_of_attack*180/np.pi)

    def set_optimal_angle_pitch_tether(self):
        print(f"Angle respect to the tether set to: {np.degrees(self.optimal_angle_pitch_tether)}")
        self.kite_model.angle_pitch_tether = self.optimal_angle_pitch_tether

    def set_optimal_speed_radial(self):
        # if self.target_drag_coefficient is None or self.target_lift_coefficient is None or self.target_angle_of_attack is None: 

        print(f"Optimal speed radial set according to the target  CL: {self.target_lift_coefficient} CD: {self.target_drag_coefficient} at aoa: {np.degrees(self.target_angle_of_attack)}")

        CR_target = ca.sqrt(self.target_lift_coefficient**2 + self.target_drag_coefficient**2)

        # self.kinematics.vr = ca.sqrt(
        #     self.kite_model.tension_tether_ground / (
        #         2 * 1.225 * self.kite_model.area_wing * CR_target * 
        #         (1 + (self.target_lift_coefficient / self.target_drag_coefficient)**2)
        #     )
        # )
        # Calculate the optimal speed_radial
        self.kite_model.speed_radial = ca.sqrt(
            self.kite_model.tension_tether_ground / (
                2 * 1.225 * self.kite_model.area_wing * CR_target * 
                (1 + (self.target_lift_coefficient / self.target_drag_coefficient)**2)
            )
        )
        # print(f"Optimal speed radial set according to the target  CL: {self.target_lift_coefficient} CD: {self.target_drag_coefficient} at aoa: {np.degrees(self.target_angle_of_attack)}")

    